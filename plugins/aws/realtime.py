from __future__ import annotations
import asyncio
import base64
import json
import uuid
from dataclasses import dataclass
from typing import AsyncIterable, Literal, Dict, Any, List

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.function_context import _create_ai_function_info
from plugins.aws._utils import _get_aws_credentials
from log import logger

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from smithy_aws_core.credentials_resolvers.environment import (
    EnvironmentCredentialsResolver,
)


EventTypes = Literal[
    "start_session",
    "input_speech_started",
    "response_content_added",
    "response_content_done",
    "function_calls_collected",
    "function_calls_finished",
    "function_calls_cancelled",
    "input_speech_transcription_completed",
    "agent_speech_transcription_completed",
    "agent_speech_stopped",
]


@dataclass
class NovaContent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str
    audio: list[rtc.AudioFrame]
    text_stream: AsyncIterable[str]
    audio_stream: AsyncIterable[rtc.AudioFrame]
    content_type: Literal["text", "audio"]


@dataclass
class InputTranscription:
    item_id: str
    transcript: str


@dataclass
class Capabilities:
    supports_truncate: bool
    input_audio_sample_rate: int | None = None


@dataclass
class ModelOptions:
    model_id: str
    voice: str
    enable_user_audio_transcription: bool
    enable_agent_audio_transcription: bool
    region: str
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None
    system_prompt: str | None
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    top_k: int | None
    tools: List[Dict[str, Any]] | None


class NovaModel:
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        model_id: str = "amazon.nova-sonic-v1:0",
        voice: str = "matthew",
        enable_user_audio_transcription: bool = True,
        enable_agent_audio_transcription: bool = True,
        region: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = 1024,
        top_p: float | None = 0.9,
        top_k: int | None = None,
        tools: List[Dict[str, Any]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """
        Initialize the NovaModel.

        Args:
            system_prompt: Initial system instructions
            model_id: Nova model ID to use
            voice: Voice ID to use for audio responses
            enable_user_audio_transcription: Whether to transcribe user audio
            enable_agent_audio_transcription: Whether to transcribe agent audio
            region: AWS region
            access_key_id: AWS access key ID (or use environment variables)
            secret_access_key: AWS secret access key (or use environment variables)
            session_token: AWS session token (optional)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling value
            top_k: Top-k sampling value
            tools: List of tool configurations
            loop: Event loop to use
        """
        super().__init__()
        self._capabilities = Capabilities(
            supports_truncate=False,
            input_audio_sample_rate=16000,
        )
        self._model_id = model_id
        self._region = region
        self._loop = loop or asyncio.get_event_loop()
        self._rt_sessions: list[NovaRealtimeSession] = []
        

        self._access_key_id, self._secret_access_key, self._session_token = _get_aws_credentials(
            access_key_id, secret_access_key, region
        )

        # Use provided credentials or check environment variables
        # self._access_key_id = access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        # self._secret_access_key = secret_access_key or os.environ.get(
        #     "AWS_SECRET_ACCESS_KEY"
        # )
        # self._session_token = session_token or os.environ.get("AWS_SESSION_TOKEN")

        if not self._access_key_id or not self._secret_access_key:
            logger.warning(
                "AWS credentials not provided directly or via environment variables"
            )

        self._opts = ModelOptions(
            model_id=model_id,
            voice=voice,
            enable_user_audio_transcription=enable_user_audio_transcription,
            enable_agent_audio_transcription=enable_agent_audio_transcription,
            region=region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            tools=tools,
        )

    @property
    def sessions(self) -> list[NovaRealtimeSession]:
        return self._rt_sessions

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities
    
    def session(
        self,
        *,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
    ) -> NovaRealtimeSession:
        session = NovaRealtimeSession(
            opts=self._opts,
            chat_ctx=chat_ctx or llm.ChatContext(),
            fnc_ctx=fnc_ctx,
            loop=self._loop,
        )
        self._rt_sessions.append(session)
        return session

    async def aclose(self) -> None:
        for session in self._rt_sessions:
            await session.aclose()


class NovaRealtimeSession(utils.EventEmitter[EventTypes]):
    # Event templates
    START_SESSION_EVENT = """{
        "event": {
            "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": %d,
                "topP": %f,
                "temperature": %f
                }
            }
        }
    }"""

    CONTENT_START_EVENT = """{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
                }
            }
        }
    }"""

    AUDIO_EVENT_TEMPLATE = """{
        "event": {
            "audioInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }"""

    TEXT_CONTENT_START_EVENT = """{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "TEXT",
            "role": "%s",
            "interactive": true,
                "textInputConfiguration": {
                    "mediaType": "text/plain"
                }
            }
        }
    }"""

    TEXT_INPUT_EVENT = """{
        "event": {
            "textInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }"""

    TOOL_CONTENT_START_EVENT = """{
        "event": {
            "contentStart": {
                "promptName": "%s",
                "contentName": "%s",
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "%s",
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }"""

    CONTENT_END_EVENT = """{
        "event": {
            "contentEnd": {
            "promptName": "%s",
            "contentName": "%s"
            }
        }
    }"""

    PROMPT_END_EVENT = """{
        "event": {
            "promptEnd": {
            "promptName": "%s"
            }
        }
    }"""

    SESSION_END_EVENT = """{
        "event": {
            "sessionEnd": {}
        }
    }"""

    def __init__(
        self,
        *,
        opts: ModelOptions,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        loop: asyncio.AbstractEventLoop,
    ):
        """
        Initialize a Nova realtime session for streaming interactions.

        Args:
            opts: Model configuration options
            chat_ctx: Chat context for the session
            fnc_ctx: Function context for tools
            loop: Event loop to use
        """
        super().__init__()
        self._loop = loop
        self._opts = opts
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._fnc_tasks = utils.aio.TaskSet()
        self._is_interrupted = False
        self._is_active = False
        self._playout_complete = asyncio.Event()
        self._playout_complete.set()

        # Session information
        self._prompt_name = str(uuid.uuid4())
        self._audio_content_name = str(uuid.uuid4())
        self._content_name = str(uuid.uuid4())
        self._active_response_id = None
        self._toolUseId = None
        self._toolName = None
        self._toolUseContent = None

        # Initialize bedrock client
        self._initialize_client()

        # Initialize queues for data processing
        self._audio_input_queue = asyncio.Queue()
        self._audio_output_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._send_ch = utils.aio.Chan[Dict[str, Any]]()
        self._init_sync_task = asyncio.create_task(asyncio.sleep(0))
        # Initialize session
        self._main_task = asyncio.create_task(
            self._main_task(), name="nova-realtime-session"
        )

    def _initialize_client(self):
        """Initialize the AWS Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._opts.region}.amazonaws.com",
            region=self._opts.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        logger.info("config---Initialize client",config)
        self._client = BedrockRuntimeClient(config=config)

    def _generate_prompt_start(self) -> str:
        """Create the prompt start event with tool configurations."""
        tools = []
        if self._fnc_ctx is not None:
            for func in self._fnc_ctx.ai_functions.values():
                # print(func,"schema")
                print(self._convert_function_to_schema(func))
                schema = json.dumps(self._convert_function_to_schema(func))
                # with open("sample.json", "w") as outfile:
                #     outfile.write(schema)
                tools.append(
                    {
                        "toolSpec": {
                            "name": func.name,
                            "description": func.description,
                            "inputSchema": {"json": schema},
                        }
                    }
                )
                print(tools)

        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self._prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": self._opts.voice,
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                }
            }
        }

        # Add tools if available
        if tools:
            prompt_start_event["event"]["promptStart"]["toolConfiguration"] = {
                "tools": tools
            }

        return json.dumps(prompt_start_event)

    def _convert_function_to_schema(self, func):
        """Convert function definition to JSON schema for Nova."""
        schema = {"type": "object", "properties": {}, "required": []}
        for param in func.arguments.values():
            param_schema = {}
            param_type = param.type
            logger.info(f"param type: {param_type}")

            if issubclass(param_type, int):
                param_schema["type"] = "integer"
            elif issubclass(param_type, str):
                param_schema["type"] = "string"
            elif issubclass(param_type, float):
                param_schema["type"] = "number"
            elif isinstance(param_type, bool):
                issubclass["type"] = "boolean"
            else:
                param_schema["type"] = "unknown"

            if param.description:
                param_schema["description"] = param.description
            if param.default is not True:
                param_schema["default"] =   False

            schema["properties"][param.name] = param_schema
            schema["required"].append(param.name)

        return schema

    def tool_result_event(self, content_name, content):
        """Create a tool result event"""
        if isinstance(content, dict):
            content_json_string = json.dumps(content)
        else:
            content_json_string = content

        tool_result_event = {
            "event": {
                "toolResult": {
                    "promptName": self._prompt_name,
                    "contentName": content_name,
                    "content": content_json_string,
                }
            }
        }
        return json.dumps(tool_result_event)

    @property
    def playout_complete(self) -> asyncio.Event | None:
        return self._playout_complete

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None:
        self._fnc_ctx = value

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return

        self._is_active = False
        self._send_ch.close()
        await self._main_task

    def chat_ctx_copy(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    async def set_chat_ctx(self, ctx: llm.ChatContext) -> None:
        self._chat_ctx = ctx.copy()

    def cancel_response(self) -> None:
        # Nova doesn't support explicit cancellation the same way as Gemini
        # Future implementation could interrupt the current response
        logger.warning("cancel_response is not fully implemented for Nova")

    def create_response(
        self,
        on_duplicate: Literal[
            "cancel_existing", "cancel_new", "keep_both"
        ] = "keep_both",
    ) -> None:
        """Create a response based on the chat context."""
        # Create a text message from the latest user message if present
        latest_msg = None
        for msg in reversed(self._chat_ctx.messages):
            if msg.role == "user" and msg.content:
                latest_msg = msg.content
                break

        if latest_msg:
            content_name = str(uuid.uuid4())
            self._send_ch.send_nowait(
                {
                    "type": "text_content",
                    "content": latest_msg,
                    "content_name": content_name,
                }
            )

    def commit_audio_buffer(self) -> None:
        # Not directly needed for Nova's API
        pass

    def server_vad_enabled(self) -> bool:
        return True

    def _push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio data to the Nova stream."""
        self._audio_input_queue.put_nowait(
            {
                "type": "audio",
                "audio_bytes": frame.data.tobytes(),
                "prompt_name": self._prompt_name,
                "content_name": self._audio_content_name,
            }
        )

    async def _main_task(self):
        """Main task for managing the bidirectional streaming."""
        try:
            # Initialize stream
            self._stream_response = (
                await self._client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(
                        model_id=self._opts.model_id
                    )
                )
            )
            self._is_active = True

            # Start tasks for sending and receiving
            send_task = asyncio.create_task(self._send_task(), name="nova-send-task")
            recv_task = asyncio.create_task(self._recv_task(), name="nova-recv-task")
            process_task = asyncio.create_task(
                self._process_audio_input(), name="nova-process-task"
            )

            # Initialize the session
            await self._initialize_session()

            # Wait for tasks to complete
            await asyncio.gather(
                send_task, recv_task, process_task, return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error in Nova session: {str(e)}")
        finally:
            self._is_active = False

    async def _initialize_session(self):
        """Initialize the session with start events and system prompt."""
        # Format session start event with temperature and max tokens
        max_tokens = self._opts.max_tokens or 1024
        top_p = self._opts.top_p or 0.9
        temperature = self._opts.temperature or 0.7

        session_start = self.START_SESSION_EVENT % (max_tokens, top_p, temperature)
        prompt_start = self._generate_prompt_start()

        # Send initialization events
        await self._send_raw_event(session_start)
        await self._send_raw_event(prompt_start)

        # Send system prompt if provided
        if self._opts.system_prompt:
            system_content_name = str(uuid.uuid4())
            text_content_start = self.TEXT_CONTENT_START_EVENT % (
                self._prompt_name,
                system_content_name,
                "SYSTEM",
            )
            text_content = self.TEXT_INPUT_EVENT % (
                self._prompt_name,
                system_content_name,
                self._opts.system_prompt,
            )
            text_content_end = self.CONTENT_END_EVENT % (
                self._prompt_name,
                system_content_name,
            )
            logger.info(
                f"Sending system prompt: {self._opts.system_prompt}"
            )

            await self._send_raw_event(text_content_start)
            await self._send_raw_event(text_content)
            await self._send_raw_event(text_content_end)

        # Send message history from chat context
        await self._send_chat_history()

    async def _send_chat_history(self):
        """Send message history from chat context."""
        for idx, msg in enumerate(self._chat_ctx.messages):
            if not msg.content:
                continue

            msg_content_name = f"history_{idx}_{str(uuid.uuid4())}"
            role = "USER" if msg.role.upper() == "USER" else "ASSISTANT"

            text_content_start = self.TEXT_CONTENT_START_EVENT % (
                self._prompt_name,
                msg_content_name,
                role,
            )
            text_content = self.TEXT_INPUT_EVENT % (
                self._prompt_name,
                msg_content_name,
                msg.content,
            )
            text_content_end = self.CONTENT_END_EVENT % (
                self._prompt_name,
                msg_content_name,
            )

            await self._send_raw_event(text_content_start)
            await self._send_raw_event(text_content)
            await self._send_raw_event(text_content_end)

    async def _send_raw_event(self, event_json):
        """Send a raw event to the Bedrock stream."""
        if not self._is_active:
            logger.debug("Stream not initialized or closed")
            return

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )

        try:
            await self._stream_response.input_stream.send(event)
            if len(event_json) > 200:
                event_type = json.loads(event_json).get("event", {}).keys()
                logger.debug(f"Sent event type: {list(event_type)}")
            else:
                logger.debug(f"Sent event: {event_json}")
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")

    async def _send_task(self):
        """Task for sending events to Bedrock."""
        try:
            # Process commands from send channel
            async for msg in self._send_ch:
                logger.debug(f"Received send task message: {msg}")
                if msg["type"] == "text_content":
                    logger.debug(f"Received send task message text: {msg}")
                    content_name = msg["content_name"]
                    content = msg["content"]

                    # Send text content
                    text_content_start = self.TEXT_CONTENT_START_EVENT % (
                        self._prompt_name,
                        content_name,
                        "USER",
                    )
                    text_content = self.TEXT_INPUT_EVENT % (
                        self._prompt_name,
                        content_name,
                        content,
                    )
                    text_content_end = self.CONTENT_END_EVENT % (
                        self._prompt_name,
                        content_name,
                    )

                    await self._send_raw_event(text_content_start)
                    await self._send_raw_event(text_content)
                    await self._send_raw_event(text_content_end)
                    await self._send_raw_event(
                        self.PROMPT_END_EVENT % self._prompt_name
                    )

                elif msg["type"] == "audio_content_start":
                    # Start audio streaming
                    content_start = self.CONTENT_START_EVENT % (
                        self._prompt_name,
                        self._audio_content_name,
                    )
                    await self._send_raw_event(content_start)

                elif msg["type"] == "audio_chunk":
                    # Send audio chunk
                    audio_event = self.AUDIO_EVENT_TEMPLATE % (
                        self._prompt_name,
                        self._audio_content_name,
                        msg["content"],
                    )
                    await self._send_raw_event(audio_event)

                elif msg["type"] == "audio_content_end":
                    # End audio content
                    content_end = self.CONTENT_END_EVENT % (
                        self._prompt_name,
                        self._audio_content_name,
                    )
                    await self._send_raw_event(content_end)
                    await self._send_raw_event(
                        self.PROMPT_END_EVENT % self._prompt_name
                    )

                elif msg["type"] == "tool_response":
                    # Send tool response
                    tool_content = msg["content_name"]
                    await self._send_raw_event(
                        self.TOOL_CONTENT_START_EVENT
                        % (self._prompt_name, tool_content, msg["tool_use_id"])
                    )
                    await self._send_raw_event(
                        self.tool_result_event(tool_content, msg["result"])
                    )
                    await self._send_raw_event(
                        self.CONTENT_END_EVENT % (self._prompt_name, tool_content)
                    )

            # End the session when done
            await self._send_raw_event(self.SESSION_END_EVENT)

        except Exception as e:
            logger.error(f"Error in send task: {str(e)}")
        finally:
            # Close the stream
            await self._stream_response.input_stream.close()

    async def _recv_task(self):
        """Task for receiving responses from Bedrock."""
        try:
            text_stream = None
            audio_stream = None

            while self._is_active:
                try:
                    output = await self._stream_response.await_output()
                    result = await output[1].receive()

                    if not result.value or not result.value.bytes_:
                        continue

                    response_data = result.value.bytes_.decode("utf-8")
                    json_data = json.loads(response_data)

                    # Handle different response types
                    if "event" in json_data:
                        event = json_data["event"]

                        if "contentStart" in event:
                            # New content is starting
                            if self._active_response_id is None:
                                self._is_interrupted = False
                                self._active_response_id = utils.shortuuid()

                                # Create streams for text and audio
                                text_stream = utils.aio.Chan[str]()
                                audio_stream = utils.aio.Chan[rtc.AudioFrame]()

                                content = NovaContent(
                                    response_id=self._active_response_id,
                                    item_id=self._active_response_id,
                                    output_index=0,
                                    content_index=0,
                                    text="",
                                    audio=[],
                                    text_stream=text_stream,
                                    audio_stream=audio_stream,
                                    content_type="audio",
                                )
                                self.emit("response_content_added", content)

                        elif "textOutput" in event:
                            # Text output
                            text_content = event["textOutput"]["content"]
                            if '{ "interrupted" : true }' in text_content:
                                self._is_interrupted = True
                                for stream in (content.text_stream, content.audio_stream):
                                    if isinstance(stream, utils.aio.Chan):
                                        stream.close()

                                self.emit("agent_speech_stopped")
                                self._is_interrupted = True

                                self._active_response_id = None
                            
                            if text_stream and not text_stream.closed:
                                text_stream.send_nowait(text_content)

                        elif "audioOutput" in event:
                            # Audio output
                            audio_content = event["audioOutput"]["content"]
                            audio_bytes = base64.b64decode(audio_content)

                            # Create audio frame
                            frame = rtc.AudioFrame(
                                data=audio_bytes,
                                sample_rate=24000,
                                num_channels=1,
                                samples_per_channel=len(audio_bytes) // 2,
                            )

                            if audio_stream and not audio_stream.closed:
                                audio_stream.send_nowait(frame)

                        elif "toolUse" in event:
                            # Tool use request
                            self._toolUseContent = event["toolUse"]
                            self._toolName = event["toolUse"]["toolName"]
                            self._toolUseId = event["toolUse"]["toolUseId"]
                            tool_content = event["toolUse"]["content"]

                            # Process function call
                            if self._fnc_ctx is not None:
                                # Create function call info
                                fnc_calls = []
                                try:
                                    # tool_args = json.loads(tool_content)
                                    fnc_call_info = _create_ai_function_info(
                                        self._fnc_ctx, self._toolUseId, self._toolName, tool_content, 
                                    )
                                    if fnc_call_info:
                                        fnc_calls.append(fnc_call_info)
                                except json.JSONDecodeError:
                                    logger.error(
                                        f"Failed to parse tool arguments: {tool_content}"
                                    )

                                if fnc_calls:
                                    self.emit("function_calls_collected", fnc_calls)

                                    # Execute each function
                                    for fnc_call_info in fnc_calls:
                                        self._fnc_tasks.create_task(
                                            self._run_fnc_task(fnc_call_info)
                                        )

                        elif "contentEnd" in event:
                            # Content is done
                            if event.get("contentEnd", {}).get("role") == "ASSISTANT":
                                if text_stream and not text_stream.closed:
                                    text_stream.close()
                                if audio_stream and not audio_stream.closed:
                                    audio_stream.close()

                                self.emit("agent_speech_stopped")
                                self._active_response_id = None

                except StopAsyncIteration:
                    # Stream is done
                    break
                except Exception as e:
                    logger.error(f"Error receiving response: {str(e)}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in receive task: {str(e)}")


    @utils.log_exceptions(logger=logger)
    async def _run_fnc_task(self, fnc_call_info: llm.FunctionCallInfo):
        """Execute a function call and send the result back to Nova."""
        logger.debug(
            "executing ai function",
            extra={
                "function": fnc_call_info.function_info.name,
            },
        )

        called_fnc = fnc_call_info.execute()

        try:
            await called_fnc.task
        except Exception as e:
            logger.exception(
                "error executing ai function",
                extra={
                    "function": fnc_call_info.function_info.name,
                },
                exc_info=e,
            )

        # Create tool response
        tool_call = llm.ChatMessage.create_tool_from_called_function(called_fnc)

        if tool_call.content is not None:
            content_name = str(uuid.uuid4())

            # Queue tool response to be sent
            self._send_ch.send_nowait(
                {
                    "type": "tool_response",
                    "tool_use_id": self._toolUseId,
                    "content_name": content_name,
                    "result": tool_call.content,
                }
            )

            self.emit("function_calls_finished", [called_fnc])

    async def _process_audio_input(self):
        """Process audio input from the queue and send to the stream."""
        try:
            audio_streaming_started = False

            while self._is_active:
                try:
                    # Get item from queue with timeout
                    # try:
                    data = await asyncio.wait_for(
                        self._audio_input_queue.get(), 0.1
                    )
                    logger.debug(f"Processing audio data: {type(data)}")
                    # except asyncio.TimeoutError:
                    #     continue

                    if data.get("type") == "audio":
                        # Start audio streaming if not already started
                        if not audio_streaming_started:
                            self._send_ch.send_nowait({"type": "audio_content_start"})
                            audio_streaming_started = True

                        # Base64 encode the audio data
                        audio_bytes = data.get("audio_bytes")
                        if audio_bytes:
                            blob = base64.b64encode(audio_bytes)
                            self._send_ch.send_nowait(
                                {"type": "audio_chunk", "content": blob.decode("utf-8")}
                            )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing input: {str(e)}")

        except Exception as e:
            logger.error(f"Audio processing task error: {str(e)}")
        finally:
            # End audio content if it was started
            if audio_streaming_started:
                self._send_ch.send_nowait({"type": "audio_content_end"})

    async def start_audio_stream(self):
        """Start streaming audio to Nova."""
        if not self._is_active:
            logger.warning("Cannot start audio stream, session not active")
            return

        self._send_ch.send_nowait({"type": "audio_content_start"})

    async def end_audio_stream(self):
        """End the audio stream."""
        if not self._is_active:
            return

        self._send_ch.send_nowait({"type": "audio_content_end"})

    def send_text(self, text: str):
        """Send text input to Nova."""
        if not text or not self._is_active:
            return

        content_name = str(uuid.uuid4())
        self._send_ch.send_nowait(
            {"type": "text_content", "content": text, "content_name": content_name}
        )

        # Update chat context
        self._chat_ctx.append(text=text, role="user")
