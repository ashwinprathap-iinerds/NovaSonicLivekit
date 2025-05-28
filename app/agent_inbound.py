from livekit.agents import llm, cli, JobProcess, JobContext, AutoSubscribe, multimodal
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import  turn_detector, aws, silero
from plugins.aws import realtime 
from livekit.agents.worker import WorkerOptions
import os
from dotenv import load_dotenv


load_dotenv(".env.local")
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("AWS_DEFAULT_REGION:", os.getenv("AWS_REGION"))

nova_system_instructions = "You are a compassionate and professional virtual assistant helping patients book appointments with healthcare providers. "
"Use a calm, friendly, and respectful tone in all your interactions, as you are supporting individuals who may not be feeling well. "
"Speak slowly and clearly with natural pauses between sentences to ensure the user can understand everything you say. "
"Ensure that the user feels heard and understood, and guide them step-by-step with clarity and patience. "


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Initialize chat context
    print(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect.
    participant = await ctx.wait_for_participant()
    print(f"Starting voice assistant for participant {participant.identity}")
    print("started")
    chat_ctx = llm.ChatContext()
    fnc_ctx = llm.FunctionContext()
    initial_context = "Hi welcome to nova sonic agent"
    print(initial_context)
    
    chat_ctx.append(
        text="Hi",
        role="user",
    )
    chat_ctx.append(
        text=initial_context,
        role="assistant",
    )

    nova_model = realtime.NovaModel(
        voice="matthew",
        temperature=0.8,
        system_prompt=nova_system_instructions,
    )

    agent = multimodal.MultimodalAgent(
        model=nova_model,
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )

    try:
        agent.start(ctx.room, participant)
        # await speak_text(initial_context, ctx.room, ctx, participant)
    except Exception as e:
        print(f"Agent failed to start: {e}")


# Run when executed as script
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
