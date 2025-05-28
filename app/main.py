import uvicorn
import os
import subprocess
from livekit import api 
from livekit.protocol.sip import CreateSIPParticipantRequest, CreateSIPOutboundTrunkRequest,SIPInboundTrunkInfo, SIPOutboundTrunkInfo, CreateSIPInboundTrunkRequest
from app.constants import API_CHECK_STATUS_MESSAGE, SERVER_ERROR
from app.models import SIPCallRequest, SIPCallResponse, CreateInboundSIPTrunkRequest, CreateOutboundSIPTrunkRequest, CreateInboundDispatchRequest
from app.outbound_trunk_utils import get_outbound_trunk
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPParticipantInfo

import os
from livekit import api
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

livekit_api = api.LiveKitAPI(
    url=os.getenv("LIVEKIT_URL"),
    api_key=os.getenv("LIVEKIT_API_KEY"),
    api_secret=os.getenv("LIVEKIT_API_SECRET")
)

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the agent process
    global agent_process
    try:
        # Start the agent script in a subprocess
        print("Downloading model...")
        download_process = subprocess.run( ["python", "-m", "app.aws_agent_outbound", "download-files"],
            env=os.environ.copy(),
            check=True)
        print(f"Model download complete. {download_process}")
        if download_process.returncode == 0:
        
            agent_process_1= subprocess.Popen(
                ["python", "-m", "app.agent_inbound", "start"],
                env=os.environ.copy(),
            )
            

        else:
            raise Exception("Model Download failure")
        
        # print(f"Started agent process with PID: {agent_process_1.pid}")
        app.livekit_api = api.LiveKitAPI()
    except Exception as e:
        print(f"Failed to start agent: {e}")
        agent_process = None
    
    yield  # FastAPI runs here



@app.get("/")
async def status_check():
    try:
        response = API_CHECK_STATUS_MESSAGE
        return JSONResponse(content=response)
    except Exception as error:
        return JSONResponse(content=error, status_code=SERVER_ERROR)


@app.post("/initiate-call", response_model=SIPCallResponse)
async def initiate_sip_call(call_request: SIPCallRequest):
    """
    Initiate a SIP call to the specified phone number and connect it to a LiveKit room
    """
    try:
        # Initialize LiveKit API client
        livekit_api = api.LiveKitAPI()
        
        # Create identity if not provided
        participant_identity = call_request.participant_identity or f"sip-{call_request.phone_number}"
        participant_name = call_request.participant_name or f"Caller {call_request.phone_number}"
        participant_metadata = call_request.participant_metadata 
        room_name = f"{participant_name}-{participant_identity}"
        company_id = call_request.company_id
        trunk = await get_outbound_trunk(livekit_api,company_id)
        
        # Create SIP participant request
        request = CreateSIPParticipantRequest(
            sip_trunk_id=trunk.sip_trunk_id,
            sip_call_to=call_request.phone_number,
            room_name=room_name,
            participant_identity=participant_identity,
            participant_name=participant_name,
            participant_metadata=participant_metadata,
            krisp_enabled=call_request.krisp_enabled,
        )
        
        # Create SIP participant
        _ = await livekit_api.sip.create_sip_participant(request)
        
        # Return success response
        return SIPCallResponse(
            success=True,
            participant_id=participant_identity,
            room_name=room_name,
            message=f"Successfully created SIP participant for {call_request.phone_number}"
        )
        
    except Exception as e:
        # Handle errors
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate call: {str(e)}"
        )


@app.post("/create-outbound-sip-trunk", response_model=dict)
async def create_outbound_sip_trunk(trunk_request: CreateOutboundSIPTrunkRequest):
    """
    Create a new SIP outbound trunk
    """
    try:
        livekit_api = api.LiveKitAPI()
        print(trunk_request.sip_uri)
        trunk = SIPOutboundTrunkInfo(
            name = trunk_request.name,
            address = trunk_request.sip_uri,
            numbers = [trunk_request.number],
            metadata=trunk_request.company_id
        
        )
        # Create SIP trunk request
        request = CreateSIPOutboundTrunkRequest(trunk=trunk
        )
        
        trunk = await livekit_api.sip.create_sip_outbound_trunk(request)
        
        # Return success response
        return {
            "success": True,
            "trunk_id": trunk.sip_trunk_id,
            "name": trunk.name,
            "sip_uri": trunk.address,
            "company_id": trunk.metadata,
            "message": f"Successfully created SIP trunk: {trunk.name}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create SIP trunk: {str(e)}"
        )
    
@app.post("/create-inbound-sip-trunk", response_model=dict)
async def create_inbound_sip_trunk(trunk_request: CreateInboundSIPTrunkRequest):
    """
    Create a new SIP Inbound trunk
    """
    try:
        livekit_api = api.LiveKitAPI()
        trunk = SIPInboundTrunkInfo(
            name = trunk_request.name,
            numbers = [trunk_request.number],
            krisp_enabled = trunk_request.krisp_enabled,
        
        )
        # Create SIP trunk request
        request = CreateSIPInboundTrunkRequest(trunk=trunk
        )
        
        trunk = await livekit_api.sip.create_sip_inbound_trunk(request)
        
        # Return success response
        return {
            "success": True,
            "trunk_id": trunk.sip_trunk_id,
            "name": trunk.name,
            "message": f"Successfully created Inbound SIP trunk: {trunk.name}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create SIP trunk: {str(e)}"
        )

@app.post("/create-inbound-dispatch-rule", response_model=dict)
async def create_inbound_dispatch_rule(dispatch_request: CreateInboundDispatchRequest):
    """
    Create a new Dispatch rule
    """
    try:
        livekit_api = api.LiveKitAPI()
        request = api.CreateSIPDispatchRuleRequest(
            trunk_ids=[dispatch_request.trunk_id],
            rule=api.SIPDispatchRule(
                dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                    room_prefix=dispatch_request.room_name_prefix,
                )
            ),
            room_config=api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(
                    agent_name=dispatch_request.agent_name,
                    metadata="job dispatch metadata",
                )]
            )
        )
        dispatch = await livekit_api.sip.create_sip_dispatch_rule(request)
        # Return success response
        print(dispatch)
        return {
            "success": True,
            "name": dispatch.agent_name,
            "message": f"Successfully created Inbound SIP trunk: {dispatch.name}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create SIP trunk: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="34.238.169.185", port=8765)