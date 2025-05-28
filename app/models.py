from pydantic import BaseModel
from typing import Optional

class SIPCallRequest(BaseModel):
    company_id: str
    phone_number: str
    participant_identity: Optional[str] = None
    participant_name: Optional[str] = None
    participant_metadata: Optional[str] = None
    krisp_enabled: bool = True


# Define response model
class SIPCallResponse(BaseModel):
    success: bool
    participant_id: Optional[str] = None
    room_name: str
    message: str

class CreateOutboundSIPTrunkRequest(BaseModel):
    name: str
    sip_uri: str
    company_id: str
    number: str

class CreateInboundSIPTrunkRequest(BaseModel):
    name: str
    number: str
    krisp_enabled: bool = True

class CreateInboundDispatchRequest(BaseModel):
    trunk_id: str
    room_name_prefix: str
    agent_name: str