from livekit.protocol.sip import ListSIPOutboundTrunkRequest

from dotenv import load_dotenv
load_dotenv()
async def list_outbound_trunk(livekit_api):
    request = ListSIPOutboundTrunkRequest()

    trunks = await livekit_api.sip.list_sip_outbound_trunk(request)
    print(trunks)
    return trunks


async def get_outbound_trunk(livekit_api, company_id: str):
    trunks = await list_outbound_trunk(livekit_api)
    matching_trunks = [trunk for trunk in trunks.items if trunk.metadata == company_id]
    if not matching_trunks:
        raise ValueError(f"No trunk found for company_id: {company_id}")
    return matching_trunks[-1]