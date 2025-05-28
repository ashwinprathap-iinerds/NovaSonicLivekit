from __future__ import annotations

import base64
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

import boto3
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.function_context import _is_optional_type

__all__ = ["_build_aws_ctx", "_build_tools", "_get_aws_credentials"]


def _get_aws_credentials(
    api_key: Optional[str], api_secret: Optional[str], region: Optional[str]
):
    region = region or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_DEFAULT_REGION must be set using the argument or by setting the AWS_DEFAULT_REGION environment variable."
        )

    # If API key and secret are provided, create a session with them
    if api_key and api_secret:
        session = boto3.Session(
            aws_access_key_id=api_key,
            aws_secret_access_key=api_secret,
            region_name=region,
        )
    else:
        session = boto3.Session(region_name=region)

    credentials = session.get_credentials()
    if not credentials or not credentials.access_key or not credentials.secret_key:
        raise ValueError("No valid AWS credentials found.")
    return credentials.access_key, credentials.secret_key, credentials.token