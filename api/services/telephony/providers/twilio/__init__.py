"""Twilio telephony provider package."""

from typing import Any, Dict

from api.services.telephony.registry import (
    ProviderSpec,
    ProviderUIField,
    ProviderUIMetadata,
    register,
)

from .config import TwilioConfigurationRequest
from .provider import TwilioProvider
from .transport import create_transport

# Optional keys read by ``region.resolve_twilio_endpoint`` and the provider.
# Forwarded only when stored, so legacy configs keep their exact shape. They are
# operator-seeded, so the API never shows them (``credentials`` holds a token)
# and an update that sends only the editable fields keeps them.
_REGION_KEYS = (
    "region",
    "edge",
    "credentials",
    "allow_non_eu_carrier_region",
    "allow_non_eu_carrier_region_reason",
    "fallback_url",
)


def _config_loader(value: Dict[str, Any]) -> Dict[str, Any]:
    loaded = {
        "provider": "twilio",
        "account_sid": value.get("account_sid"),
        "auth_token": value.get("auth_token"),
        "from_numbers": value.get("from_numbers", []),
    }
    loaded.update({key: value[key] for key in _REGION_KEYS if key in value})
    return loaded


_UI_METADATA = ProviderUIMetadata(
    display_name="Twilio",
    docs_url="https://docs.dograh.com/integrations/telephony/twilio",
    fields=[
        ProviderUIField(
            name="account_sid",
            label="Account SID",
            type="text",
            sensitive=True,
            description="Twilio Account SID (starts with AC)",
        ),
        ProviderUIField(
            name="auth_token",
            label="Auth Token",
            type="password",
            sensitive=True,
            description="Twilio Auth Token",
        ),
        ProviderUIField(
            name="from_numbers",
            label="Phone Numbers",
            type="string-array",
            description="E.164-formatted Twilio phone numbers used for outbound calls",
        ),
    ],
)


SPEC = ProviderSpec(
    name="twilio",
    provider_cls=TwilioProvider,
    config_loader=_config_loader,
    transport_factory=create_transport,
    transport_sample_rate=8000,
    config_request_cls=TwilioConfigurationRequest,
    ui_metadata=_UI_METADATA,
    account_id_credential_field="account_sid",
    server_managed_credential_fields=_REGION_KEYS,
)


register(SPEC)


__all__ = [
    "SPEC",
    "TwilioConfigurationRequest",
    "TwilioProvider",
    "create_transport",
]
