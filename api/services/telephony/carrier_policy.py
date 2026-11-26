"""Carriers this platform refuses to carry calls or callbacks for (VOZ-N0-29, VOZ-N0-40).

No telephony imports: the provider factory and callback_auth both read this module, and callback_auth imports the
factory.
"""

from api.enums import WorkflowRunMode

# Cloudonix callbacks carry no signature this platform verifies, so Cloudonix is disabled in every deployment until
# follow-up VOZ-FU-N0-29-cloudonix-signing verifies its scheme against a real account.
UNVERIFIED_CARRIERS = frozenset({WorkflowRunMode.CLOUDONIX.value})


class CarrierDisabledError(Exception):
    """A call path resolved a carrier in ``UNVERIFIED_CARRIERS``. The message is ``code``, the stable code
    callers surface (an HTTP detail, a campaign failure reason, a websocket close reason)."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.code = f"{provider_name}_signing_unverified"
        super().__init__(self.code)


def refuse_disabled_carrier(provider_name: str) -> None:
    """Raise ``CarrierDisabledError`` for a carrier in ``UNVERIFIED_CARRIERS``."""
    if provider_name in UNVERIFIED_CARRIERS:
        raise CarrierDisabledError(provider_name)
