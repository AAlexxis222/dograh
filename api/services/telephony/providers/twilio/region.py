"""Twilio region resolution: which API host and which credentials a call uses.

Twilio keeps API keys and Auth Tokens per region, so changing the host is not
enough: the credentials of the chosen region have to travel with it. Without a
cell policy the behaviour is exactly the upstream one (US1, ``api.twilio.com``,
the flat ``account_sid``/``auth_token`` pair). With ``CARRIER_REGION_POLICY=eu``
the default region is Ireland (``ie1`` / ``dublin``) and any region outside the
EEA is refused unless the config carries an explicit, reasoned exception.

Config shape read here (everything optional, legacy flat configs keep working)::

    {
      "region": "ie1", "edge": "dublin",
      "credentials": {"ie1": {"account_sid": "AC..", "auth_token": ".."}},
      "allow_non_eu_carrier_region": true,
      "allow_non_eu_carrier_region_reason": "why this cell may use a non-EEA region",
      "account_sid": "AC..", "auth_token": ".."   # legacy: credentials of us1
    }
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

POLICY_ENV = "CARRIER_REGION_POLICY"
EU_POLICY = "eu"

US1 = "us1"
EU_DEFAULT_REGION = "ie1"
_EEA_REGIONS = frozenset({"ie1"})
_DEFAULT_EDGE = {"ie1": "dublin", "au1": "sydney"}


class RegionError(Exception):
    """A Twilio region cannot be used as configured (VOZ-AC-B0-28 shape).

    ``code`` is stable and machine-readable; ``reason`` says what is wrong and
    ``hint`` what to do about it. Both are always non-empty.
    """

    def __init__(
        self,
        code: str,
        reason: str,
        hint: str,
        *,
        where: str = "twilio telephony configuration",
    ):
        super().__init__(f"{code} at {where}: {reason} (hint: {hint})")
        self.code = code
        self.reason = reason
        self.hint = hint
        self.where = where


@dataclass(frozen=True)
class TwilioEndpoint:
    """The resolved API endpoint and the credentials that belong to its region.

    ``edge`` is ``None`` for the US1 default host, which has no region/edge pair.
    """

    region: str
    edge: str | None
    account_sid: str | None
    auth_token: str | None
    base_url: str

    @property
    def serializer_args(self) -> dict[str, str]:
        """``region``/``edge`` for pipecat's serializer (both or neither)."""
        return {"region": self.region, "edge": self.edge} if self.edge else {}


def cell_policy_from_env() -> str | None:
    return os.environ.get(POLICY_ENV) or None


def resolve_twilio_endpoint(
    config: Mapping[str, Any], cell_policy: str | None
) -> TwilioEndpoint:
    policy = _normalize_policy(cell_policy)
    region = config.get("region") or (EU_DEFAULT_REGION if policy else US1)
    if policy:
        _require_eea_or_exception(region, config)

    edge = None if region == US1 else config.get("edge") or _DEFAULT_EDGE.get(region)
    if region != US1 and not edge:
        raise RegionError(
            "carrier_region_edge_missing",
            f"Twilio region '{region}' has no known default edge and the "
            "configuration does not set one.",
            "Set 'edge' in the Twilio telephony configuration "
            "(for example 'dublin' for ie1).",
        )

    account_sid, auth_token = _credentials_for(config, region)
    if not (account_sid and auth_token) and (region != US1 or policy):
        raise RegionError(
            "carrier_region_credentials_missing",
            f"No complete account_sid/auth_token for Twilio region '{region}'.",
            f"Add credentials.{region}.account_sid and credentials.{region}.auth_token "
            "from a Twilio account in that region (API keys and Auth Tokens are "
            "per region).",
        )

    host = (
        f"https://api.{edge}.{region}.twilio.com" if edge else "https://api.twilio.com"
    )
    return TwilioEndpoint(
        region=region,
        edge=edge,
        account_sid=account_sid,
        auth_token=auth_token,
        base_url=f"{host}/2010-04-01/Accounts/{account_sid}",
    )


def _normalize_policy(cell_policy: str | None) -> str | None:
    policy = (cell_policy or "").strip().lower()
    if policy in ("", EU_POLICY):
        return policy or None
    raise RegionError(
        "carrier_region_policy_invalid",
        f"{POLICY_ENV}='{cell_policy}' is not a known policy.",
        f"Set {POLICY_ENV} to '{EU_POLICY}' or leave it unset.",
        where=POLICY_ENV,
    )


def _require_eea_or_exception(region: str, config: Mapping[str, Any]) -> None:
    if region in _EEA_REGIONS:
        return
    reason = str(config.get("allow_non_eu_carrier_region_reason") or "").strip()
    if config.get("allow_non_eu_carrier_region") is True and reason:
        logger.warning(
            "Twilio region '{}' outside the EEA allowed by configuration: {}",
            region,
            reason,
        )
        return
    raise RegionError(
        "carrier_region_not_eu",
        f"Twilio region '{region}' is outside the EEA and this cell requires the EU.",
        "Use region 'ie1', or set allow_non_eu_carrier_region=true together "
        "with a non-empty allow_non_eu_carrier_region_reason.",
    )


def _credentials_for(
    config: Mapping[str, Any], region: str
) -> tuple[str | None, str | None]:
    """Credentials of ``region``; flat top-level ones are the legacy ``us1`` pair."""
    by_region = config.get("credentials") or {}
    if not isinstance(by_region, Mapping):
        raise RegionError(
            "carrier_region_credentials_invalid",
            f"'credentials' must map region names to account_sid/auth_token, "
            f"got {type(by_region).__name__}.",
            'Set credentials to an object such as {"ie1": {"account_sid": "AC..", '
            '"auth_token": ".."}}.',
        )
    regional = by_region.get(region)
    if not isinstance(regional, Mapping):
        regional = config if region == US1 else {}
    return regional.get("account_sid"), regional.get("auth_token")
