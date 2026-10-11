"""Database access for telephony phone numbers.

Phone numbers are first-class entities (PSTN, SIP URI, or SIP extension)
owned by a telephony configuration. They power both outbound caller-ID
selection and inbound call routing.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select
from sqlalchemy.orm.attributes import set_committed_value

from api.db.base_client import BaseDBClient
from api.db.models import (
    TelephonyConfigurationModel,
    TelephonyPhoneNumberModel,
    WorkflowModel,
)
from api.db.telephony_configuration_client import account_id_for
from api.utils.telephony_address import normalize_telephony_address


class TelephonyPhoneNumberConflictError(Exception):
    """Raised when a phone number violates a DB constraint."""


class TelephonyPhoneNumberClient(BaseDBClient):
    async def list_phone_numbers_for_config(
        self, telephony_configuration_id: int
    ) -> List[TelephonyPhoneNumberModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(TelephonyPhoneNumberModel)
                .where(
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == telephony_configuration_id
                )
                .order_by(TelephonyPhoneNumberModel.created_at)
            )
            return list(result.scalars().all())

    async def list_phone_numbers_with_workflow_name_for_config(
        self, telephony_configuration_id: int
    ) -> List[Tuple[TelephonyPhoneNumberModel, Optional[str]]]:
        """Same as :meth:`list_phone_numbers_for_config` but also returns the
        inbound workflow's display name (or None) for each row, fetched via a
        single LEFT JOIN so we don't load entire workflow rows."""
        async with self.async_session() as session:
            result = await session.execute(
                select(TelephonyPhoneNumberModel, WorkflowModel.name)
                .join(
                    WorkflowModel,
                    WorkflowModel.id == TelephonyPhoneNumberModel.inbound_workflow_id,
                    isouter=True,
                )
                .where(
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == telephony_configuration_id
                )
                .order_by(TelephonyPhoneNumberModel.created_at)
            )
            return [(row, name) for row, name in result.all()]

    async def list_active_normalized_addresses_for_config(
        self, telephony_configuration_id: int
    ) -> List[str]:
        """Active phone numbers as canonical address strings (E.164 for PSTN,
        normalized SIP otherwise) — the shape providers want in their
        ``from_numbers`` list for caller-ID and rate-limit pool keys."""
        async with self.async_session() as session:
            result = await session.execute(
                select(TelephonyPhoneNumberModel.address_normalized)
                .where(
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == telephony_configuration_id,
                    TelephonyPhoneNumberModel.is_active.is_(True),
                )
                .order_by(TelephonyPhoneNumberModel.created_at)
            )
            return [row[0] for row in result.all()]

    async def get_phone_number(
        self, phone_number_id: int
    ) -> Optional[TelephonyPhoneNumberModel]:
        async with self.async_session() as session:
            return await session.get(TelephonyPhoneNumberModel, phone_number_id)

    async def get_phone_number_for_config(
        self, phone_number_id: int, telephony_configuration_id: int
    ) -> Optional[TelephonyPhoneNumberModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(TelephonyPhoneNumberModel).where(
                    TelephonyPhoneNumberModel.id == phone_number_id,
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == telephony_configuration_id,
                )
            )
            return result.scalars().first()

    async def find_active_phone_number_for_inbound(
        self,
        organization_id: int,
        address: str,
        provider: str,
        country_hint: Optional[str] = None,
    ) -> Optional[TelephonyPhoneNumberModel]:
        """Inbound routing primary lookup for an active number and config."""
        normalized = normalize_telephony_address(address, country_hint=country_hint)

        async with self.async_session() as session:
            result = await session.execute(
                select(TelephonyPhoneNumberModel)
                .join(
                    TelephonyConfigurationModel,
                    TelephonyConfigurationModel.id
                    == TelephonyPhoneNumberModel.telephony_configuration_id,
                )
                .where(
                    TelephonyPhoneNumberModel.organization_id == organization_id,
                    TelephonyPhoneNumberModel.address_normalized
                    == normalized.canonical,
                    TelephonyPhoneNumberModel.is_active.is_(True),
                    TelephonyConfigurationModel.provider == provider,
                    TelephonyConfigurationModel.inactive.is_(False),
                )
            )
            return result.scalars().first()

    async def _configurations_for_account(
        self, session, stmt, provider: str, account_id: str
    ) -> List[Tuple[TelephonyConfigurationModel, TelephonyPhoneNumberModel]]:
        """The ``(config, phone)`` rows of ``stmt`` whose credentials name ``account_id``.

        ``stmt`` is the caller's query without the account filter. The ``account_id``
        column is the index, not the truth: a configuration saved by a release that
        predates the column (a rolling update, or an image rolled back and forward
        again) can carry NULL, or the id it had before its credentials were edited.
        So every column hit is checked against the credentials with the writers'
        mapping, and when nothing verifies, the provider's other rows that own the
        number are checked the same way. Every row found out of step is re-keyed.
        """
        # populate_existing: the compare-and-set below needs the stored value, not a
        # copy this session may already hold.
        stmt = stmt.execution_options(populate_existing=True)
        keyed = (
            await session.execute(
                stmt.where(TelephonyConfigurationModel.account_id == account_id)
            )
        ).all()
        rows = []
        fixes = {}  # config id -> (config, value read, value to store)
        for config, phone in keyed:
            actual = account_id_for(provider, config.credentials)
            if actual == account_id:
                rows.append((config, phone))
            else:
                fixes[config.id] = (config, config.account_id, actual)
        if not rows:
            others = await session.execute(
                stmt.where(
                    TelephonyConfigurationModel.account_id.is_distinct_from(account_id)
                )
            )
            for config, phone in others.all():
                if account_id_for(provider, config.credentials) == account_id:
                    rows.append((config, phone))
                    fixes[config.id] = (config, config.account_id, account_id)
        await self._store_account_ids(session, rows, fixes)
        return rows

    async def _store_account_ids(self, session, rows, fixes) -> None:
        """Write the re-keyed values through the read's own session (one connection).

        Best effort: the call has already matched, so a failing write is logged and
        the next lookup retries it. The returned rows are detached first, so neither
        the commit nor a rollback can expire them. ``updated_at`` is set to itself so
        the heal does not look like an edit, and each write only applies if the
        column still holds the value that was read.
        """
        if not fixes:
            return
        try:
            # One config can own several of the numbers asked for: detach each object once.
            for detached in {id(o): o for row in rows for o in row}.values():
                session.expunge(detached)
            for config_id, (_, read, value) in fixes.items():
                await session.execute(
                    update(TelephonyConfigurationModel)
                    .where(
                        TelephonyConfigurationModel.id == config_id,
                        TelephonyConfigurationModel.account_id.is_(None)
                        if read is None
                        else TelephonyConfigurationModel.account_id == read,
                    )
                    .values(
                        account_id=value,
                        updated_at=TelephonyConfigurationModel.updated_at,
                    )
                )
            await session.commit()
        except Exception as exc:
            logger.warning(
                f"account_id_heal_failed: could not re-key configurations "
                f"{sorted(fixes)}: {exc!r}; the lookup result is unaffected"
            )
            return
        for config, _, value in fixes.values():
            set_committed_value(config, "account_id", value)
        logger.info(
            f"Re-keyed account_id on configurations "
            f"{ {i: v for i, (_, _, v) in fixes.items()} }: it disagreed with their credentials"
        )

    async def find_inbound_route_by_account(
        self,
        provider: str,
        account_id: str,
        to_number: str,
        country_hint: Optional[str] = None,
        organization_id: Optional[int] = None,
    ) -> Optional[Tuple[TelephonyConfigurationModel, TelephonyPhoneNumberModel]]:
        """Combined primary-path lookup for inbound dispatch.

        One SQL roundtrip that joins ``telephony_configurations`` and
        ``telephony_phone_numbers`` and matches all of:
        provider, ``account_id`` column == account_id (each hit is verified
        against the row's credentials; see ``_configurations_for_account``),
        ``phone.address_normalized == canonical(to_number)``, and
        ``phone.is_active``, and a non-parked configuration. Replaces the
        previous pattern of resolving the config and the phone number in two
        separate queries with a Python-side loop over candidate configs.

        Returns ``(config, phone_number)`` or None when the primary path
        misses (e.g. legacy non-E.164 stored addresses); the caller should
        fall back to the fuzzy ``numbers_match`` path in that case.

        A lookup can write: rows whose ``account_id`` disagrees with their
        credentials are re-keyed through the same session, best effort (a
        failing write is logged and the match is still returned).
        """
        if not (provider and account_id and to_number):
            return None

        normalized = normalize_telephony_address(to_number, country_hint=country_hint)

        async with self.async_session() as session:
            stmt = (
                select(TelephonyConfigurationModel, TelephonyPhoneNumberModel)
                .join(
                    TelephonyPhoneNumberModel,
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == TelephonyConfigurationModel.id,
                )
                .where(
                    TelephonyConfigurationModel.provider == provider,
                    TelephonyPhoneNumberModel.address_normalized
                    == normalized.canonical,
                    TelephonyPhoneNumberModel.is_active.is_(True),
                    TelephonyConfigurationModel.inactive.is_(False),
                )
            )
            if organization_id is not None:
                stmt = stmt.where(
                    TelephonyConfigurationModel.organization_id == organization_id
                )
            rows = await self._configurations_for_account(
                session, stmt, provider, account_id
            )

            if not rows:
                logger.info(
                    f"Inbound route lookup miss — provider={provider} "
                    f"account_id={account_id!r} "
                    f"to={to_number!r} canonical={normalized.canonical!r} "
                    f"org_scope={organization_id}"
                )
                return None

            # The (provider, account_id, address_normalized) tuple is meant
            # to be globally unique; the write paths that keep it so live in
            # ``api.services.telephony.inbound_routing``. When it isn't, the
            # call silently lands in whichever org Postgres returned first, so
            # name every candidate rather than only the winner.
            if len(rows) > 1:
                candidates = ", ".join(
                    f"config={cfg.id}/org={cfg.organization_id}/name={cfg.name!r}"
                    f"/phone={phone.id}"
                    for cfg, phone in rows
                )
                logger.error(
                    f"Ambiguous inbound route — provider={provider} "
                    f"account_id={account_id!r} "
                    f"canonical={normalized.canonical!r} matched {len(rows)} "
                    f"rows, using the first: {candidates}"
                )

            config, phone_number = rows[0][0], rows[0][1]
            logger.info(
                f"Inbound route resolved — provider={provider} "
                f"account_id={account_id!r} "
                f"canonical={normalized.canonical!r} -> config={config.id} "
                f"org={config.organization_id} name={config.name!r} "
                f"phone={phone_number.id} address={phone_number.address!r} "
                f"inbound_workflow_id={phone_number.inbound_workflow_id}"
            )
            return config, phone_number

    async def find_inbound_route_by_called_number(
        self,
        provider: str,
        to_number: str,
        country_hint: Optional[str] = None,
    ) -> Optional[Tuple[TelephonyConfigurationModel, TelephonyPhoneNumberModel]]:
        """Inbound route lookup when the webhook omits account_id.

        Some providers (Exotel Voicebot Applet dynamic URL) POST/GET inbound
        webhooks without an account identifier. In that case we match on
        ``(provider, canonical called number)`` across all orgs. Returns a
        route only when exactly one active, non-parked config owns that
        number; returns None on zero matches or when multiple configs would
        race for the same call.
        """
        if not (provider and to_number):
            return None

        normalized = normalize_telephony_address(to_number, country_hint=country_hint)

        async with self.async_session() as session:
            stmt = (
                select(TelephonyConfigurationModel, TelephonyPhoneNumberModel)
                .join(
                    TelephonyPhoneNumberModel,
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == TelephonyConfigurationModel.id,
                )
                .where(
                    TelephonyConfigurationModel.provider == provider,
                    TelephonyPhoneNumberModel.address_normalized
                    == normalized.canonical,
                    TelephonyPhoneNumberModel.is_active.is_(True),
                    TelephonyConfigurationModel.inactive.is_(False),
                )
                .limit(2)
            )
            result = await session.execute(stmt)
            rows = result.all()
            if len(rows) == 1:
                return rows[0][0], rows[0][1]
            return None

    async def find_inbound_routing_conflicts(
        self,
        provider: str,
        account_id: str,
        addresses_normalized: Sequence[str],
        exclude_configuration_id: Optional[int] = None,
    ) -> List[Tuple[TelephonyConfigurationModel, TelephonyPhoneNumberModel]]:
        """Rows that already hold one of these inbound routing keys.

        Inbound dispatch keys on (provider, account_id, address_normalized) —
        see ``find_inbound_route_by_account``. That tuple must be globally unique
        or two orgs race for the same call. The rule and the decision of when to
        apply it live in ``api.services.telephony.inbound_routing``; this is its
        query.

        Returns every conflicting (config, phone_number) pair, possibly owned by
        other organizations. ``exclude_configuration_id`` drops the configuration
        being updated so it cannot conflict with itself. Addresses are matched as
        already-canonical ``address_normalized`` values — callers normalize.

        Like ``find_inbound_route_by_account``, it re-keys rows whose ``account_id``
        disagrees with their credentials (best effort, same session) before it
        answers.
        """
        if not (provider and account_id and addresses_normalized):
            return []

        async with self.async_session() as session:
            stmt = (
                select(TelephonyConfigurationModel, TelephonyPhoneNumberModel)
                .join(
                    TelephonyPhoneNumberModel,
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == TelephonyConfigurationModel.id,
                )
                .where(
                    TelephonyConfigurationModel.provider == provider,
                    TelephonyPhoneNumberModel.address_normalized.in_(
                        list(addresses_normalized)
                    ),
                )
            )
            if exclude_configuration_id is not None:
                stmt = stmt.where(
                    TelephonyConfigurationModel.id != exclude_configuration_id
                )
            return await self._configurations_for_account(
                session, stmt, provider, account_id
            )

    async def create_phone_number(
        self,
        organization_id: int,
        telephony_configuration_id: int,
        address: str,
        country_code: Optional[str] = None,
        label: Optional[str] = None,
        inbound_workflow_id: Optional[int] = None,
        telephony_trunk_id: Optional[int] = None,
        is_active: bool = True,
        is_default_caller_id: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TelephonyPhoneNumberModel:
        normalized = normalize_telephony_address(address, country_hint=country_code)

        async with self.async_session() as session:
            if is_default_caller_id:
                await self._clear_default_caller_id(session, telephony_configuration_id)

            row = TelephonyPhoneNumberModel(
                organization_id=organization_id,
                telephony_configuration_id=telephony_configuration_id,
                address=address,
                address_normalized=normalized.canonical,
                address_type=normalized.address_type,
                country_code=country_code or normalized.country_code,
                label=label,
                inbound_workflow_id=inbound_workflow_id,
                telephony_trunk_id=telephony_trunk_id,
                is_active=is_active,
                is_default_caller_id=is_default_caller_id,
                extra_metadata=extra_metadata or {},
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as e:
                await session.rollback()
                raise TelephonyPhoneNumberConflictError(str(e)) from e
            await session.refresh(row)
            return row

    async def update_phone_number(
        self,
        phone_number_id: int,
        telephony_configuration_id: int,
        label: Optional[str] = None,
        inbound_workflow_id: Optional[int] = None,
        telephony_trunk_id: Optional[int] = None,
        is_active: Optional[bool] = None,
        country_code: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        clear_inbound_workflow: bool = False,
        clear_trunk: bool = False,
    ) -> Optional[TelephonyPhoneNumberModel]:
        """Partial update. ``address`` is intentionally immutable — create a new
        row instead. Set ``clear_inbound_workflow``/``clear_trunk`` to null out
        the respective FK."""
        async with self.async_session() as session:
            row = await session.get(TelephonyPhoneNumberModel, phone_number_id)
            if not row or row.telephony_configuration_id != telephony_configuration_id:
                return None

            if label is not None:
                row.label = label
            if inbound_workflow_id is not None:
                row.inbound_workflow_id = inbound_workflow_id
            elif clear_inbound_workflow:
                row.inbound_workflow_id = None
            if telephony_trunk_id is not None:
                row.telephony_trunk_id = telephony_trunk_id
            elif clear_trunk:
                row.telephony_trunk_id = None
            if is_active is not None:
                row.is_active = is_active
            if country_code is not None:
                row.country_code = country_code
            if extra_metadata is not None:
                row.extra_metadata = extra_metadata

            await session.commit()
            await session.refresh(row)
            return row

    async def set_default_caller_id(
        self, phone_number_id: int, telephony_configuration_id: int
    ) -> Optional[TelephonyPhoneNumberModel]:
        async with self.async_session() as session:
            row = await session.get(TelephonyPhoneNumberModel, phone_number_id)
            if not row or row.telephony_configuration_id != telephony_configuration_id:
                return None
            await self._clear_default_caller_id(session, telephony_configuration_id)
            row.is_default_caller_id = True
            await session.commit()
            await session.refresh(row)
            return row

    async def get_default_caller_id(
        self, telephony_configuration_id: int
    ) -> Optional[TelephonyPhoneNumberModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(TelephonyPhoneNumberModel).where(
                    TelephonyPhoneNumberModel.telephony_configuration_id
                    == telephony_configuration_id,
                    TelephonyPhoneNumberModel.is_default_caller_id.is_(True),
                )
            )
            return result.scalars().first()

    async def delete_phone_number(
        self, phone_number_id: int, telephony_configuration_id: int
    ) -> bool:
        async with self.async_session() as session:
            row = await session.get(TelephonyPhoneNumberModel, phone_number_id)
            if not row or row.telephony_configuration_id != telephony_configuration_id:
                return False
            await session.delete(row)
            await session.commit()
            return True

    @staticmethod
    async def _clear_default_caller_id(
        session, telephony_configuration_id: int
    ) -> None:
        await session.execute(
            update(TelephonyPhoneNumberModel)
            .where(
                TelephonyPhoneNumberModel.telephony_configuration_id
                == telephony_configuration_id,
                TelephonyPhoneNumberModel.is_default_caller_id.is_(True),
            )
            .values(is_default_caller_id=False)
        )
