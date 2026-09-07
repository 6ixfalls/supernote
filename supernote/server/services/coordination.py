import logging
import time
from abc import ABC, abstractmethod

from sqlalchemy import and_, delete, func, or_, select, text
from sqlalchemy.dialects.sqlite import insert

from supernote.server.db.models.kv import KeyValueDO
from supernote.server.db.session import DatabaseSessionManager

logger = logging.getLogger(__name__)

DEFAULT_TTL = 31536000  # 1 year in seconds


class CoordinationService(ABC):
    """Interface for distributed locks and key-value state (tokens).

    This acts as a "redis-like" architecture for handling:
    1. Distributed Locks (prevent concurrent syncs).
    2. Session Tokens (Stateful JWT validity).
    """

    @abstractmethod
    async def set_value(self, key: str, value: str, ttl: int | None = None) -> None:
        """Set a key-value pair with optional TTL."""

    @abstractmethod
    async def set_value_if_absent(
        self, key: str, value: str, ttl: int | None = None
    ) -> bool:
        """Set a key only if it does not exist, returning whether it was set."""

    @abstractmethod
    async def get_value(self, key: str) -> str | None:
        """Get a value by key."""

    @abstractmethod
    async def delete_value(self, key: str) -> None:
        """Delete a key."""

    @abstractmethod
    async def delete_values(
        self, key_prefix: str, value: str, *, exact_value: bool = False
    ) -> list[str]:
        """Delete matching entries and return their keys."""

    @abstractmethod
    async def delete_user_sessions(
        self,
        account: str,
        user_id: int,
    ) -> list[str]:
        """Atomically delete sessions and aliases for one stable user identity."""

    @abstractmethod
    async def pop_value(self, key: str) -> str | None:
        """Get and delete a value atomically (if possible) or sequentially."""

    @abstractmethod
    async def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Atomically increment a value. Returns new value."""


class SqliteCoordinationService(CoordinationService):
    """SQLite-backed implementation for distributed locks and key-value state."""

    def __init__(self, session_manager: DatabaseSessionManager) -> None:
        self._session_manager = session_manager

    async def _cleanup(self) -> None:
        """Cleanup expired keys."""
        # This could be run periodically or on access.
        # For simplicity, we trust on-access checks or external cleanup jobs.

    async def set_value(self, key: str, value: str, ttl: int | None = None) -> None:
        """Set a key-value pair with optional TTL."""
        async with self._session_manager.session() as session:
            now = time.time()
            expiry = now + (ttl if ttl else DEFAULT_TTL)

            # Some short-lived values (such as abandoned login challenges) use
            # unique keys and will never be read again. Reclaim all expired
            # entries on writes so those values cannot accumulate indefinitely.
            await session.execute(delete(KeyValueDO).where(KeyValueDO.expiry < now))

            # Upsert
            stmt = select(KeyValueDO).where(KeyValueDO.key == key)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.value = value
                existing.expiry = expiry
            else:
                new_kv = KeyValueDO(key=key, value=value, expiry=expiry)
                session.add(new_kv)

            await session.commit()

    async def set_value_if_absent(
        self, key: str, value: str, ttl: int | None = None
    ) -> bool:
        """Set a key only if it does not exist, returning whether it was set."""
        async with self._session_manager.session() as session:
            now = time.time()
            expiry = now + (ttl if ttl else DEFAULT_TTL)

            await session.execute(delete(KeyValueDO).where(KeyValueDO.expiry < now))
            stmt = (
                insert(KeyValueDO)
                .values(key=key, value=value, expiry=expiry)
                .on_conflict_do_nothing(index_elements=[KeyValueDO.key])
                .returning(KeyValueDO.key)
            )
            result = await session.execute(stmt)
            inserted = result.scalar_one_or_none() is not None
            await session.commit()
            return inserted

    async def get_value(self, key: str) -> str | None:
        """Get a value by key."""
        async with self._session_manager.session() as session:
            stmt = select(KeyValueDO).where(KeyValueDO.key == key)
            result = await session.execute(stmt)
            kv = result.scalar_one_or_none()

            if not kv:
                return None

            if time.time() > kv.expiry:
                # Lazy delete
                await session.execute(delete(KeyValueDO).where(KeyValueDO.key == key))
                await session.commit()
                return None

            return kv.value

    async def delete_value(self, key: str) -> None:
        """Delete a key."""
        async with self._session_manager.session() as session:
            stmt = delete(KeyValueDO).where(KeyValueDO.key == key)
            await session.execute(stmt)
            await session.commit()

    async def delete_values(
        self, key_prefix: str, value: str, *, exact_value: bool = False
    ) -> list[str]:
        """Delete matching key-value pairs without loading session tokens."""
        async with self._session_manager.session() as session:
            value_clause = (
                KeyValueDO.value == value
                if exact_value
                else func.substr(KeyValueDO.value, 1, len(value)) == value
            )
            stmt = (
                delete(KeyValueDO)
                .where(
                    func.substr(KeyValueDO.key, 1, len(key_prefix)) == key_prefix,
                    value_clause,
                )
                .returning(KeyValueDO.key)
            )
            result = await session.execute(stmt)
            deleted_keys = [str(key) for key in result.scalars().all()]
            await session.commit()
            return deleted_keys

    async def delete_user_sessions(
        self,
        account: str,
        user_id: int,
    ) -> list[str]:
        """Atomically delete one user's sessions and historical aliases."""
        alias_prefix = "account-alias:"
        async with self._session_manager.session() as session:
            now = time.time()
            alias_stmt = select(KeyValueDO.key, KeyValueDO.expiry).where(
                func.substr(KeyValueDO.key, 1, len(alias_prefix)) == alias_prefix,
                KeyValueDO.value == str(user_id),
            )
            alias_result = await session.execute(alias_stmt)
            alias_rows = alias_result.all()
            alias_keys = [str(row.key) for row in alias_rows]
            aliases = {
                str(row.key).removeprefix(alias_prefix)
                for row in alias_rows
                if row.expiry >= now
            }

            current_alias_stmt = select(KeyValueDO.value, KeyValueDO.expiry).where(
                KeyValueDO.key == f"{alias_prefix}{account}"
            )
            current_alias_row = (
                await session.execute(current_alias_stmt)
            ).one_or_none()
            current_alias = (
                current_alias_row.value
                if current_alias_row is not None and current_alias_row.expiry >= now
                else None
            )

            accounts = aliases | {account}
            legacy_accounts = set(aliases)
            if current_alias is None or current_alias == str(user_id):
                legacy_accounts.add(account)

            user_id_suffix = f"|{user_id}"
            # V3 sessions carry the stable user ID, so revoke them by identity
            # regardless of which historical account name they contain. This
            # also covers account names whose alias belongs to an earlier user.
            first_separator = func.instr(KeyValueDO.value, "|")
            value_clauses = [
                and_(
                    first_separator > 0,
                    func.substr(KeyValueDO.value, first_separator + 1, 3) == "v3|",
                    func.substr(KeyValueDO.value, -len(user_id_suffix))
                    == user_id_suffix,
                )
            ]
            for session_account in accounts:
                if session_account not in legacy_accounts:
                    continue

                v2_prefix = f"{session_account}|v2|"
                legacy_prefix = f"{session_account}|"
                value_clauses.append(
                    func.substr(KeyValueDO.value, 1, len(v2_prefix)) == v2_prefix
                )
                value_clauses.append(
                    and_(
                        func.substr(KeyValueDO.value, 1, len(legacy_prefix))
                        == legacy_prefix,
                        func.instr(
                            func.substr(KeyValueDO.value, len(legacy_prefix) + 1),
                            "|",
                        )
                        == 0,
                    )
                )

            session_prefix = "session:"
            stmt = (
                delete(KeyValueDO)
                .where(
                    func.substr(KeyValueDO.key, 1, len(session_prefix))
                    == session_prefix,
                    or_(*value_clauses),
                )
                .returning(KeyValueDO.key)
            )
            result = await session.execute(stmt)
            deleted_keys = [str(key) for key in result.scalars().all()]
            if alias_keys:
                await session.execute(
                    delete(KeyValueDO).where(KeyValueDO.key.in_(alias_keys))
                )
            await session.commit()
            return deleted_keys

    async def pop_value(self, key: str) -> str | None:
        """Get and delete a value atomically."""
        async with self._session_manager.session() as session:
            # A single DELETE ... RETURNING statement prevents two concurrent
            # consumers from both observing the same one-time value.
            stmt = (
                delete(KeyValueDO)
                .where(KeyValueDO.key == key)
                .returning(KeyValueDO.value, KeyValueDO.expiry)
            )
            result = await session.execute(stmt)
            row = result.one_or_none()

            if not row:
                return None

            await session.commit()

            if time.time() > row.expiry:
                return None
            return str(row.value)

    async def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Atomically increment a value. Returns the new value.

        If key does not exist, it is created with value `amount`.
        If key exists, its value is incremented.
        TTL is effective only if a new key is created.
        """
        async with self._session_manager.session() as session:
            now = time.time()
            expiry = now + (ttl if ttl else DEFAULT_TTL)

            # 1. Cleanup expired key if any (enforce fresh start if expired)
            stmt_del_expired = (
                delete(KeyValueDO)
                .where(KeyValueDO.key == key)
                .where(KeyValueDO.expiry < now)
            )
            await session.execute(stmt_del_expired)

            # 2. Upsert with RETURNING
            # DB stores value as String. We cast to int for math, then back to string.
            # On Conflict (key exists), we update value. We DO NOT update expiry (Redis behavior).

            sql = text("""
                INSERT INTO key_values (key, value, expiry)
                VALUES (:key, :initial_val, :expiry)
                ON CONFLICT(key) DO UPDATE SET
                    value = CAST(CAST(value AS INTEGER) + :amount AS TEXT)
                RETURNING value
            """)

            result = await session.execute(
                sql,
                {
                    "key": key,
                    "initial_val": str(amount),
                    "expiry": expiry,
                    "amount": amount,
                },
            )
            row = result.first()
            await session.commit()

            if row:
                return int(row[0])
            return amount  # Should not happen with RETURNING
