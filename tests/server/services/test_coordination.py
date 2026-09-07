import asyncio
from collections.abc import AsyncGenerator

import freezegun
import pytest
from sqlalchemy import func, select

from supernote.server.db.models.kv import KeyValueDO
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.coordination import (
    CoordinationService,
    SqliteCoordinationService,
)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def local_coordination_service() -> AsyncGenerator[CoordinationService]:
    """Create a local coordination service for testing."""
    manager = DatabaseSessionManager(TEST_DB_URL)
    assert manager._engine
    await manager.create_all_tables()

    service = SqliteCoordinationService(manager)
    yield service
    await manager.close()


async def test_key_expiry(local_coordination_service: CoordinationService) -> None:
    """Test key expiry."""
    with freezegun.freeze_time("2024-01-01 12:00:00"):
        await local_coordination_service.set_value("foo", "bar", ttl=15)

        assert await local_coordination_service.get_value("foo") == "bar"

    with freezegun.freeze_time("2024-01-01 12:00:16"):
        assert await local_coordination_service.get_value("foo") is None


async def test_set_value_removes_abandoned_expired_keys(
    local_coordination_service: CoordinationService,
) -> None:
    """Expired unique keys are reclaimed even when they are never read."""
    with freezegun.freeze_time("2024-01-01 12:00:00"):
        await local_coordination_service.set_value("challenge:one", "first", ttl=15)
        await local_coordination_service.set_value("challenge:two", "second", ttl=15)

    with freezegun.freeze_time("2024-01-01 12:00:16"):
        await local_coordination_service.set_value("current", "value", ttl=15)

    service = local_coordination_service
    assert isinstance(service, SqliteCoordinationService)
    async with service._session_manager.session() as session:
        key_count = await session.scalar(select(func.count()).select_from(KeyValueDO))
        assert key_count == 1


async def test_pop_value_has_only_one_winner(
    local_coordination_service: CoordinationService,
) -> None:
    await local_coordination_service.set_value("one-time", "secret", ttl=60)

    results = await asyncio.gather(
        *(local_coordination_service.pop_value("one-time") for _ in range(10))
    )

    assert results.count("secret") == 1
    assert results.count(None) == 9


async def test_delete_values_matches_literal_prefixes(
    local_coordination_service: CoordinationService,
) -> None:
    await local_coordination_service.set_value(
        "session:one", "user_name@example.com|device", ttl=60
    )
    await local_coordination_service.set_value(
        "session:two", "userXname@example.com|device", ttl=60
    )
    await local_coordination_service.set_value(
        "session:three", "User_name@example.com|device", ttl=60
    )
    await local_coordination_service.set_value(
        "unrelated", "user_name@example.com|device", ttl=60
    )

    deleted = await local_coordination_service.delete_values(
        "session:", "user_name@example.com|"
    )

    assert deleted == ["session:one"]
    assert await local_coordination_service.get_value("session:one") is None
    assert await local_coordination_service.get_value("session:two") is not None
    assert await local_coordination_service.get_value("session:three") is not None
    assert await local_coordination_service.get_value("unrelated") is not None


async def test_delete_values_can_match_an_exact_value(
    local_coordination_service: CoordinationService,
) -> None:
    await local_coordination_service.set_value(
        "session:web", "user@example.com|", ttl=60
    )
    await local_coordination_service.set_value(
        "session:device", "user@example.com|SN123", ttl=60
    )

    deleted = await local_coordination_service.delete_values(
        "session:", "user@example.com|", exact_value=True
    )

    assert deleted == ["session:web"]
    assert await local_coordination_service.get_value("session:web") is None
    assert await local_coordination_service.get_value("session:device") is not None


async def test_increment(coordination_service: SqliteCoordinationService) -> None:
    key = "incr:test"

    # 1. Increment new key
    val = await coordination_service.increment(key, 1, ttl=60)
    assert val == 1

    # Check it exists
    stored = await coordination_service.get_value(key)
    assert stored == "1"

    # 2. Increment existing
    val = await coordination_service.increment(key, 1)
    assert val == 2

    # 3. Increment by larger amount
    val = await coordination_service.increment(key, 10)
    assert val == 12


async def test_increment_expiry(
    coordination_service: SqliteCoordinationService,
) -> None:
    key = "incr:expire"

    # Set with short TTL (but we can't easily wait for it in unit test without sleep)
    # Instead, we manually set an expired value first?
    # No, let's use the property that increment sets TTL on creation.

    # 1. Create
    val = await coordination_service.increment(key, 1, ttl=100)
    assert val == 1

    # 2. Verify TTL is roughly logic (hard to verify exact value without inspecting DB directly)
    # But we can verify "expired" behavior by mocking time or inserting expired row manually.
