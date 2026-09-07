import hashlib
import hmac
import time
from unittest.mock import patch

import jwt
import pytest
from sqlalchemy import select

from supernote.models.auth import Equipment
from supernote.models.user import UpdateEmailDTO, UpdatePasswordDTO, UserRegisterDTO
from supernote.server.config import AuthConfig, ServerConfig
from supernote.server.db.models.user import UserDO
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.coordination import CoordinationService
from supernote.server.services.user import JWT_ALGORITHM, UserService
from supernote.server.utils.hashing import hash_with_salt


@pytest.fixture
def test_users() -> list[str]:
    """Fixture to clear all test users.

    We use this to verify we can bootstrap properly.
    """
    return []


async def test_bootstrap_first_user_is_admin(
    session_manager: DatabaseSessionManager, coordination_service: CoordinationService
) -> None:
    """Test that the first registered user becomes an admin."""
    config = AuthConfig(enable_registration=True)
    service = UserService(config, coordination_service, session_manager)

    # Register first user (should be admin)
    pw_md5 = hashlib.md5(b"password").hexdigest()
    dto1 = UserRegisterDTO(
        email="admin@example.com", password=pw_md5, user_name="Admin"
    )
    user1 = await service.register(dto1)
    assert user1.is_admin is True

    # Register second user (should NOT be admin)
    dto2 = UserRegisterDTO(email="user@example.com", password=pw_md5, user_name="User")
    user2 = await service.register(dto2)
    assert user2.is_admin is False


async def test_bootstrap_bypasses_disabled_registration(
    session_manager: DatabaseSessionManager,
    coordination_service: CoordinationService,
) -> None:
    """Test that bootstrapping works even if registration is disabled."""
    # Config has registration DISABLED
    config = AuthConfig(enable_registration=False)
    service = UserService(config, coordination_service, session_manager)

    # Register first user (should succeed because of bootstrap)
    pw_md5 = hashlib.md5(b"password").hexdigest()
    dto1 = UserRegisterDTO(
        email="bootstrap@example.com", password=pw_md5, user_name="Bootstrap"
    )
    user1 = await service.register(dto1)

    assert user1.is_admin is True

    # Register second user (should FAIL because registration is disabled)
    dto2 = UserRegisterDTO(email="fail@example.com", password=pw_md5, user_name="Fail")
    with pytest.raises(ValueError, match="Registration is disabled"):
        await service.register(dto2)


async def test_admin_create_user_bypass(
    session_manager: DatabaseSessionManager, coordination_service: CoordinationService
) -> None:
    """Test that create_user allows creating users when disabled."""
    config = AuthConfig(enable_registration=False)
    service = UserService(config, coordination_service, session_manager)

    # Bootstrap first
    pw_md5 = hashlib.md5(b"pw").hexdigest()
    await service.register(
        UserRegisterDTO(email="admin@example.com", password=pw_md5, user_name="Admin")
    )

    # Explicitly use create_user (Admin action simulation)
    dto2 = UserRegisterDTO(email="new@example.com", password=pw_md5, user_name="New")
    user2 = await service.create_user(dto2)

    assert user2.email == "new@example.com"


async def test_retrieve_password_success(
    user_service: UserService,
    session_manager: DatabaseSessionManager,
    coordination_service: CoordinationService,
    server_config: ServerConfig,
) -> None:
    # Setup
    email = "forgot@example.com"
    old_pw = hashlib.md5(b"old").hexdigest()
    new_pw = hashlib.md5(b"new").hexdigest()

    # Create user directly in DB
    async with session_manager.session() as session:
        user = UserDO(
            email=email, password_md5=old_pw, display_name="Forgot", is_active=True
        )
        session.add(user)
        await session.commit()

    # Validation: Pass scalars
    result: bool = await user_service.retrieve_password(email, new_pw)
    assert result is True

    # Verify in DB
    async with session_manager.session() as session:
        user_result = await session.execute(select(UserDO).where(UserDO.email == email))
        user = user_result.scalar_one()
        assert user.password_md5 == new_pw


async def test_retrieve_password_invalid_md5(user_service: UserService) -> None:
    with pytest.raises(ValueError, match="Invalid password format"):
        await user_service.retrieve_password("user@example.com", "plain_text_password")


async def test_retrieve_password_user_not_found(user_service: UserService) -> None:
    result = await user_service.retrieve_password(
        "nonexistent@example.com", hashlib.md5(b"pw").hexdigest()
    )
    assert result is False


async def test_register_login_flow(user_service: UserService) -> None:
    """Register and login a user."""
    # Register
    pw_md5 = hashlib.md5(b"password123").hexdigest()
    dto = UserRegisterDTO(
        email="unique_test_reg@example.com",
        password=pw_md5,
        user_name="Test User",
    )
    user = await user_service.register(dto)
    assert user.id is not None
    assert user.email == "unique_test_reg@example.com"

    # Login

    pw_hash = hashlib.md5(b"password123").hexdigest()

    # Need to mock the challenge flow
    code, ts = await user_service.generate_random_code("unique_test_reg@example.com")

    # Client logic: hash(md5(pw), code)
    client_hash = hash_with_salt(pw_hash, code)
    login_vo = await user_service.login(
        "unique_test_reg@example.com",
        client_hash,
        ts,
        equipment_no="dev1",
        ip="127.0.0.1",
        login_method="2",
    )
    assert login_vo is not None
    assert login_vo.token is not None

    # Verify records
    records, total = await user_service.query_login_records(
        "unique_test_reg@example.com", 1, 10
    )
    assert total == 1
    assert records[0].ip == "127.0.0.1"


async def test_login_challenges_are_independent_and_one_time(
    user_service: UserService,
    server_config: ServerConfig,
    coordination_service: CoordinationService,
    session_manager: DatabaseSessionManager,
) -> None:
    email = "challenge@example.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    await user_service.register(UserRegisterDTO(email=email, password=password_md5))

    # Issuing a second challenge in the same millisecond must not overwrite
    # the first one or change the timestamp field's epoch-millisecond semantics.
    now = time.time()
    expected_timestamp_ms = int(now * 1000)
    second_user_service = UserService(
        server_config.auth, coordination_service, session_manager
    )
    with patch("supernote.server.services.user.time.time", return_value=now):
        first_code, first_timestamp = await user_service.generate_random_code(email)
        second_code, second_timestamp = await second_user_service.generate_random_code(
            email
        )
    assert first_timestamp == str(expected_timestamp_ms)
    assert second_timestamp == str(expected_timestamp_ms + 1)

    first_proof = hash_with_salt(password_md5, first_code)
    second_proof = hash_with_salt(password_md5, second_code)
    with patch(
        "supernote.server.services.user.hmac.compare_digest",
        wraps=hmac.compare_digest,
    ) as compare_digest:
        assert await user_service.verify_login_hash(email, first_proof, first_timestamp)
        compare_digest.assert_called_once()

    # A captured proof cannot be replayed after the atomic consume.
    assert not await user_service.verify_login_hash(email, first_proof, first_timestamp)
    assert await user_service.verify_login_hash(email, second_proof, second_timestamp)


async def test_failed_login_proof_consumes_challenge(user_service: UserService) -> None:
    email = "failed-challenge@example.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    await user_service.register(UserRegisterDTO(email=email, password=password_md5))
    code, timestamp = await user_service.generate_random_code(email)

    assert not await user_service.verify_login_hash(email, "wrong", timestamp)
    assert not await user_service.verify_login_hash(
        email, hash_with_salt(password_md5, code), timestamp
    )


async def test_update_password(
    user_service: UserService, coordination_service: CoordinationService
) -> None:
    """Update a user's password."""
    # Register
    old_md5 = hashlib.md5(b"old").hexdigest()
    await user_service.register(UserRegisterDTO(email="pw@test.com", password=old_md5))
    await coordination_service.set_value(
        "session:web-token",
        f"pw@test.com|v3|{Equipment.WEB.value}|WEB|1",
        ttl=3600,
    )
    await coordination_service.set_value(
        "session:device-token",
        f"pw@test.com|v3|{Equipment.TERMINAL.value}|device|1",
        ttl=3600,
    )
    await coordination_service.set_value(
        "session:legacy-ambiguous-token", "pw@test.com|", ttl=3600
    )
    await coordination_service.set_value(
        "session:legacy-device-token", "pw@test.com|SN123", ttl=3600
    )

    # Update
    new_md5 = hashlib.md5(b"new").hexdigest()
    await user_service.update_password(
        "pw@test.com", UpdatePasswordDTO(password=new_md5)
    )
    assert await coordination_service.get_value("session:web-token") is None
    assert await coordination_service.get_value("session:device-token") is not None
    assert (
        await coordination_service.get_value("session:legacy-ambiguous-token")
        is not None
    )
    assert (
        await coordination_service.get_value("session:legacy-device-token") is not None
    )

    # Login with old fails (we'd need a full login flow to test, strictly speaking)
    # But we can check DB directly via service internal method if we exposed it, or just trust update worked.
    # Let's try to login with NEW password
    code, ts = await user_service.generate_random_code("pw@test.com")
    new_hash = hashlib.md5(b"new").hexdigest()
    client_hash = hash_with_salt(new_hash, code)

    login_vo = await user_service.login("pw@test.com", client_hash, ts)
    assert login_vo is not None


async def test_update_email_preserves_device_session(
    user_service: UserService,
    coordination_service: CoordinationService,
    server_config: ServerConfig,
) -> None:
    old_email = "old-email@test.com"
    new_email = "new-email@test.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    await user_service.register(UserRegisterDTO(email=old_email, password=password_md5))

    code, timestamp = await user_service.generate_random_code(old_email)
    login = await user_service.login(
        old_email,
        hash_with_salt(password_md5, code),
        timestamp,
        equipment=Equipment.TERMINAL,
        equipment_no="SN123",
    )
    assert login is not None

    legacy_token = jwt.encode(
        {"sub": old_email},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{legacy_token}", f"{old_email}|SN-LEGACY", ttl=3600
    )

    v2_token = jwt.encode(
        {"sub": old_email, "equipment_no": "SN-V2", "nonce": "v2"},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{v2_token}",
        f"{old_email}|v2|{Equipment.TERMINAL.value}|SN-V2",
        ttl=3600,
    )

    await user_service.update_email(old_email, UpdateEmailDTO(email=new_email))

    for token, equipment_no in (
        (login.token, "SN123"),
        (legacy_token, "SN-LEGACY"),
        (v2_token, "SN-V2"),
    ):
        device_session = await user_service.verify_token(token)
        assert device_session is not None
        assert device_session.email == new_email
        assert device_session.equipment_no == equipment_no


async def test_update_email_publishes_alias_before_revoking_web_sessions(
    user_service: UserService,
    coordination_service: CoordinationService,
    server_config: ServerConfig,
) -> None:
    old_email = "rename-old@test.com"
    new_email = "rename-new@test.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    await user_service.register(UserRegisterDTO(email=old_email, password=password_md5))

    token = jwt.encode(
        {"sub": old_email},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{token}", f"{old_email}|SN-LEGACY", ttl=3600
    )

    original_revoke = user_service.revoke_web_sessions

    async def assert_device_session_is_valid(account: str) -> None:
        device_session = await user_service.verify_token(token)
        assert device_session is not None
        assert device_session.email == new_email
        await original_revoke(account)

    with patch.object(
        user_service,
        "revoke_web_sessions",
        side_effect=assert_device_session_is_valid,
    ):
        await user_service.update_email(old_email, UpdateEmailDTO(email=new_email))


async def test_update_email_does_not_reassign_an_existing_alias(
    user_service: UserService,
    coordination_service: CoordinationService,
    server_config: ServerConfig,
) -> None:
    old_email = "reused@test.com"
    first_email = "first-renamed@test.com"
    second_email = "second-renamed@test.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    await user_service.register(UserRegisterDTO(email=old_email, password=password_md5))

    legacy_token = jwt.encode(
        {"sub": old_email, "owner": "first"},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{legacy_token}", f"{old_email}|SN-LEGACY", ttl=3600
    )

    await user_service.update_email(old_email, UpdateEmailDTO(email=first_email))
    await user_service.register(UserRegisterDTO(email=old_email, password=password_md5))
    await user_service.update_email(old_email, UpdateEmailDTO(email=second_email))

    session = await user_service.verify_token(legacy_token)
    assert session is not None
    assert session.email == first_email


async def test_unregister(user_service: UserService) -> None:
    """Unregister a user."""
    pw_md5 = hashlib.md5(b"pw").hexdigest()
    await user_service.register(UserRegisterDTO(email="del@test.com", password=pw_md5))
    assert await user_service.check_user_exists("del@test.com")

    await user_service.unregister("del@test.com")
    assert not await user_service.check_user_exists("del@test.com")


async def test_unregister_revokes_renamed_sessions_before_id_reuse(
    user_service: UserService,
    coordination_service: CoordinationService,
    server_config: ServerConfig,
) -> None:
    old_email = "deleted-old@test.com"
    current_email = "deleted-current@test.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    deleted_user = await user_service.register(
        UserRegisterDTO(email=old_email, password=password_md5)
    )
    legacy_token = jwt.encode(
        {"sub": old_email},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{legacy_token}", f"{old_email}|SN-LEGACY", ttl=3600
    )

    await user_service.update_email(old_email, UpdateEmailDTO(email=current_email))
    assert await user_service.verify_token(legacy_token) is not None

    await user_service.unregister(current_email)
    assert await coordination_service.get_value(f"session:{legacy_token}") is None
    assert await coordination_service.get_value(f"account-alias:{old_email}") is None

    replacement = await user_service.register(
        UserRegisterDTO(email="replacement@test.com", password=password_md5)
    )
    assert replacement.id == deleted_user.id
    assert await user_service.verify_token(legacy_token) is None


async def test_unregister_does_not_revoke_new_owner_of_reused_email(
    user_service: UserService,
    coordination_service: CoordinationService,
    server_config: ServerConfig,
) -> None:
    old_email = "shared-history@test.com"
    renamed_email = "original-owner@test.com"
    password_md5 = hashlib.md5(b"password").hexdigest()
    original = await user_service.register(
        UserRegisterDTO(email=old_email, password=password_md5)
    )

    original_token = jwt.encode(
        {"sub": old_email, "owner": "original"},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{original_token}", f"{old_email}|SN-LEGACY", ttl=3600
    )
    await user_service.update_email(old_email, UpdateEmailDTO(email=renamed_email))

    replacement = await user_service.register(
        UserRegisterDTO(email=old_email, password=password_md5)
    )
    replacement_token = jwt.encode(
        {"sub": old_email, "owner": "replacement"},
        server_config.auth.secret_key,
        algorithm=JWT_ALGORITHM,
    )
    await coordination_service.set_value(
        f"session:{replacement_token}",
        f"{old_email}|v3|{Equipment.TERMINAL.value}|SN-NEW|{replacement.id}",
        ttl=3600,
    )

    await user_service.unregister(renamed_email)

    assert await user_service.verify_token(original_token) is None
    replacement_session = await user_service.verify_token(replacement_token)
    assert replacement_session is not None
    assert replacement_session.user_id == replacement.id
    assert replacement.id != original.id


async def test_unregister_revokes_v3_session_with_alias_owned_by_another_user(
    user_service: UserService,
    coordination_service: CoordinationService,
) -> None:
    shared_email = "reused-alias@test.com"
    password_md5 = hashlib.md5(b"password").hexdigest()

    await user_service.register(
        UserRegisterDTO(email=shared_email, password=password_md5)
    )
    await user_service.update_email(
        shared_email, UpdateEmailDTO(email="first-owner@test.com")
    )

    second_owner = await user_service.register(
        UserRegisterDTO(email=shared_email, password=password_md5)
    )
    code, timestamp = await user_service.generate_random_code(shared_email)
    login = await user_service.login(
        shared_email,
        hash_with_salt(password_md5, code),
        timestamp,
        equipment=Equipment.TERMINAL,
        equipment_no="SN-SECOND",
    )
    assert login is not None
    await user_service.update_email(
        shared_email, UpdateEmailDTO(email="second-owner@test.com")
    )

    await user_service.unregister("second-owner@test.com")

    assert await coordination_service.get_value(f"session:{login.token}") is None
    replacement = await user_service.register(
        UserRegisterDTO(email="replacement@test.com", password=password_md5)
    )
    assert replacement.id == second_owner.id
    assert await user_service.verify_token(login.token) is None


async def test_token_expiration_by_equipment(user_service: UserService) -> None:
    """Verify that token expiration depends on equipment type."""
    email = "exp_test@example.com"
    pw_md5 = hashlib.md5(b"password123").hexdigest()
    await user_service.register(UserRegisterDTO(email=email, password=pw_md5))

    # 1. Login as WEB (Default)
    code, ts = await user_service.generate_random_code(email)
    client_hash = hash_with_salt(pw_md5, code)

    login_web = await user_service.login(
        email, client_hash, ts, equipment=Equipment.WEB
    )
    assert login_web is not None

    # Decode JWT to check exp
    payload_web = jwt.decode(login_web.token, options={"verify_signature": False})
    exp_web = payload_web["exp"]
    iat_web = payload_web["iat"]

    # Should be significantly less than 10 years
    # (Usually 24h, but depends on test environment)
    assert 60 < (exp_web - iat_web) < 10 * 365 * 24 * 3600

    # 2. Login as TERMINAL (Device)
    code, ts = await user_service.generate_random_code(email)
    client_hash = hash_with_salt(pw_md5, code)

    login_term = await user_service.login(
        email, client_hash, ts, equipment=Equipment.TERMINAL
    )
    assert login_term is not None

    payload_term = jwt.decode(login_term.token, options={"verify_signature": False})
    exp_term = payload_term["exp"]
    iat_term = payload_term["iat"]

    ten_years_seconds = 10 * 365 * 24 * 3600
    assert (exp_term - iat_term) == ten_years_seconds

    # 3. Login as APP (Companion App)
    code, ts = await user_service.generate_random_code(email)
    client_hash = hash_with_salt(pw_md5, code)

    login_app = await user_service.login(
        email, client_hash, ts, equipment=Equipment.APP
    )
    assert login_app is not None

    payload_app = jwt.decode(login_app.token, options={"verify_signature": False})
    exp_app = payload_app["exp"]
    iat_app = payload_app["iat"]

    assert (exp_app - iat_app) == ten_years_seconds


async def test_register_invalid_email(user_service: UserService) -> None:
    # Test cases for invalid emails
    invalid_emails = [
        "plainaddress",
        "#@%^%#$@#$@#.com",
        "@example.com",
        "Joe Smith <email@example.com>",
        "email.example.com",
        "email@example@example.com",
        ".email@example.com",
        "email.@example.com",
        "email..email@example.com",
        "email@example.com (Joe Smith)",
        "email@example",
        "email@-example.com",
        # "email@example.web", # This one might actually pass the simple regex depending on strictness, but let's test common bad ones
        # "email@111.222.333.44444",
        "email@example..com",
        "Abc..123@example.com",
    ]

    pw_md5 = hashlib.md5(b"password").hexdigest()
    for email in invalid_emails:
        try:
            await user_service.register(
                UserRegisterDTO(email=email, password=pw_md5, user_name="Test User")
            )
            pytest.fail(f"Email '{email}' should have failed validation but passed")
        except ValueError as e:
            assert str(e) == "Invalid email address format"


async def test_register_valid_email(user_service: UserService) -> None:
    # Test cases for valid emails
    valid_emails = [
        "email@example.com",
        "firstname.lastname@example.com",
        "email@subdomain.example.com",
        "firstname+lastname@example.com",
        "email@123.123.123.123",
        "1234567890@example.com",
        "email@example-one.com",
        "_______@example.com",
        "email@example.name",
        "email@example.museum",
        "email@example.co.jp",
        "firstname-lastname@example.com",
    ]

    pw_md5 = hashlib.md5(b"password").hexdigest()
    for email in valid_emails:
        # Should not raise
        try:
            user = await user_service.register(
                UserRegisterDTO(email=email, password=pw_md5, user_name="Test User")
            )
            assert user.email == email
        except ValueError as e:
            pytest.fail(f"Valid email '{email}' failed validation: {e}")
        # We don't bother cleanup since we're using unique email addresses
