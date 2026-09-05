import base64
import hashlib

import pytest
import yarl
from aiohttp.test_utils import TestClient

from supernote.client.client import Client
from supernote.client.login_client import LoginClient
from supernote.server.services.coordination import CoordinationService
from tests.server.conftest import TEST_PASSWORD, TEST_USERNAME


@pytest.fixture(name="login_client")
async def supernote_login_client_fixture(client: TestClient) -> LoginClient:
    base_url = str(client.make_url("/"))
    real_client = Client(client.session, host=base_url)
    return LoginClient(real_client)


def calculate_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


async def test_scenario_oauth_cold_login(
    client: TestClient,
    login_client: LoginClient,
    create_test_user: None,
) -> None:
    """
    Scenario 1: Cold Start (Not Logged In) - Complete Flow
    1. User clicks 'Login with Supernote' and is redirected to Bridge -> Login Page.
    2. User performs actual login via API.
    3. Frontend 'Resumes Session' with new token.
    4. Code exchange completes successfully.
    """
    # 1. Authorize -> Bridge -> Login Page
    verifier = "v" * 50
    client_id = "http://localhost:3000"
    redirect_uri = "http://localhost:3000/callback"

    resp1 = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": "cold-state",
            "scope": "supernote:all",
            "code_challenge": calculate_s256(verifier),
            "code_challenge_method": "S256",
        },
        allow_redirects=False,
    )
    assert resp1.status in (302, 307)
    bridge_path = yarl.URL(resp1.headers["Location"]).path_qs

    resp2 = await client.get(bridge_path, allow_redirects=False)
    assert resp2.status in (302, 307)
    assert "/#login" in resp2.headers["Location"]
    assert "return_to" in resp2.headers["Location"]

    # 2. Real User Login
    fresh_token = await login_client.login(TEST_USERNAME, TEST_PASSWORD)
    assert fresh_token

    # 3. Resume Session (POST to Bridge)
    resp3 = await client.post(
        bridge_path,
        headers={"x-access-token": fresh_token},
        data={"consent": "approve"},
        allow_redirects=False,
    )
    assert resp3.status == 200
    callback_url = (await resp3.json())["redirect_url"]

    # 4. Token Exchange
    code = yarl.URL(callback_url).query["code"]
    token_resp = await client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )
    assert token_resp.status == 200
    assert "access_token" in await token_resp.json()


async def test_scenario_oauth_warm_session(
    client: TestClient,
    login_client: LoginClient,
    create_test_user: None,
) -> None:
    """
    Scenario 2: Warm Session (Already Logged In)
    1. User is already logged in (simulated by having a token).
    2. User triggers OAuth flow.
    3. Frontend fast-forwards through bridge using existing token.
    """
    # 1. Warm up session
    token = await login_client.login(TEST_USERNAME, TEST_PASSWORD)

    # 2. Authorize
    verifier = "v" * 50
    client_id = "http://localhost:3000"
    redirect_uri = "http://localhost:3000/callback"

    resp1 = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": "warm-state",
            "code_challenge": calculate_s256(verifier),
            "code_challenge_method": "S256",
        },
        allow_redirects=False,
    )
    bridge_path = yarl.URL(resp1.headers["Location"]).path_qs

    # 3. Bridge Fast-Forward
    resp2 = await client.post(
        bridge_path,
        headers={"x-access-token": token},
        data={"consent": "approve"},
        allow_redirects=False,
    )
    assert resp2.status == 200
    callback_url = (await resp2.json())["redirect_url"]

    # 4. Exchange
    code = yarl.URL(callback_url).query["code"]
    token_resp = await client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )
    assert token_resp.status == 200


async def test_scenario_security_edge_cases(
    client: TestClient,
    login_client: LoginClient,
    create_test_user: None,
) -> None:
    """
    Scenario 3: Security & Error Handling
    1. Invalid Token handling on Bridge (POST vs GET).
    2. IndieAuth validation (Valid vs Mismatch).
    """
    # Setup
    params = {
        "client_id": "http://localhost:3000/callback",
        "redirect_uri": "http://localhost:3000/callback",
        "code_challenge": calculate_s256("v" * 50),
        "code_challenge_method": "S256",
    }

    # 1a. POST with Invalid Token -> 401 Unauthorized (API Mode)
    resp_invalid = await client.post(
        "/login-bridge",
        params=params,
        headers={"x-access-token": "bad-token"},
    )
    assert resp_invalid.status == 401

    # 1b. GET with Invalid Token -> Redirect to Login (Browser Mode)
    resp_redirect = await client.get(
        "/login-bridge",
        params=params,
        headers={"x-access-token": "bad-token"},
        allow_redirects=False,
    )
    assert resp_redirect.status in (302, 307)
    assert "/#login" in resp_redirect.headers["Location"]

    # 2a. IndieAuth accepts callbacks on the client ID's origin.
    token = await login_client.login(TEST_USERNAME, TEST_PASSWORD)
    resp_indie = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": "http://localhost:5000",
            "redirect_uri": "http://localhost:5000/callback",
            "code_challenge": calculate_s256("v" * 50),
            "code_challenge_method": "S256",
        },
        allow_redirects=False,
    )
    bridge_path = yarl.URL(resp_indie.headers["Location"]).path_qs

    resp_indie_bridge = await client.post(
        bridge_path,
        headers={"x-access-token": token},
        data={"consent": "approve"},
    )
    assert resp_indie_bridge.status == 200
    assert "localhost:5000/callback" in (await resp_indie_bridge.json())["redirect_url"]

    # 2b. IndieAuth Mismatch
    resp_mismatch = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": "http://localhost:3000",
            "redirect_uri": "http://evil.com/callback",
            "code_challenge": calculate_s256("v" * 50),
            "code_challenge_method": "S256",
        },
        allow_redirects=True,
    )
    assert resp_mismatch.status == 400

    # A string-prefix lookalike must not count as the same origin.
    resp_prefix_attack = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": "http://localhost:3000",
            "redirect_uri": "http://localhost:3000.evil.test/callback",
            "code_challenge": calculate_s256("v" * 50),
            "code_challenge_method": "S256",
        },
        allow_redirects=True,
    )
    assert resp_prefix_attack.status == 400


async def test_oauth_requires_consent_and_pkce(
    client: TestClient,
    login_client: LoginClient,
    create_test_user: None,
) -> None:
    token = await login_client.login(TEST_USERNAME, TEST_PASSWORD)
    callback = "http://localhost:3000/callback"

    missing_pkce = await client.post(
        "/login-bridge",
        params={"client_id": callback, "redirect_uri": callback},
        headers={"x-access-token": token},
        data={"consent": "approve"},
    )
    assert missing_pkce.status == 400

    verifier = "v" * 50
    authorize = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": callback,
            "redirect_uri": callback,
            "code_challenge": calculate_s256(verifier),
            "code_challenge_method": "S256",
        },
        allow_redirects=False,
    )
    bridge_path = yarl.URL(authorize.headers["Location"]).path_qs

    consent = await client.post(
        bridge_path, headers={"x-access-token": token}
    )
    assert consent.status == 200
    consent_body = await consent.json()
    assert consent_body["consent_required"] is True
    assert "redirect_url" not in consent_body


async def test_oauth_code_refresh_and_revocation_are_one_use(
    client: TestClient,
    login_client: LoginClient,
    coordination_service: CoordinationService,
    create_test_user: None,
) -> None:
    session_token = await login_client.login(TEST_USERNAME, TEST_PASSWORD)
    callback = "http://localhost:3000/callback"
    verifier = "v" * 50
    authorize = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": callback,
            "redirect_uri": callback,
            "code_challenge": calculate_s256(verifier),
            "code_challenge_method": "S256",
        },
        allow_redirects=False,
    )
    bridge = await client.post(
        yarl.URL(authorize.headers["Location"]).path_qs,
        headers={"x-access-token": session_token},
        data={"consent": "approve"},
    )
    code = yarl.URL((await bridge.json())["redirect_url"]).query["code"]
    exchange_data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": callback,
        "redirect_uri": callback,
        "code_verifier": verifier,
    }

    first_exchange = await client.post("/token", data=exchange_data)
    assert first_exchange.status == 200
    first_tokens = await first_exchange.json()
    assert (await client.post("/token", data=exchange_data)).status == 400

    refresh_data = {
        "grant_type": "refresh_token",
        "refresh_token": first_tokens["refresh_token"],
        "client_id": callback,
    }
    refreshed = await client.post("/token", data=refresh_data)
    assert refreshed.status == 200
    refreshed_tokens = await refreshed.json()
    assert refreshed_tokens["refresh_token"] != first_tokens["refresh_token"]
    assert (await client.post("/token", data=refresh_data)).status == 400
    assert (
        await coordination_service.get_value(
            f"mcp:access_token:{first_tokens['access_token']}"
        )
        is None
    )

    revoked = await client.post(
        "/revoke",
        data={
            "token": refreshed_tokens["refresh_token"],
            "token_type_hint": "refresh_token",
            "client_id": callback,
            "client_secret": "",
        },
    )
    assert revoked.status == 200
    assert (
        await coordination_service.get_value(
            f"mcp:access_token:{refreshed_tokens['access_token']}"
        )
        is None
    )
    assert (
        await coordination_service.get_value(
            f"mcp:refresh_token:{refreshed_tokens['refresh_token']}"
        )
        is None
    )
    assert (
        await client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refreshed_tokens["refresh_token"],
                "client_id": callback,
            },
        )
    ).status == 400
