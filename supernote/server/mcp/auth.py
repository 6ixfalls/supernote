"""An OAuth Authorization server for Supernote MCP."""

import logging
import re
import secrets
import time
from typing import override
from urllib.parse import quote, urlencode, urlparse

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import RevocationOptions
from mcp.shared.auth import (
    InvalidRedirectUriError,
    InvalidScopeError,
    OAuthClientInformationFull,
    OAuthToken,
)
from pydantic import AnyHttpUrl, AnyUrl, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from supernote.server.services.coordination import CoordinationService
from supernote.server.services.user import UserService
from supernote.server.utils.auth_utils import get_token_from_request

from .models import (
    SupernoteAccessToken,
    SupernoteAuthorizationCode,
    SupernoteRefreshToken,
)

_LOGGER = logging.getLogger(__name__)

_ACCESS_TOKEN_TTL = 3600
_REFRESH_TOKEN_TTL = 86400 * 30
_PKCE_S256_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


class SupernoteOAuthClientInformationFull(OAuthClientInformationFull):
    """OAuth 2.1 Client Information for Supernote MCP."""

    @override
    def validate_redirect_uri(self, redirect_uri: AnyUrl | None) -> AnyUrl:
        """Require an exact URI or a same-origin URL-client callback."""
        if redirect_uri is None:
            raise InvalidRedirectUriError("Redirect URI must be specified")
        redirect_uri_str = str(redirect_uri)
        for registered_redirect_uri in self.redirect_uris or ():
            if secrets.compare_digest(
                redirect_uri_str, str(registered_redirect_uri)
            ):
                return redirect_uri

            # URL client identifiers do not have a registration document in this
            # provider. Permit callback paths on that client's origin without
            # restoring the unsafe string-prefix matching previously used here.
            registered = urlparse(str(registered_redirect_uri))
            requested = urlparse(redirect_uri_str)
            registered_port = registered.port or (
                443 if registered.scheme == "https" else 80
            )
            requested_port = requested.port or (
                443 if requested.scheme == "https" else 80
            )
            if (
                registered.scheme in ("http", "https")
                and requested.scheme == registered.scheme
                and requested.hostname == registered.hostname
                and requested_port == registered_port
            ):
                return redirect_uri
        raise InvalidRedirectUriError(
            f"Redirect URI '{redirect_uri}' not in allowed list"
        )


class SupernoteOAuthProvider(
    OAuthAuthorizationServerProvider[
        SupernoteAuthorizationCode, SupernoteRefreshToken, SupernoteAccessToken
    ]
):
    """OAuth 2.1 Provider for Supernote MCP."""

    def __init__(
        self,
        user_service: UserService,
        coordination_service: CoordinationService,
        issuer_url: str,
    ):
        self.user_service = user_service
        self.issuer_url = issuer_url
        self._coordination = coordination_service

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Retrieve client information by client ID."""
        # Support dynamic IndieAuth-style clients (Client ID is a URL)
        try:
            parsed = urlparse(client_id)
        except ValueError:
            return None
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return SupernoteOAuthClientInformationFull(
                client_id=client_id,
                redirect_uris=[AnyHttpUrl(client_id)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="supernote:all",
                token_endpoint_auth_method="none",
            )
        return None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """Saves client information as part of registering it."""
        raise NotImplementedError("Dynamic client registration not supported")

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Called as part of the /authorize endpoint."""
        # We redirect to a bridge that handles login/session check, passing
        # the full set of OAuth params.
        query_params = params.model_dump(exclude_none=True)
        if params.scopes is not None:
            query_params["scopes"] = " ".join(params.scopes)
        query_params["code_challenge_method"] = "S256"
        query_params["client_id"] = client.client_id
        query = urlencode(query_params)
        return f"{self.issuer_url}/login-bridge?{query}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> SupernoteAuthorizationCode | None:
        """Loads an AuthorizationCode by its code string."""
        key = f"mcp:auth_code:{authorization_code}"
        data = await self._coordination.get_value(key)
        if not data:
            return None
        return SupernoteAuthorizationCode.model_validate_json(data)

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: SupernoteAuthorizationCode,
    ) -> OAuthToken:
        """Exchanges an authorization code for an access token and refresh token."""
        key = f"mcp:auth_code:{authorization_code.code}"
        consumed = await self._coordination.pop_value(key)
        if consumed is None:
            raise TokenError(
                error="invalid_grant",
                error_description="authorization code has already been used",
            )

        stored_code = SupernoteAuthorizationCode.model_validate_json(consumed)
        if stored_code != authorization_code:
            raise TokenError(
                error="invalid_grant",
                error_description="authorization code is invalid",
            )

        access_token = SupernoteAccessToken(
            token=secrets.token_urlsafe(32),
            user_id=authorization_code.user_id,
            client_id=authorization_code.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time() + _ACCESS_TOKEN_TTL),
        )
        refresh_token = SupernoteRefreshToken(
            token=secrets.token_urlsafe(32),
            user_id=authorization_code.user_id,
            client_id=authorization_code.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time() + _REFRESH_TOKEN_TTL),
        )

        # Store tokens
        await self._coordination.set_value(
            f"mcp:access_token:{access_token.token}",
            access_token.model_dump_json(),
            ttl=_ACCESS_TOKEN_TTL,
        )
        await self._coordination.set_value(
            f"mcp:refresh_token:{refresh_token.token}",
            refresh_token.model_dump_json(),
            ttl=_REFRESH_TOKEN_TTL,
        )
        await self._store_token_pair(access_token.token, refresh_token.token)

        return OAuthToken(
            access_token=access_token.token,
            refresh_token=refresh_token.token,
            token_type="Bearer",
            expires_in=_ACCESS_TOKEN_TTL,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> SupernoteRefreshToken | None:
        """Loads a RefreshToken by its token string."""
        key = f"mcp:refresh_token:{refresh_token}"
        data = await self._coordination.get_value(key)
        if not data:
            return None
        return SupernoteRefreshToken.model_validate_json(data)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: SupernoteRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Exchanges a refresh token for an access token and refresh token."""
        refresh_key = f"mcp:refresh_token:{refresh_token.token}"
        consumed = await self._coordination.pop_value(refresh_key)
        if consumed is None:
            raise TokenError(
                error="invalid_grant",
                error_description="refresh token has already been used",
            )

        stored_refresh = SupernoteRefreshToken.model_validate_json(consumed)
        if stored_refresh != refresh_token:
            raise TokenError(
                error="invalid_grant",
                error_description="refresh token is invalid",
            )

        # Invalidate the access token paired with the rotated refresh token.
        old_access_token = await self._coordination.pop_value(
            f"mcp:token_pair:{refresh_token.token}"
        )
        if old_access_token:
            await self._coordination.delete_value(
                f"mcp:access_token:{old_access_token}"
            )
            await self._coordination.delete_value(
                f"mcp:token_pair:{old_access_token}"
            )

        new_access_token = SupernoteAccessToken(
            token=secrets.token_urlsafe(32),
            user_id=refresh_token.user_id,
            client_id=refresh_token.client_id,
            scopes=scopes or refresh_token.scopes,
            expires_at=int(time.time() + _ACCESS_TOKEN_TTL),
        )
        new_refresh_token = SupernoteRefreshToken(
            token=secrets.token_urlsafe(32),
            user_id=refresh_token.user_id,
            client_id=refresh_token.client_id,
            scopes=new_access_token.scopes,
            expires_at=int(time.time() + _REFRESH_TOKEN_TTL),
        )
        await self._coordination.set_value(
            f"mcp:access_token:{new_access_token.token}",
            new_access_token.model_dump_json(),
            ttl=_ACCESS_TOKEN_TTL,
        )
        await self._coordination.set_value(
            f"mcp:refresh_token:{new_refresh_token.token}",
            new_refresh_token.model_dump_json(),
            ttl=_REFRESH_TOKEN_TTL,
        )
        await self._store_token_pair(
            new_access_token.token, new_refresh_token.token
        )
        return OAuthToken(
            access_token=new_access_token.token,
            refresh_token=new_refresh_token.token,
            token_type="Bearer",
            expires_in=_ACCESS_TOKEN_TTL,
            scope=" ".join(new_access_token.scopes),
        )

    async def load_access_token(self, token: str) -> SupernoteAccessToken | None:
        """Loads an access token by its token."""
        # 1. Try to load as an MCP access token from coordination service
        key = f"mcp:access_token:{token}"
        data = await self._coordination.get_value(key)
        if not data:
            return None
        return SupernoteAccessToken.model_validate_json(data)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        """Revokes an access or refresh token."""
        token_key = (
            f"mcp:refresh_token:{token.token}"
            if isinstance(token, RefreshToken)
            else f"mcp:access_token:{token.token}"
        )
        await self._coordination.delete_value(token_key)

        paired_token = await self._coordination.pop_value(
            f"mcp:token_pair:{token.token}"
        )
        if paired_token:
            await self._coordination.delete_value(
                f"mcp:access_token:{paired_token}"
            )
            await self._coordination.delete_value(
                f"mcp:refresh_token:{paired_token}"
            )
            await self._coordination.delete_value(
                f"mcp:token_pair:{paired_token}"
            )

    async def _store_token_pair(
        self, access_token: str, refresh_token: str
    ) -> None:
        await self._coordination.set_value(
            f"mcp:token_pair:{access_token}",
            refresh_token,
            ttl=_REFRESH_TOKEN_TTL,
        )
        await self._coordination.set_value(
            f"mcp:token_pair:{refresh_token}",
            access_token,
            ttl=_REFRESH_TOKEN_TTL,
        )


def create_auth_app(
    user_service: UserService,
    coordination_service: CoordinationService,
    issuer_url: str,
) -> Starlette:
    """Create a Starlette app for the MCP Authorization Server."""
    provider = SupernoteOAuthProvider(user_service, coordination_service, issuer_url)
    routes = create_auth_routes(
        provider=provider,
        issuer_url=AnyHttpUrl(issuer_url),
        revocation_options=RevocationOptions(enabled=True),
    )
    app = Starlette(routes=routes, debug=False)

    # Add login-bridge route
    async def login_bridge(request: Request) -> RedirectResponse | JSONResponse:
        """Handling the OAuth login flow bridging the SPA and the MCP server.

        1. Browser visits /authorize -> Redirects here (/login-bridge).
        2. If User is NOT logged in:
           - GET request: Redirects to SPA login page (/#login) with return_to set to this URL.
           - SPA handles login, then sees return_to pointing to /login-bridge.
           - SPA makes background POST request to this URL with x-access-token header.
        3. If User IS logged in (or via POST with token):
           - Validates session.
           - Requires the SPA to collect an explicit approve/deny decision.
           - Generates OAuth Authorization Code.
           - Returns JSON with 'redirect_url' containing the code (callback URL).
           - SPA redirects the browser to that callback URL.
        """
        # Extract and verify token
        token = get_token_from_request(request)
        session = await user_service.verify_token(token) if token else None
        if not session:
            # If this was an API background call (POST), return 401 JSON
            if request.method == "POST":
                return JSONResponse({"error": "unauthorized"}, status_code=401)

            # Not logged in: Redirect to web UI login page.
            # We pass the bridge URL as return_to so the SPA knows where to return.
            # We keep the query params (OAuth params) so they are preserved.
            login_url = f"/#login?return_to={quote(str(request.url))}"
            return RedirectResponse(url=login_url)

        # Extract OAuth params from query
        client_id = request.query_params.get("client_id")
        redirect_uri = request.query_params.get("redirect_uri")
        state = request.query_params.get("state")

        code_challenge = request.query_params.get("code_challenge")
        code_challenge_method = request.query_params.get("code_challenge_method")

        if not client_id or not redirect_uri:
            return JSONResponse({"error": "invalid_request"}, status_code=400)

        if (
            code_challenge_method != "S256"
            or code_challenge is None
            or not _PKCE_S256_PATTERN.fullmatch(code_challenge)
        ):
            return JSONResponse(
                {
                    "error": "invalid_request",
                    "error_description": "PKCE with a valid S256 challenge is required",
                },
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )

        # Validate Client compatibility
        client_info = await provider.get_client(client_id)
        if not client_info:
            return JSONResponse({"error": "invalid_client"}, status_code=400)

        try:
            validated_redirect_uri = client_info.validate_redirect_uri(
                AnyHttpUrl(redirect_uri)
            )
        except (InvalidRedirectUriError, ValidationError):
            return JSONResponse(
                {"error": "invalid_request", "error_description": "Invalid redirect URI"},
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )

        consent = None
        if request.method == "POST":
            form = await request.form()
            form_consent = form.get("consent")
            if isinstance(form_consent, str):
                consent = form_consent

        try:
            requested_scopes = client_info.validate_scope(
                request.query_params.get("scopes") or "supernote:all"
            )
        except InvalidScopeError:
            return JSONResponse(
                {"error": "invalid_scope"},
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )
        if requested_scopes is None:  # pragma: no cover - default supplied above
            requested_scopes = ["supernote:all"]
        if consent == "deny":
            final_url = construct_redirect_uri(
                str(validated_redirect_uri),
                error="access_denied",
                state=state,
            )
            return JSONResponse(
                {"redirect_url": final_url}, headers={"Cache-Control": "no-store"}
            )
        if consent != "approve":
            return JSONResponse(
                {
                    "consent_required": True,
                    "client_id": client_id,
                    "redirect_uri": str(validated_redirect_uri),
                    "scopes": requested_scopes,
                },
                headers={"Cache-Control": "no-store"},
            )

        # Create the authorization code
        code_str = secrets.token_urlsafe(16)
        auth_code = SupernoteAuthorizationCode(
            code=code_str,
            user_id=session.email,
            client_id=client_id,
            redirect_uri=validated_redirect_uri,
            scopes=requested_scopes,
            code_challenge=code_challenge,
            expires_at=int(time.time() + 600),
            redirect_uri_provided_explicitly=True,
        )

        # Store auth code
        await coordination_service.set_value(
            f"mcp:auth_code:{code_str}",
            auth_code.model_dump_json(),
            ttl=600,
        )

        # Return result
        callback_params = {"code": code_str}
        if state:
            callback_params["state"] = state

        final_url = construct_redirect_uri(
            str(validated_redirect_uri), **callback_params
        )

        return JSONResponse(
            {"redirect_url": final_url}, headers={"Cache-Control": "no-store"}
        )

    app.add_route("/login-bridge", login_bridge, methods=["GET", "POST"])
    return app
