"""Socket.IO server manager and event handling for Supernote server."""

import logging
from http import HTTPStatus
from urllib.parse import parse_qs

import socketio
from aiohttp import web

from supernote.models.socket import (
    SocketHandshakeParams,
    SocketIoClientMessage,
    SocketIoEvent,
    SocketMessageData,
)
from supernote.server.config import ServerConfig
from supernote.server.services.user import SessionState, UserService
from supernote.server.socket_auth import (
    verify_handshake_signature,
)

logger = logging.getLogger(__name__)


def _extract_handshake_params(environ: dict) -> SocketHandshakeParams:
    """Extract handshake authentication parameters from the WSGI/ASGI connection environment.

    Socket.IO passes the request environment dictionary to the connection handler,
    containing the raw QUERY_STRING sent during transport establishment.
    """
    query_string = environ.get("QUERY_STRING", "")
    parsed_query = parse_qs(query_string)

    token = parsed_query.get("token", [""])[0]
    conn_type = parsed_query.get("type", ["file"])[0]
    random_val = parsed_query.get("random", [""])[0]
    sign = parsed_query.get("sign", [""])[0]

    return SocketHandshakeParams(
        token=token,
        type=conn_type,
        random=random_val,
        sign=sign,
    )


class SocketIOServerManager:
    """Manages the Socket.IO server lifecycle, authentication, session rooms, and message delivery."""

    def __init__(self, config: ServerConfig, user_service: UserService) -> None:
        self.config = config
        self._user_service = user_service
        self._sio = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            allow_eio3=True,
            logger=False,
            engineio_logger=False,
        )
        self._sid_to_user: dict[str, str] = {}
        self._sid_to_user_id: dict[str, int] = {}
        self._user_id_to_sids: dict[int, set[str]] = {}
        self._sid_to_token: dict[str, str] = {}
        self._setup_handlers()

    async def _index_session(self, sid: str, session: SessionState) -> None:
        """Index a socket by stable identity and refresh its email room."""
        previous_user_id = self._sid_to_user_id.get(sid)
        if previous_user_id != session.user_id:
            if previous_user_id is not None:
                previous_sids = self._user_id_to_sids.get(previous_user_id)
                if previous_sids is not None:
                    previous_sids.discard(sid)
                    if not previous_sids:
                        self._user_id_to_sids.pop(previous_user_id, None)
            self._sid_to_user_id[sid] = session.user_id
            self._user_id_to_sids.setdefault(session.user_id, set()).add(sid)

        connected_user = self._sid_to_user.get(sid)
        if connected_user != session.email:
            if connected_user:
                await self._sio.leave_room(sid, f"user_{connected_user}")
            self._sid_to_user[sid] = session.email
            await self._sio.enter_room(sid, f"user_{session.email}")

    async def _is_authorized(self, sid: str) -> bool:
        """Revalidate a connected socket against current server-side state."""
        token = self._sid_to_token.get(sid)
        if not token:
            return False
        session = await self._user_service.verify_token(token)
        if not session:
            return False

        await self._index_session(sid, session)
        return True

    async def _disconnect_if_unauthorized(self, sid: str) -> bool:
        if await self._is_authorized(sid):
            return False
        logger.warning(
            "Disconnecting Socket.IO session with revoked authorization: %s", sid
        )
        await self._sio.disconnect(sid)
        return True

    @property
    def sio(self) -> socketio.AsyncServer:
        """Access the underlying AsyncServer instance."""
        return self._sio

    def attach_to_app(self, app: web.Application) -> None:
        """Attach the Socket.IO server instance to an aiohttp web Application."""
        self._sio.attach(app)
        app["socketio_manager"] = self

    async def _send_error(self, sid: str, status: HTTPStatus, message: str) -> None:
        """Send a SocketMessageData error response to a specific session ID."""
        err_data = SocketMessageData(
            code=str(status.value),
            msg=message,
        )
        await self._sio.emit(
            SocketIoEvent.SERVER_MESSAGE.value,
            err_data.to_json(),
            to=sid,
        )

    def _setup_handlers(self) -> None:
        @self._sio.event
        async def connect(sid: str, environ: dict) -> bool:
            params = _extract_handshake_params(environ)

            # Verify handshake signature
            if not verify_handshake_signature(params):
                logger.error(
                    "Socket.IO connect rejected: invalid signature (sid=%s)", sid
                )
                await self._send_error(
                    sid, HTTPStatus.FORBIDDEN, "sign verification failed"
                )
                return False

            # Use the same stateful session authority as authenticated HTTP routes.
            # A valid JWT alone is insufficient because sessions can be revoked.
            session = await self._user_service.verify_token(params.token)
            if not session:
                logger.error("Socket.IO connect rejected: invalid token (sid=%s)", sid)
                await self._send_error(
                    sid, HTTPStatus.FORBIDDEN, "token verification failed"
                )
                return False
            user_id = session.email

            # Associate session ID with user_id and join user room
            self._sid_to_token[sid] = params.token
            await self._index_session(sid, session)
            user_room = f"user_{user_id}"
            logger.info(
                "Socket.IO connection established for user=%s (sid=%s, room=%s)",
                user_id,
                sid,
                user_room,
            )
            return True

        @self._sio.event
        async def disconnect(sid: str) -> None:
            user_id = self._sid_to_user.pop(sid, None)
            stable_user_id = self._sid_to_user_id.pop(sid, None)
            if stable_user_id is not None:
                user_sids = self._user_id_to_sids.get(stable_user_id)
                if user_sids is not None:
                    user_sids.discard(sid)
                    if not user_sids:
                        self._user_id_to_sids.pop(stable_user_id, None)
            self._sid_to_token.pop(sid, None)
            logger.info("Socket.IO disconnected for user=%s (sid=%s)", user_id, sid)

        @self._sio.on(SocketIoEvent.CLIENT_MESSAGE.value)
        async def on_client_message(sid: str, data: str) -> None:
            if await self._disconnect_if_unauthorized(sid):
                return
            user_id = self._sid_to_user.get(sid)
            logger.debug(
                "Received ClientMessage from user=%s (sid=%s): %s", user_id, sid, data
            )

            if data == SocketIoClientMessage.STATUS.value:
                # Heartbeat check reply
                await self._sio.emit(
                    SocketIoEvent.SERVER_MESSAGE.value,
                    "true",
                    to=sid,
                )
            elif data == SocketIoClientMessage.RECEIVED.value:
                # Client delivery acknowledgment
                logger.debug("Client ACK received for user=%s (sid=%s)", user_id, sid)

        @self._sio.on(SocketIoEvent.RATTA_PING.value)
        async def on_ratta_ping(sid: str, data: str) -> None:
            if await self._disconnect_if_unauthorized(sid):
                return
            user_id = self._sid_to_user.get(sid)
            logger.debug(
                "Received ratta_ping from user=%s (sid=%s): %s", user_id, sid, data
            )
            await self._sio.emit(
                SocketIoEvent.RATTA_PING.value,
                SocketIoClientMessage.RECEIVED.value,
                to=sid,
            )

    async def send_message(self, user_id: str, message: SocketMessageData) -> None:
        """Send a SocketMessageData event to all connected sessions for a given user.

        Args:
            user_id: Target user email or identifier.
            message: SocketMessageData payload to deliver.
        """
        payload_str = message.to_json()
        logger.debug("Emitting ServerMessage to user=%s: %s", user_id, payload_str)
        target_user = await self._user_service.get_user_by_email(user_id)
        if target_user is None:
            return
        stable_user_id = target_user.id

        for sid in list(self._user_id_to_sids.get(stable_user_id, ())):
            if await self._disconnect_if_unauthorized(sid):
                continue
            if self._sid_to_user_id.get(sid) != stable_user_id:
                continue
            await self._sio.emit(
                SocketIoEvent.SERVER_MESSAGE.value,
                payload_str,
                to=sid,
            )


def setup_socketio(
    app: web.Application, config: ServerConfig, user_service: UserService
) -> SocketIOServerManager:
    """Initialize and attach Socket.IO server manager to an aiohttp Application.

    Args:
        app: The aiohttp web Application.
        config: The server configuration instance.
        user_service: Shared stateful authentication service.

    Returns:
        The configured SocketIOServerManager instance.
    """
    manager = SocketIOServerManager(config, user_service)
    manager.attach_to_app(app)
    return manager
