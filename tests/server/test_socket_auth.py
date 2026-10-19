from supernote.models.socket import SocketHandshakeParams
from supernote.server.socket_auth import (
    compute_handshake_signature,
    verify_handshake_signature,
)


def test_compute_and_verify_handshake_signature_valid() -> None:
    token = "test-token"
    conn_type = "file"
    random_val = "rnd123"
    sign = compute_handshake_signature(token, conn_type, random_val)

    params = SocketHandshakeParams(
        token=token,
        type=conn_type,
        random=random_val,
        sign=sign,
    )

    assert verify_handshake_signature(params) is True


def test_verify_handshake_signature_invalid() -> None:
    params = SocketHandshakeParams(
        token="test-token",
        type="file",
        random="rnd123",
        sign="wrong_signature",
    )

    assert verify_handshake_signature(params) is False


def test_verify_handshake_signature_missing_sign() -> None:
    params = SocketHandshakeParams(
        token="test-token",
        type="file",
        random="rnd123",
        sign="",
    )
    assert verify_handshake_signature(params) is False
