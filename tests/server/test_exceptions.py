"""Unit tests for supernote.server.exceptions module."""

from supernote.models.base import ErrorCode
from supernote.server.exceptions import FileNotFound, ParentNotFound


def test_parent_not_found() -> None:
    """Verify ParentNotFound default message, status code, error code, and inheritance."""
    err = ParentNotFound()
    assert isinstance(err, FileNotFound)
    assert err.message == "Parent directory is missing"
    assert err.status_code == 404
    assert err.error_code == ErrorCode.PATH_NOT_FOUND.value

    custom_err = ParentNotFound("Custom parent missing message")
    assert custom_err.message == "Custom parent missing message"
