import hashlib

import pytest
from aiohttp.test_utils import TestClient

from supernote.client import Client
from supernote.client.auth import ConstantAuth
from supernote.client.device import DeviceClient
from supernote.server.services.blob import BlobStorage
from supernote.server.services.user import UserService

USER_A = "user_a@example.com"
USER_B = "user_b@example.com"
DEFAULT_FOLDERS = ["EXPORT", "INBOX", "SCREENSHOT", "NOTE", "DOCUMENT"]


@pytest.fixture
def test_users() -> list[str]:
    return [USER_A, USER_B]


async def test_multi_user_content_with_same_hash_isolation(
    client: TestClient,
    user_service: UserService,
    blob_storage: BlobStorage,
    create_test_user: None,
) -> None:
    """Test that users uploading same content have INDEPENDENT blobs (KV separation)."""
    # Setup Users
    token_a = await user_service.create_session(USER_A)
    token_b = await user_service.create_session(USER_B)

    base_url = str(client.make_url(""))

    client_a = Client(client.session, auth=ConstantAuth(token_a), host=base_url)
    file_client_a = DeviceClient(client_a)

    client_b = Client(client.session, auth=ConstantAuth(token_b), host=base_url)
    file_client_b = DeviceClient(client_b)

    # Common Content
    common_content = b"Shared Content Block"

    # User A uploads
    await file_client_a.upload_content(
        path="/doc_a.txt",
        content=common_content,
        equipment_no="EQ001",
    )

    # User B uploads SAME content
    await file_client_b.upload_content(
        path="/doc_b.txt",
        content=common_content,
        equipment_no="EQ002",
    )

    # User A Deletes their file
    info_a = await file_client_a.query_by_path(path="/doc_a.txt", equipment_no="EQ001")
    assert info_a.entries_vo
    file_id_a = info_a.entries_vo.id

    await file_client_a.delete(id=int(file_id_a), equipment_no="EQ001")

    # User B should still be able to download/read
    downloaded_b = await file_client_b.download_content(
        path="/doc_b.txt", equipment_no="EQ002"
    )
    assert downloaded_b == common_content, (
        "User B content must persist after User A deletion"
    )


async def test_multi_user_content_with_same_paths(
    client: TestClient,
    create_test_user: None,
    user_service: UserService,
) -> None:
    """Test that users can upload distrinct content with the same path."""
    # Setup Users
    token_a = await user_service.create_session(USER_A)
    token_b = await user_service.create_session(USER_B)

    base_url = str(client.make_url(""))

    client_a = Client(client.session, auth=ConstantAuth(token_a), host=base_url)
    file_client_a = DeviceClient(client_a)

    client_b = Client(client.session, auth=ConstantAuth(token_b), host=base_url)
    file_client_b = DeviceClient(client_b)

    # User A uploads a file
    filename = "shared.note"
    content_a = b"User A content"
    hash_a = hashlib.md5(content_a).hexdigest()

    await file_client_a.upload_content(
        path=f"/{filename}",
        content=content_a,
        equipment_no="EQ001",
    )

    # User A list files should see their file
    entries_a = await file_client_a.list_folder(path="/", equipment_no="EQ001")
    assert [e.name for e in entries_a.entries] == [*DEFAULT_FOLDERS, "shared.note"]

    # User B list files should NOT see User A's file
    entries_b = await file_client_b.list_folder(path="/", equipment_no="EQ002")
    assert [e.name for e in entries_b.entries] == DEFAULT_FOLDERS

    # User B uploads a file with the same name
    content_b = b"Content from User B"
    hash_b = hashlib.md5(content_b).hexdigest()

    await file_client_b.upload_content(
        path=f"/{filename}",
        content=content_b,
        equipment_no="EQ002",
    )

    # 4. User A queries their file - SHOULD STILL HAVE THEIR CONTENT
    info_a = await file_client_a.query_by_path(
        path=f"/{filename}", equipment_no="EQ001"
    )
    assert info_a.entries_vo
    assert info_a.entries_vo.content_hash == hash_a, (
        "User A's file should NOT be clobbered by User B"
    )

    # 5. User B queries their file - SHOULD HAVE THEIR CONTENT
    info_b = await file_client_b.query_by_path(
        path=f"/{filename}", equipment_no="EQ002"
    )
    assert info_b.entries_vo
    assert info_b.entries_vo.content_hash == hash_b, (
        "User B should have their own file content"
    )
