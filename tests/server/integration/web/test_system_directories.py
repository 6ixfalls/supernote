import pytest

from supernote.client.device import DeviceClient
from supernote.client.exceptions import ApiException
from supernote.client.web import WebClient


async def test_web_cannot_delete_system_directories(web_client: WebClient) -> None:
    """Verify that system directories cannot be deleted via Web API."""
    # List root to get IDs
    res = await web_client.list_query(directory_id=0)
    assert res.success

    # Find Note folder
    note_folder = next(f for f in res.user_file_vo_list if f.file_name == "Note")
    note_id = int(note_folder.id)

    # Attempt to delete
    with pytest.raises(ApiException, match="Cannot delete system directory"):
        await web_client.file_delete(id_list=[note_id], parent_id=0)

    # Verify still exists
    res_after = await web_client.list_query(directory_id=0)
    assert any(f.file_name == "Note" for f in res_after.user_file_vo_list)


async def test_device_cannot_delete_system_directories(
    device_client: DeviceClient,
) -> None:
    """Verify that system directories cannot be deleted via Device API."""
    # Device sees NOTE container
    res = await device_client.list_folder("/", recursive=False)
    note_container = next(e for e in res.entries if e.name == "NOTE")
    note_container_id = int(note_container.id)

    # Attempt to delete by ID (Device API uses delete_folder_v3 via .delete())
    with pytest.raises(ApiException, match="Cannot delete system directory"):
        await device_client.delete(id=note_container_id, equipment_no="TEST")

    # Verify still exists
    res_after = await device_client.list_folder("/", recursive=False)
    assert any(e.name == "NOTE" for e in res_after.entries)


async def test_device_cannot_delete_flattened_folders(
    device_client: DeviceClient,
) -> None:
    """Verify that device cannot delete Note even if it knows the ID."""
    # 1. Find NOTE's ID
    res = await device_client.list_folder("/", recursive=False)
    note_container = next(e for e in res.entries if e.name == "NOTE")

    # 2. Find Note folder INSIDE NOTE
    children_res = await device_client.list_folder(folder_id=int(note_container.id))
    note_folder = next(e for e in children_res.entries if e.name == "Note")
    note_id = int(note_folder.id)

    # 3. Attempt to delete
    with pytest.raises(ApiException, match="Cannot delete system directory"):
        await device_client.delete(id=note_id, equipment_no="TEST")


async def test_a_users_own_folder_may_share_a_system_name(
    web_client: WebClient,
) -> None:
    """A folder named `Document` outside its system location can be deleted."""
    res = await web_client.list_query(directory_id=0)
    note_folder = next(f for f in res.user_file_vo_list if f.file_name == "Note")

    created = await web_client.create_folder(
        parent_id=int(note_folder.id), name="Document"
    )

    await web_client.file_delete(
        id_list=[int(created.id)], parent_id=int(note_folder.id)
    )

    children = await web_client.list_query(directory_id=int(note_folder.id))
    assert not any(f.file_name == "Document" for f in children.user_file_vo_list)


async def test_new_user_root_folders_use_firmware_names(
    device_client: DeviceClient,
) -> None:
    """A new account's root folders are spelled as the device firmware spells them."""
    res = await device_client.list_folder("/", recursive=False)
    assert sorted(e.name for e in res.entries) == [
        "DOCUMENT",
        "EXPORT",
        "INBOX",
        "NOTE",
        "SCREENSHOT",
    ]


async def test_web_root_listing_capitalizes_firmware_names(
    web_client: WebClient,
) -> None:
    """The web API shows the stored EXPORT as Export, like the reference server."""
    res = await web_client.list_query(directory_id=0)
    names = {f.file_name for f in res.user_file_vo_list}
    assert {"Export", "Inbox", "Screenshot"} <= names
    assert not {"EXPORT", "INBOX", "SCREENSHOT"} & names


async def test_device_creating_a_system_folder_adds_no_second_root(
    device_client: DeviceClient,
) -> None:
    """The device creating /EXPORT on a fresh account finds the existing folder."""
    await device_client.create_folder(path="/EXPORT", equipment_no="SN123456")

    res = await device_client.list_folder("/", recursive=False)
    assert [e.name for e in res.entries if e.name.casefold() == "export"] == ["EXPORT"]


async def test_device_paths_are_stable_across_upload_query_and_list(
    device_client: DeviceClient,
) -> None:
    """A file uploaded to /INBOX is reported at the same path by every device API."""
    await device_client.upload_content(
        path="/INBOX/a.pdf", content=b"%PDF-1.4", equipment_no="SN123456"
    )

    queried = await device_client.query_by_path(
        path="/INBOX/a.pdf", equipment_no="SN123456"
    )
    assert queried.entries_vo is not None

    listed = await device_client.list_folder("/INBOX", equipment_no="SN123456")
    assert [e.path_display for e in listed.entries] == [queried.entries_vo.path_display]
    assert queried.entries_vo.path_display == "INBOX/a.pdf"


async def test_web_path_query_capitalizes_firmware_names(
    web_client: WebClient,
) -> None:
    """/api/file/path/query capitalizes firmware root folder names like the root listing."""
    res = await web_client.list_query(directory_id=0)
    export = next(f for f in res.user_file_vo_list if f.file_name == "Export")
    sub = await web_client.create_folder(parent_id=int(export.id), name="Sub")

    path_vo = await web_client.path_query(id=int(sub.id))
    assert path_vo.path == "Export/Sub"
