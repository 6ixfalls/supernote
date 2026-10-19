import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from supernote.server.constants import USER_DATA_BUCKET
from supernote.server.db.models.file import RecycleFileDO, UserFileDO
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.exceptions import ParentNotFound
from supernote.server.services.blob import BlobStorage
from supernote.server.services.integrity import IntegrityService
from supernote.server.services.vfs import VirtualFileSystem


async def test_vfs_directory_operations(db_session: AsyncSession) -> None:
    vfs = VirtualFileSystem(db_session)
    user_id = 999
    root_id = 0

    # Create Directory
    folder = await vfs.create_directory(user_id, root_id, "MyFolder")
    assert folder.id > 0
    assert folder.file_name == "MyFolder"
    assert folder.directory_id == root_id

    # Helper to check listing
    children = await vfs.list_directory(user_id, root_id)
    assert len(children) == 1
    assert children[0].id == folder.id

    # Create sub-directory
    subfolder = await vfs.create_directory(user_id, folder.id, "SubFolder")
    assert subfolder.directory_id == folder.id

    # List sub-directory
    sub_children = await vfs.list_directory(user_id, folder.id)
    assert len(sub_children) == 1
    assert sub_children[0].file_name == "SubFolder"


async def test_vfs_file_operations(db_session: AsyncSession) -> None:
    vfs = VirtualFileSystem(db_session)
    user_id = 888

    # Create File
    file_node = await vfs.create_or_update_file(
        user_id, 0, "test.txt", size=100, md5="hash", storage_key="test-key"
    )
    assert file_node.file_name == "test.txt"
    assert file_node.is_folder == "N"

    # Verify in list
    children = await vfs.list_directory(user_id, 0)
    assert len(children) == 1
    assert children[0].md5 == "hash"

    # Soft Delete
    deleted = await vfs.delete_node(user_id, file_node.id)
    assert deleted is True

    # Verify gone from list
    children = await vfs.list_directory(user_id, 0)
    assert len(children) == 0

    # Verify can't get
    node = await vfs.get_node_by_id(user_id, file_node.id)
    assert node is None


async def test_vfs_ensure_directory_path_category_resolution(
    db_session: AsyncSession,
) -> None:
    """Verify ensure_directory_path resolves 'Note' to NOTE/Note system folder."""
    vfs = VirtualFileSystem(db_session)
    user_id = 999
    note_dir = await vfs.create_directory(user_id, 0, "NOTE")
    note_subdir = await vfs.create_directory(user_id, note_dir.id, "Note")

    resolved_id = await vfs.ensure_directory_path(user_id, "Note")
    assert resolved_id == note_subdir.id


async def test_vfs_resolve_path_category_resolution(
    db_session: AsyncSession,
) -> None:
    """Verify resolve_path resolves 'Note' to NOTE/Note system folder."""
    vfs = VirtualFileSystem(db_session)
    user_id = 999
    note_dir = await vfs.create_directory(user_id, 0, "NOTE")
    note_subdir = await vfs.create_directory(user_id, note_dir.id, "Note")

    node = await vfs.resolve_path(user_id, "Note")
    assert node is not None
    assert node.id == note_subdir.id


async def test_vfs_resolve_mismatched_container_path(
    db_session: AsyncSession,
) -> None:
    """Verify resolve_path('DOCUMENT/Note') returns None and does not match NOTE/Note."""
    vfs = VirtualFileSystem(db_session)
    user_id = 999
    note_dir = await vfs.create_directory(user_id, 0, "NOTE")
    await vfs.create_directory(user_id, note_dir.id, "Note")
    await vfs.create_directory(user_id, 0, "DOCUMENT")

    # DOCUMENT/Note does not exist, so it must return None
    node = await vfs.resolve_path(user_id, "DOCUMENT/Note")
    assert node is None


async def test_soft_delete_folder_cascades_inactivation(
    db_session: AsyncSession,
    session_manager: DatabaseSessionManager,
    blob_storage: BlobStorage,
) -> None:
    """Verify soft-deleting a folder cascades inactivation across multiple folder layers."""
    vfs = VirtualFileSystem(db_session)
    user_id = 1001

    parent = await vfs.create_directory(user_id, 0, "Docs")
    child_dir = await vfs.create_directory(user_id, parent.id, "Work")
    grandchild_dir = await vfs.create_directory(user_id, child_dir.id, "2026")

    await blob_storage.put(USER_DATA_BUCKET, "k1", b"content1")
    await blob_storage.put(USER_DATA_BUCKET, "k2", b"content2")
    await blob_storage.put(USER_DATA_BUCKET, "k3", b"content3")
    await vfs.create_or_update_file(
        user_id, parent.id, "root_doc.txt", size=8, md5="m1", storage_key="k1"
    )
    await vfs.create_or_update_file(
        user_id, child_dir.id, "sub_doc.txt", size=8, md5="m2", storage_key="k2"
    )
    await vfs.create_or_update_file(
        user_id, grandchild_dir.id, "deep_doc.txt", size=8, md5="m3", storage_key="k3"
    )

    assert await vfs.get_total_usage(user_id) == 24
    assert len(await vfs.search_files(user_id, "doc")) == 4
    assert len(await vfs.list_directory(user_id, parent.id)) == 2

    integrity = IntegrityService(session_manager, blob_storage)
    report_before = await integrity.verify_user_storage(user_id)
    assert report_before.orphans == 0
    assert report_before.scanned == 6

    assert await vfs.delete_node(user_id, parent.id) is True

    assert len(await vfs.list_directory(user_id, 0)) == 0
    assert len(await vfs.list_directory(user_id, parent.id)) == 0
    assert len(await vfs.list_directory(user_id, child_dir.id)) == 0
    assert len(await vfs.list_directory(user_id, grandchild_dir.id)) == 0
    assert len(await vfs.search_files(user_id, "doc")) == 0
    assert await vfs.get_total_usage(user_id) == 0

    report_after = await integrity.verify_user_storage(user_id)
    assert report_after.orphans == 0
    assert report_after.scanned == 0


async def test_restore_folder_selective_reactivation(
    db_session: AsyncSession,
) -> None:
    """Verify restoring a multi-layer folder reactivates cascaded descendants but not independently deleted files or subfolders."""
    vfs = VirtualFileSystem(db_session)
    user_id = 1002

    # Layer 1 -> Layer 2 -> Layer 3
    folder = await vfs.create_directory(user_id, 0, "Projects")
    subfolder = await vfs.create_directory(user_id, folder.id, "Sub")
    deep_folder = await vfs.create_directory(user_id, subfolder.id, "Deep")
    file1 = await vfs.create_or_update_file(
        user_id, deep_folder.id, "independent.txt", size=10, md5="m1", storage_key="k1"
    )
    file2 = await vfs.create_or_update_file(
        user_id, deep_folder.id, "cascaded.txt", size=20, md5="m2", storage_key="k2"
    )

    # Second branch: an independently deleted subfolder with its own child file
    archived_sub = await vfs.create_directory(user_id, folder.id, "ArchivedSub")
    archived_file = await vfs.create_or_update_file(
        user_id, archived_sub.id, "archived.txt", size=30, md5="m3", storage_key="k3"
    )

    # Independently delete file1 and archived_sub before deleting top-level folder
    assert await vfs.delete_node(user_id, file1.id) is True
    assert await vfs.delete_node(user_id, archived_sub.id) is True
    assert await vfs.delete_node(user_id, folder.id) is True

    recycle_items = await vfs.list_recycle(user_id)
    assert len(recycle_items) == 3
    folder_recycle = next(r for r in recycle_items if r.file_id == folder.id)
    file1_recycle = next(r for r in recycle_items if r.file_id == file1.id)
    archived_recycle = next(r for r in recycle_items if r.file_id == archived_sub.id)

    # Restore top-level folder
    assert await vfs.restore_node(user_id, folder_recycle.id) is True

    # All 3 active folder layers and cascaded file2 are restored
    assert await vfs.get_node_by_id(user_id, folder.id) is not None
    assert await vfs.get_node_by_id(user_id, subfolder.id) is not None
    assert await vfs.get_node_by_id(user_id, deep_folder.id) is not None
    assert await vfs.get_node_by_id(user_id, file2.id) is not None

    # Independently deleted file1 and archived_sub (plus its child) remain inactive
    assert await vfs.get_node_by_id(user_id, file1.id) is None
    assert await vfs.get_node_by_id(user_id, archived_sub.id) is None
    assert await vfs.get_node_by_id(user_id, archived_file.id) is None
    for inactive_id in (file1.id, archived_sub.id, archived_file.id):
        stmt = select(UserFileDO).where(UserFileDO.id == inactive_id)
        assert (await db_session.execute(stmt)).scalar_one().is_active == "N"

    recycle_after = await vfs.list_recycle(user_id)
    assert {r.id for r in recycle_after} == {file1_recycle.id, archived_recycle.id}

    # Restoring archived_sub and file1 independently now succeeds
    assert await vfs.restore_node(user_id, archived_recycle.id) is True
    assert await vfs.get_node_by_id(user_id, archived_sub.id) is not None
    assert await vfs.get_node_by_id(user_id, archived_file.id) is not None

    assert await vfs.restore_node(user_id, file1_recycle.id) is True
    assert await vfs.get_node_by_id(user_id, file1.id) is not None


async def test_restore_fails_when_parent_missing_or_inactive(
    db_session: AsyncSession,
) -> None:
    """Verify restoring an item whose parent is inactive or missing fails with ParentNotFound."""
    vfs = VirtualFileSystem(db_session)
    user_id = 1003

    folder = await vfs.create_directory(user_id, 0, "ParentDir")
    child = await vfs.create_or_update_file(
        user_id, folder.id, "child.txt", size=10, md5="m", storage_key="k"
    )

    await vfs.delete_node(user_id, child.id)
    await vfs.delete_node(user_id, folder.id)

    recycle_items = await vfs.list_recycle(user_id)
    child_recycle = next(r for r in recycle_items if r.file_id == child.id)

    with pytest.raises(ParentNotFound, match="Parent directory is missing"):
        await vfs.restore_node(user_id, child_recycle.id)

    await db_session.execute(delete(UserFileDO).where(UserFileDO.id == folder.id))
    await db_session.commit()

    with pytest.raises(ParentNotFound, match="Parent directory is missing"):
        await vfs.restore_node(user_id, child_recycle.id)

    assert await vfs.restore_node(user_id, 999999) is False


async def test_purge_folder_removes_all_descendants_and_recycle_entries(
    db_session: AsyncSession,
) -> None:
    """Verify permanently purging a folder removes all descendant file records and recycle bin entries."""
    vfs = VirtualFileSystem(db_session)
    user_id = 1004

    f1 = await vfs.create_directory(user_id, 0, "F1")
    f2 = await vfs.create_directory(user_id, f1.id, "F2")
    await vfs.create_or_update_file(
        user_id, f1.id, "file1.txt", size=5, md5="m1", storage_key="k1"
    )
    file2 = await vfs.create_or_update_file(
        user_id, f2.id, "file2.txt", size=10, md5="m2", storage_key="k2"
    )

    await vfs.delete_node(user_id, file2.id)
    await vfs.delete_node(user_id, f1.id)

    recycle_entries = await vfs.list_recycle(user_id)
    assert len(recycle_entries) == 2
    f1_recycle = next(r for r in recycle_entries if r.file_id == f1.id)

    purged_nodes = await vfs.purge_recycle(user_id, [f1_recycle.id])
    assert len(purged_nodes) == 4

    remaining_files = (
        (
            await db_session.execute(
                select(UserFileDO).where(UserFileDO.user_id == user_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(remaining_files) == 0

    remaining_recycle = (
        (
            await db_session.execute(
                select(RecycleFileDO).where(RecycleFileDO.user_id == user_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(remaining_recycle) == 0
