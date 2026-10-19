"""Tests for system task persistence utilities."""

from supernote.models.base import ProcessingStatus
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.utils.tasks import get_task, update_task_status


async def test_get_and_update_task_status(
    session_manager: DatabaseSessionManager,
) -> None:
    """Verify update_task_status inserts and updates status, data, and last_error."""
    file_id = 500
    task_type = "EMBEDDING_GENERATION"
    key = "page_p0"

    assert await get_task(session_manager, file_id, task_type, key) is None

    # Insert initial COMPLETED task with model version in data
    await update_task_status(
        session_manager,
        file_id=file_id,
        task_type=task_type,
        key=key,
        status=ProcessingStatus.COMPLETED,
        data="model-v1",
    )

    task = await get_task(session_manager, file_id, task_type, key)
    assert task is not None
    assert task.file_id == file_id
    assert task.task_type == task_type
    assert task.key == key
    assert task.status == ProcessingStatus.COMPLETED
    assert task.data == "model-v1"
    assert task.last_error is None

    # Upsert on conflict to update data to a new model version
    await update_task_status(
        session_manager,
        file_id=file_id,
        task_type=task_type,
        key=key,
        status=ProcessingStatus.COMPLETED,
        data="model-v2",
    )

    updated = await get_task(session_manager, file_id, task_type, key)
    assert updated is not None
    assert updated.id == task.id
    assert updated.status == ProcessingStatus.COMPLETED
    assert updated.data == "model-v2"
    assert updated.last_error is None

    # Upsert on failure sets last_error and clears data
    await update_task_status(
        session_manager,
        file_id=file_id,
        task_type=task_type,
        key=key,
        status=ProcessingStatus.FAILED,
        error="boom",
    )

    failed = await get_task(session_manager, file_id, task_type, key)
    assert failed is not None
    assert failed.id == task.id
    assert failed.status == ProcessingStatus.FAILED
    assert failed.data is None
    assert failed.last_error == "boom"
