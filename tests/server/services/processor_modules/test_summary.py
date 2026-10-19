import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from supernote.models.base import BooleanEnum
from supernote.models.summary import (
    METADATA_SEGMENTS,
    AddSummaryDTO,
    AddSummaryGroupDTO,
    SummaryItem,
    UpdateSummaryGroupDTO,
)
from supernote.server.ai.models.fake.summary import FakeSummaryModel
from supernote.server.ai.prompts import SummarySegment
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import NotePageContentDO, SystemTaskDO
from supernote.server.db.models.user import UserDO
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.file import FileService
from supernote.server.services.processor_modules.summary import SummaryModule
from supernote.server.services.summary import SummaryService
from supernote.server.utils.paths import (
    get_summary_group_id,
    get_summary_id,
    get_transcript_id,
)


@pytest.fixture
def mock_summary_service() -> MagicMock:
    service = MagicMock(spec=SummaryService)
    service.get_summary_by_uuid = AsyncMock(return_value=None)
    service.add_summary = AsyncMock()
    service.update_summary = AsyncMock()
    service.add_group = AsyncMock()
    service.update_group = AsyncMock()
    return service


@pytest.fixture
def summary_module(
    file_service: FileService,
    fake_summary_model: FakeSummaryModel,
    mock_summary_service: MagicMock,
) -> SummaryModule:
    return SummaryModule(
        file_service=file_service,
        summary_service=mock_summary_service,
        summary_model=fake_summary_model,
    )


async def test_summary_success(
    summary_module: SummaryModule,
    session_manager: DatabaseSessionManager,
    fake_summary_model: FakeSummaryModel,
    mock_summary_service: MagicMock,
) -> None:
    user_id = 100
    user_email = "test@example.com"
    file_id = 999
    storage_key = "test_storage_key"

    async with session_manager.session() as session:
        user = UserDO(id=user_id, email=user_email, password_md5="hash")
        session.add(user)

        user_file = UserFileDO(
            id=file_id,
            user_id=user_id,
            storage_key=storage_key,
            file_name="real.note",
            directory_id=0,
        )
        session.add(user_file)

        p1 = NotePageContentDO(
            file_id=file_id,
            page_index=0,
            page_id="p0",
            content_hash="h1",
            text_content="Page 1 text",
        )
        p2 = NotePageContentDO(
            file_id=file_id,
            page_index=1,
            page_id="p1",
            content_hash="h2",
            text_content="Page 2 text",
        )
        session.add(p1)
        session.add(p2)
        await session.commit()

    await summary_module.run(file_id, session_manager)

    assert len(fake_summary_model.calls) == 1
    prompt_arg = fake_summary_model.calls[0]
    assert "TRANSCRIPT:" in prompt_arg
    assert "Page 1 text" in prompt_arg
    assert "Page 2 text" in prompt_arg

    group_call = mock_summary_service.add_group.call_args_list[0]
    assert group_call.args[0] == user_email
    dto_group = group_call.args[1]
    assert isinstance(dto_group, AddSummaryGroupDTO)
    assert dto_group.unique_identifier == get_summary_group_id(storage_key)
    assert dto_group.name == "real.note"

    transcript_call = mock_summary_service.add_summary.call_args_list[0]
    assert transcript_call.args[0] == user_email
    dto = transcript_call.args[1]
    assert isinstance(dto, AddSummaryDTO)
    assert dto.unique_identifier == get_transcript_id(storage_key)
    assert dto.parent_unique_identifier == get_summary_group_id(storage_key)
    assert dto.content is not None
    assert "Page 1 text" in dto.content
    assert "Page 2 text" in dto.content
    assert dto.data_source == "OCR"

    ai_call = mock_summary_service.add_summary.call_args_list[1]
    assert ai_call.args[0] == user_email
    dto_ai = ai_call.args[1]
    assert dto_ai.unique_identifier == get_summary_id(storage_key)
    assert dto_ai.parent_unique_identifier == get_summary_group_id(storage_key)
    assert "## 2023-10-27 (Pages 1)" in dto_ai.content
    assert "Summary of (" in dto_ai.content
    assert dto_ai.data_source == "fake-summary-v1"

    # Check Metadata
    assert dto_ai.metadata is not None
    meta = json.loads(dto_ai.metadata)
    assert meta[METADATA_SEGMENTS][0]["page_refs"] == [1]

    async with session_manager.session() as session:
        task = (
            (
                await session.execute(
                    select(SystemTaskDO)
                    .where(SystemTaskDO.file_id == file_id)
                    .where(SystemTaskDO.task_type == "SUMMARY_GENERATION")
                    .where(SystemTaskDO.key == "global")
                )
            )
            .scalars()
            .first()
        )
        assert task is not None
        assert task.status == "COMPLETED"


async def test_summary_idempotency_update(
    file_service: FileService,
    session_manager: DatabaseSessionManager,
    mock_summary_service: MagicMock,
) -> None:
    user_id = 101
    user_email = "update@example.com"
    file_id = 888
    storage_key = "update_key"

    async with session_manager.session() as session:
        user = UserDO(id=user_id, email=user_email, password_md5="hash")
        session.add(user)
        user_file = UserFileDO(
            id=file_id,
            user_id=user_id,
            storage_key=storage_key,
            file_name="update.note",
            directory_id=0,
        )
        session.add(user_file)
        session.add(
            NotePageContentDO(
                file_id=file_id,
                page_index=0,
                page_id="p0",
                content_hash="h1",
                text_content="Some text",
            )
        )
        await session.commit()

    custom_summary_model = FakeSummaryModel(
        segments=[
            SummarySegment(
                date_range="2023-10-28",
                summary="New AI Summary",
                extracted_dates=[],
                page_refs=[3, 4],
            )
        ]
    )
    summary_module = SummaryModule(
        file_service=file_service,
        summary_service=mock_summary_service,
        summary_model=custom_summary_model,
    )

    existing_group = SummaryItem(
        id=10,
        unique_identifier=get_summary_group_id(storage_key),
        name="update.note",
        is_summary_group=BooleanEnum.YES,
    )
    existing_summary = SummaryItem(id=11, unique_identifier=get_summary_id(storage_key))
    mock_summary_service.get_summary_by_uuid.side_effect = [
        existing_group,
        None,
        existing_summary,
    ]

    await summary_module.run(file_id, session_manager)

    mock_summary_service.update_group.assert_called_once()
    update_group_dto = mock_summary_service.update_group.call_args.args[1]
    assert isinstance(update_group_dto, UpdateSummaryGroupDTO)
    assert update_group_dto.id == 10
    assert update_group_dto.unique_identifier == get_summary_group_id(storage_key)
    assert update_group_dto.name == "update.note"

    mock_summary_service.add_summary.assert_called_once()
    transcript_dto = mock_summary_service.add_summary.call_args.args[1]
    assert transcript_dto.unique_identifier == get_transcript_id(storage_key)
    assert transcript_dto.parent_unique_identifier == get_summary_group_id(storage_key)

    mock_summary_service.update_summary.assert_called_once()
    update_call = mock_summary_service.update_summary.call_args
    assert update_call is not None
    update_dto = update_call.args[1]

    assert update_dto.id == existing_summary.id
    assert update_dto.parent_unique_identifier == get_summary_group_id(storage_key)
    assert "## 2023-10-28 (Pages 3, 4)" in update_dto.content
    assert "New AI Summary" in update_dto.content

    assert update_dto.metadata is not None
    meta = json.loads(update_dto.metadata)
    assert meta[METADATA_SEGMENTS][0]["page_refs"] == [3, 4]


async def test_summary_transcript_with_dates(
    file_service: FileService,
    session_manager: DatabaseSessionManager,
    mock_summary_service: MagicMock,
) -> None:
    user_id = 100
    user_email = "test@example.com"
    file_id = 123
    storage_key = "date_test_key"

    async with session_manager.session() as session:
        user = UserDO(id=user_id, email=user_email, password_md5="hash")
        session.add(user)
        user_file = UserFileDO(
            id=file_id,
            user_id=user_id,
            storage_key=storage_key,
            file_name="dates.note",
            directory_id=0,
        )
        session.add(user_file)

        session.add(
            NotePageContentDO(
                file_id=file_id,
                page_index=0,
                page_id="P20231027120000abc",
                text_content="Content on Oct 27",
            )
        )
        await session.commit()

    empty_summary_model = FakeSummaryModel(segments=[])
    summary_module = SummaryModule(
        file_service=file_service,
        summary_service=mock_summary_service,
        summary_model=empty_summary_model,
    )

    await summary_module.run(file_id, session_manager)

    mock_summary_service.add_group.assert_called_once()
    group_dto = mock_summary_service.add_group.call_args.args[1]
    assert group_dto.unique_identifier == get_summary_group_id(storage_key)
    assert group_dto.name == "dates.note"

    transcript_call = mock_summary_service.add_summary.call_args_list[0]
    dto = transcript_call.args[1]
    assert dto.parent_unique_identifier == get_summary_group_id(storage_key)
    assert "--- Page 1 ---" in dto.content
    assert "Page ID: P20231027120000abc" in dto.content
    assert "Page Date (Inferred): 2023-10-27" in dto.content
