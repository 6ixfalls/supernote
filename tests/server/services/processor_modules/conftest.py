from pathlib import Path
from unittest.mock import MagicMock

import pytest

from supernote.server.ai.models.fake.embedding import FakeEmbeddingModel
from supernote.server.ai.models.fake.ocr import FakeOcrModel
from supernote.server.ai.models.fake.summary import FakeSummaryModel
from supernote.server.config import AuthConfig, ServerConfig
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.blob import BlobStorage
from supernote.server.services.file import FileService
from supernote.server.services.user import UserService


@pytest.fixture
def file_service(
    storage_root: Path,
    blob_storage: BlobStorage,
    user_service: UserService,
    session_manager: DatabaseSessionManager,
) -> FileService:
    return FileService(
        storage_root=storage_root,
        blob_storage=blob_storage,
        user_service=user_service,
        session_manager=session_manager,
        event_bus=MagicMock(),
    )


@pytest.fixture
def server_config_gemini(tmp_path: Path) -> ServerConfig:
    conf = ServerConfig(
        auth=AuthConfig(secret_key="secret"),
        storage_dir=str(tmp_path),
    )
    conf.gemini_api_key = "fake-key"
    conf.gemini_ocr_model = "gemini-2.0-flash-exp"
    conf.gemini_embedding_model = "text-embedding-004"
    return conf


@pytest.fixture
def fake_ocr_model() -> FakeOcrModel:
    return FakeOcrModel()


@pytest.fixture
def fake_embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def fake_summary_model() -> FakeSummaryModel:
    return FakeSummaryModel()
