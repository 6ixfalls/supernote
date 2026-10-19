from supernote.models.auth import LoginVO
from supernote.models.base import BooleanEnum
from supernote.models.file_web import (
    FileLabelSearchDTO,
    FileLabelSearchVO,
    UserFileSearchVO,
    UserFileVO,
)


def test_user_file_vo_datetime_parsing() -> None:
    json_data = """
    {
        "id": "123",
        "directoryId": "456",
        "fileName": "test.txt",
        "size": 100,
        "md5": "abc",
        "isFolder": "N",
        "createTime": "176722962336",
        "updateTime": "176722963237"
    }
    """
    vo = UserFileVO.from_json(json_data)

    assert vo.id == "123"
    assert vo.create_time == 176722962336
    assert vo.update_time == 176722963237

    assert vo.is_folder == BooleanEnum.NO
    # inner_name matches default None
    assert vo.inner_name is None


def test_user_file_vo_optional_fields() -> None:
    # Test with minimum fields to ensure defaults work
    json_data = """
    {
        "id": "123",
        "directoryId": "456",
        "fileName": "test.txt",
        "size": 100,
        "md5": "abc"
    }
    """
    vo = UserFileVO.from_json(json_data)
    assert vo.is_folder == BooleanEnum.NO
    assert vo.create_time is None
    assert vo.update_time is None


def test_file_label_search_dto_and_vo() -> None:
    dto = FileLabelSearchDTO(file_name="note", directory_id=0)
    d = dto.to_dict()
    assert d["fileName"] == "note"
    assert d["directoryId"] == 0
    assert FileLabelSearchDTO.from_dict(d) == dto

    vo = FileLabelSearchVO(
        user_file_search_vo_list=[
            UserFileSearchVO(id="1", directory_id="0", file_name="note.note")
        ]
    )
    vd = vo.to_dict()
    assert len(vd["userFileSearchVOList"]) == 1
    assert vd["userFileSearchVOList"][0]["fileName"] == "note.note"


def test_login_vo_avatars_and_update_time() -> None:
    vo = LoginVO(
        token="jwt",
        counts="0",
        user_name="Alice",
        avatars_url="https://example.com/a.png",
        last_update_time="2026-10-19T00:00:00Z",
    )
    d = vo.to_dict()
    assert d["avatarsUrl"] == "https://example.com/a.png"
    assert d["lastUpdateTime"] == "2026-10-19T00:00:00Z"
    assert LoginVO.from_dict(d) == vo
