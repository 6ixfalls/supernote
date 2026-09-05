import time

from sqlalchemy import BigInteger, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from supernote.server.db.base import Base


class DeviceBindRequestDO(Base):
    """A device bind awaiting approval from the target account."""

    __tablename__ = "device_bind_requests"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "equipment_no", name="uq_device_bind_requests_user_equipment"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True)
    equipment_no: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String)
    total_capacity: Mapped[str] = mapped_column(String)
    create_time: Mapped[int] = mapped_column(
        BigInteger, default=lambda: int(time.time() * 1000)
    )
