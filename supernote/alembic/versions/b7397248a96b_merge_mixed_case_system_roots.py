"""merge mixed-case system root folders into the firmware's names

Revision ID: b7397248a96b
Revises: 68964804740d
Create Date: 2026-10-18 15:00:00.000000

"""

import logging
import time
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7397248a96b"
down_revision: Union[str, None] = "68964804740d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

ROOT_RENAMES = {"Export": "EXPORT", "Inbox": "INBOX", "Screenshot": "SCREENSHOT"}


def _children(bind: sa.Connection, user_id: int, parent_id: int) -> list[sa.Row]:
    return list(
        bind.execute(
            sa.text(
                "SELECT id, file_name, is_folder FROM f_user_file "
                "WHERE user_id = :user_id AND directory_id = :parent_id "
                "AND is_active = 'Y' ORDER BY id"
            ),
            {"user_id": user_id, "parent_id": parent_id},
        )
    )


def _free_name(name: str, is_folder: str, taken: set[tuple[str, str]]) -> str:
    """Pick `name (n).ext` the way the server autorenames a moved item."""
    base, ext = name, ""
    if is_folder != "Y" and "." in name:
        base, ext = name.rsplit(".", 1)
        ext = f".{ext}"
    counter = 1
    while (candidate := f"{base} ({counter}){ext}", is_folder) in taken:
        counter += 1
    return candidate


def _merge(
    bind: sa.Connection, user_id: int, src_id: int, dst_id: int, now: int
) -> None:
    """Move everything in folder `src_id` into folder `dst_id`, then retire `src_id`."""
    existing = {(c.file_name, c.is_folder): c for c in _children(bind, user_id, dst_id)}
    src_children = _children(bind, user_id, src_id)
    placed = set(existing)
    taken = set(existing) | {(c.file_name, c.is_folder) for c in src_children}

    for child in src_children:
        clash = existing.get((child.file_name, child.is_folder))
        if clash is not None and child.is_folder == "Y":
            _merge(bind, user_id, child.id, clash.id, now)
            continue

        new_name = child.file_name
        if (new_name, child.is_folder) in placed:
            new_name = _free_name(child.file_name, child.is_folder, taken)
        placed.add((new_name, child.is_folder))
        taken.add((new_name, child.is_folder))
        existing[(new_name, child.is_folder)] = child
        bind.execute(
            sa.text(
                "UPDATE f_user_file SET directory_id = :dst, file_name = :name, "
                "update_time = :now WHERE id = :id"
            ),
            {"dst": dst_id, "name": new_name, "now": now, "id": child.id},
        )

    # Reparent any soft-deleted (Recycle Bin) children so restoring them finds an active parent.
    bind.execute(
        sa.text(
            "UPDATE f_user_file SET directory_id = :dst, update_time = :now "
            "WHERE user_id = :user_id AND directory_id = :src AND is_active = 'N'"
        ),
        {"dst": dst_id, "src": src_id, "user_id": user_id, "now": now},
    )
    bind.execute(
        sa.text("DELETE FROM f_user_file WHERE id = :id"),
        {"id": src_id},
    )


def upgrade() -> None:
    bind = op.get_bind()
    now = int(time.time() * 1000)

    for old_name, new_name in ROOT_RENAMES.items():
        rows = bind.execute(
            sa.text(
                "SELECT id, user_id FROM f_user_file WHERE directory_id = 0 "
                "AND file_name = :name AND is_folder = 'Y' AND is_active = 'Y' "
                "ORDER BY id"
            ),
            {"name": old_name},
        ).fetchall()

        for row in rows:
            target = bind.execute(
                sa.text(
                    "SELECT id FROM f_user_file WHERE user_id = :user_id "
                    "AND directory_id = 0 AND file_name = :name "
                    "AND is_folder = 'Y' AND is_active = 'Y' ORDER BY id LIMIT 1"
                ),
                {"user_id": row.user_id, "name": new_name},
            ).scalar()

            if target is None:
                bind.execute(
                    sa.text(
                        "UPDATE f_user_file SET file_name = :name, update_time = :now "
                        "WHERE id = :id"
                    ),
                    {"name": new_name, "now": now, "id": row.id},
                )
                logger.info(
                    "Renamed root %s to %s for user %s", old_name, new_name, row.user_id
                )
            else:
                _merge(bind, row.user_id, row.id, target, now)
                logger.info(
                    "Merged root %s into %s for user %s",
                    old_name,
                    new_name,
                    row.user_id,
                )


def downgrade() -> None:
    # Merged folders cannot be split apart again; the names are left as they are.
    pass
