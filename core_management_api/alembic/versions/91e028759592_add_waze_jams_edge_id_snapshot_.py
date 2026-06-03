"""add waze_jams edge_id snapshot_timestamp index

Revision ID: 91e028759592
Revises: f2c9d7a4b6e1
Create Date: 2026-06-02 17:28:17.973634

TTH-13 Fase 1 (CT-13.3): índice compuesto ``(edge_id, snapshot_timestamp)`` sobre la
hypertable ``waze_jams``. Optimiza el read-path de la serie del día por arista
(``WazeJamsRepo.series_for_day``), que filtra por rango de ``snapshot_timestamp`` y
ordena por ``(edge_id, snapshot_timestamp)``. Calca el índice compuesto de
``predictions`` (``ix_predictions_intersection_id_generated_at``). ``op.create_index``
sobre una hypertable ya tiene precedente en el repo (``vision_aggregates``,
revisión ``5b4beac1055d``), así que es Alembic puro sin variante Timescale.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '91e028759592'
down_revision: Union[str, Sequence[str], None] = 'f2c9d7a4b6e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "ix_waze_jams_edge_id_snapshot_timestamp",
        "waze_jams",
        ["edge_id", "snapshot_timestamp"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_waze_jams_edge_id_snapshot_timestamp", table_name="waze_jams"
    )
