"""add predictions table

Revision ID: f2c9d7a4b6e1
Revises: 5b4beac1055d
Create Date: 2026-05-31 12:00:00.000000

TTH-09 Fase 5 (CT-09.5): persistencia append-only de cada predicción del GRU.
Grano fila-por-paso: una fila por (inferencia × dirección × paso); una llamada a
/predictions/predict produce 4 direcciones × 30 pasos = 120 filas. ``intersection_id``
es el opaco del request (SIN FK a graph_nodes, contrato §8); el modelo es clasificador,
se persiste ``level`` (argmax) + ``probs`` (6 softmax), NO un ratio. Fuente de datos de
HU-14 (el join predicho-vs-real es HU-14, fuera de esta tabla).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f2c9d7a4b6e1"
down_revision: Union[str, Sequence[str], None] = "5b4beac1055d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "predictions",
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("intersection_id", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column(
            "probs",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("prediction_id"),
    )
    op.create_index(
        op.f("ix_predictions_prediction_id"),
        "predictions",
        ["prediction_id"],
        unique=False,
    )
    op.create_index(
        "ix_predictions_intersection_id_generated_at",
        "predictions",
        ["intersection_id", "generated_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_predictions_intersection_id_generated_at", table_name="predictions"
    )
    op.drop_index(
        op.f("ix_predictions_prediction_id"), table_name="predictions"
    )
    op.drop_table("predictions")
