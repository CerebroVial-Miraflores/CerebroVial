"""add motor_decisions and engine_active_state

Revision ID: b1f7c4d2a890
Revises: 99319147948b
Create Date: 2026-05-26 14:00:00.000000

TTH-10 (parcial): persistencia append-only de decisiones del motor adaptativo
y entidad mutable de estrategia vigente por intersección. Las dos tablas
anclan a graph_nodes.node_id (DHU-021 V1 — write-path resuelve el
intersection_id opaco del payload contra esta FK).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b1f7c4d2a890"
down_revision: Union[str, Sequence[str], None] = "99319147948b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "motor_decisions",
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("node_id", sa.String(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("cycle_seconds", sa.Float(), nullable=False),
        sa.Column("flow_total", sa.Float(), nullable=False),
        sa.Column("y_load_factor", sa.Float(), nullable=True),
        sa.Column("next_phase", sa.String(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column(
            "phase_timings",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "adjustments",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "inputs_snapshot",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["node_id"], ["graph_nodes.node_id"]),
        sa.PrimaryKeyConstraint("decision_id"),
    )
    op.create_index(
        op.f("ix_motor_decisions_decision_id"),
        "motor_decisions",
        ["decision_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_motor_decisions_node_id"),
        "motor_decisions",
        ["node_id"],
        unique=False,
    )
    op.create_index(
        "ix_motor_decisions_node_id_decided_at",
        "motor_decisions",
        ["node_id", sa.text("decided_at DESC")],
        unique=False,
    )

    op.create_table(
        "engine_active_state",
        sa.Column("node_id", sa.String(), nullable=False),
        sa.Column("active_decision_id", sa.String(length=36), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=False),
        sa.Column("activated_by", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["node_id"], ["graph_nodes.node_id"]),
        sa.ForeignKeyConstraint(
            ["active_decision_id"], ["motor_decisions.decision_id"]
        ),
        sa.PrimaryKeyConstraint("node_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("engine_active_state")
    op.drop_index(
        "ix_motor_decisions_node_id_decided_at", table_name="motor_decisions"
    )
    op.drop_index(
        op.f("ix_motor_decisions_node_id"), table_name="motor_decisions"
    )
    op.drop_index(
        op.f("ix_motor_decisions_decision_id"), table_name="motor_decisions"
    )
    op.drop_table("motor_decisions")
