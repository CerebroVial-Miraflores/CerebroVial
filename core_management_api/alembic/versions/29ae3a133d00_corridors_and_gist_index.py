"""corridors + corridor_edges + GiST idempotente sobre graph_edges.geom

Revision ID: 29ae3a133d00
Revises: f3a9c1d2e4b7
Create Date: 2026-06-07 12:00:00.000000

Track feature/tomtom (Fase B-1) — solo DDL:

- ``corridors``: corredor dibujado por el operador (entidad propia). NADA de tráfico
  de TomTom vive acá.
- ``corridor_edges``: puente corredor → graph_edges (dirigido; el sentido vive en
  ``sequence``). ``tomtom_openlr`` es un IDENTIFICADOR de segmento (OpenLR, estándar
  abierto), NO un Result de tráfico: los valores de tráfico y la geometría de TomTom
  JAMÁS se persisten (ToS 11.6.1). UniqueConstraint(corridor_id, sequence) protege el
  orden de la cadena.
- GiST idempotente sobre ``graph_edges.geom``: el índice existe en la BD de dev viva
  pero está COMENTADO en la migración inicial (775d2d1db8b4:59), así que una BD recreada
  desde Alembic NO lo tendría y el matching geométrico de Fase B haría seq scan. Se crea
  con ``IF NOT EXISTS`` y guard de dialecto (PostgreSQL-only, patrón populate_geom_from_edges)
  para no chocar con la BD que ya lo tiene ni romper SQLite (tests). Cierra DEUDA-GIST-MIGRACION.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "29ae3a133d00"
down_revision: Union[str, Sequence[str], None] = "f3a9c1d2e4b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "corridors",
        sa.Column("corridor_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("corridor_id"),
    )

    op.create_table(
        "corridor_edges",
        sa.Column("corridor_id", sa.String(), nullable=False),
        sa.Column("edge_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("tomtom_openlr", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["corridor_id"], ["corridors.corridor_id"]),
        sa.ForeignKeyConstraint(["edge_id"], ["graph_edges.edge_id"]),
        sa.PrimaryKeyConstraint("corridor_id", "edge_id"),
        sa.UniqueConstraint(
            "corridor_id", "sequence", name="uq_corridor_edges_corridor_sequence"
        ),
    )

    # GiST sobre graph_edges.geom: idempotente + PostgreSQL-only. Cierra la brecha de
    # reproducibilidad (el índice vive en la BD de dev pero no en las migraciones).
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_geom "
            "ON graph_edges USING gist (geom)"
        )


def downgrade() -> None:
    op.drop_table("corridor_edges")
    op.drop_table("corridors")

    # NO se dropea idx_graph_edges_geom a propósito: preexistía en la BD viva FUERA de
    # Alembic (migración inicial lo tiene comentado). Dropearlo en downgrade rompería esa
    # BD de quien nunca lo creó vía migración. El downgrade solo revierte lo que ESTA
    # revisión agregó como tablas; el índice queda intacto.
