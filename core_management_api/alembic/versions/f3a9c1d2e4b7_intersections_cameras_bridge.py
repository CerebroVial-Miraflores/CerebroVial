"""intersections + intersection_edges + cameras accesorio (Fase A)

Revision ID: f3a9c1d2e4b7
Revises: 91e028759592
Create Date: 2026-06-05 10:00:00.000000

Fase A — modelo de datos de intersecciones semaforizadas del PMU de Miraflores:

- ``intersections``: entidad de primera clase (intersection_id == nombre del mapeo
  ``documentation/contracts/mapeo_pmu_edges_v2.yaml``). junction_id es el junction SUMO;
  tls_id se puebla solo cuando está verificado (hoy: larco_benavides — DEUDA-CTRL-TLS).
- ``intersection_edges``: puente intersección → graph_edges con dirección. FK real a
  graph_edges.edge_id (id SUMO crudo): exige el net real cargado
  (``scripts/build_graph_geometry.py``) antes de sembrar.
- ``cameras``: degradada a accesorio de intersección. Se agrega intersection_id
  (FK→intersections) y stream_url; se quita node_id (FK→graph_nodes).

geom con spatial_index=False: la fase NO crea índices GIST (consistente con los GIST
comentados de la migración inicial 775d2d1db8b4). D-016.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import geoalchemy2


revision: str = "f3a9c1d2e4b7"
down_revision: Union[str, Sequence[str], None] = "91e028759592"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "intersections",
        sa.Column("intersection_id", sa.String(), nullable=False),
        sa.Column("junction_id", sa.String(), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("los_pmu", sa.String(), nullable=True),
        sa.Column("tls_id", sa.String(), nullable=True),
        sa.Column(
            "geom",
            geoalchemy2.types.Geometry(
                geometry_type="POINT",
                srid=4326,
                dimension=2,
                from_text="ST_GeomFromEWKT",
                name="geometry",
                spatial_index=False,
            ),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("intersection_id"),
    )
    op.create_index(
        op.f("ix_intersections_intersection_id"),
        "intersections",
        ["intersection_id"],
        unique=False,
    )

    op.create_table(
        "intersection_edges",
        sa.Column("intersection_id", sa.String(), nullable=False),
        sa.Column("edge_id", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["intersection_id"], ["intersections.intersection_id"]
        ),
        sa.ForeignKeyConstraint(["edge_id"], ["graph_edges.edge_id"]),
        sa.PrimaryKeyConstraint("intersection_id", "edge_id"),
        sa.CheckConstraint(
            "direction IN ('incoming', 'outgoing')",
            name="ck_intersection_edges_direction",
        ),
    )

    # cameras: accesorio de intersección. Agrega intersection_id + stream_url,
    # quita node_id (en PostgreSQL, drop_column elimina su FK dependiente).
    op.add_column(
        "cameras",
        sa.Column(
            "intersection_id",
            sa.String(),
            sa.ForeignKey("intersections.intersection_id"),
            nullable=True,
        ),
    )
    op.add_column("cameras", sa.Column("stream_url", sa.String(), nullable=True))
    op.drop_column("cameras", "node_id")


def downgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column(
            "node_id",
            sa.String(),
            sa.ForeignKey("graph_nodes.node_id"),
            nullable=True,
        ),
    )
    op.drop_column("cameras", "stream_url")
    op.drop_column("cameras", "intersection_id")

    op.drop_table("intersection_edges")
    op.drop_index(
        op.f("ix_intersections_intersection_id"), table_name="intersections"
    )
    op.drop_table("intersections")
