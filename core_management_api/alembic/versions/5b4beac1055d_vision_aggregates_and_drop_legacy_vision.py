"""vision_aggregates and drop legacy vision tables

Revision ID: 5b4beac1055d
Revises: b1f7c4d2a890
Create Date: 2026-05-27

TTH-08 Fase 2 (DHU-024 §2): crea la tabla canonica `vision_aggregates` del modulo
de vision reescrito y borra las tablas legacy `vision_tracks` / `vision_flows`
(modeladas en 775d2d1db8b4 + hipertabilizadas en daec5fdcfcdd, nunca pobladas).
"""
from typing import Sequence, Union

import geoalchemy2
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b4beac1055d'
down_revision: Union[str, Sequence[str], None] = 'b1f7c4d2a890'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- 1. Crear vision_aggregates (tth-08-fase1-diseno.md §6.5) ---
    op.create_table(
        'vision_aggregates',
        sa.Column('camera_id', sa.String(), nullable=False),
        sa.Column('zone_id', sa.String(), nullable=False),
        # TIMESTAMPTZ deliberado: el dominio define window_start/window_end como
        # tz-aware UTC (§5.4); TIMESTAMP naive perderia esa garantia.
        sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('window_duration_seconds', sa.Float(), nullable=False),
        sa.Column('unique_vehicles', sa.Integer(), nullable=False),
        sa.Column('car_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('bus_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('truck_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('motorcycle_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('mean_speed_kmh', sa.Float(), nullable=True),
        sa.Column('flow_vehicles_per_hour', sa.Float(), nullable=False),
        sa.Column('mean_occupancy', sa.Float(), nullable=False),
        sa.Column('density_vehicles_per_km', sa.Float(), nullable=True),
        sa.Column('queue', sa.Integer(), nullable=True),
        sa.CheckConstraint('unique_vehicles >= 0', name='ck_vision_aggregates_unique_vehicles_nonneg'),
        sa.CheckConstraint('car_count >= 0', name='ck_vision_aggregates_car_count_nonneg'),
        sa.CheckConstraint('bus_count >= 0', name='ck_vision_aggregates_bus_count_nonneg'),
        sa.CheckConstraint('truck_count >= 0', name='ck_vision_aggregates_truck_count_nonneg'),
        sa.CheckConstraint('motorcycle_count >= 0', name='ck_vision_aggregates_motorcycle_count_nonneg'),
        sa.CheckConstraint('flow_vehicles_per_hour >= 0', name='ck_vision_aggregates_flow_nonneg'),
        sa.CheckConstraint('mean_occupancy BETWEEN 0 AND 1', name='ck_vision_aggregates_occupancy_range'),
        sa.PrimaryKeyConstraint('camera_id', 'zone_id', 'window_start'),
    )
    # Indice por tiempo explicito: suple el indice por defecto de create_hypertable
    # (evita un indice anonimo de TimescaleDB que generaria drift en autogenerate).
    op.create_index(op.f('ix_vision_aggregates_window_start'), 'vision_aggregates', ['window_start'], unique=False)
    op.execute(
        "SELECT create_hypertable('vision_aggregates', 'window_start', "
        "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
    )

    # --- 2. Borrar tablas legacy ---
    op.drop_index(op.f('ix_vision_tracks_track_uuid'), table_name='vision_tracks')
    op.drop_index(op.f('ix_vision_tracks_entry_timestamp'), table_name='vision_tracks')
    op.drop_table('vision_tracks')
    op.drop_index(op.f('ix_vision_flows_timestamp_bin'), table_name='vision_flows')
    op.drop_index(op.f('ix_vision_flows_flow_id'), table_name='vision_flows')
    op.drop_table('vision_flows')


def downgrade() -> None:
    # TimescaleDB no des-hipertabiliza: recrear tabla + indices + create_hypertable
    # tal como 775d2d1db8b4 (creacion) y daec5fdcfcdd (hypertables) las dejaron.
    op.create_table(
        'vision_flows',
        sa.Column('flow_id', sa.String(), nullable=False),
        sa.Column('camera_id', sa.String(), nullable=False),
        sa.Column('timestamp_bin', sa.DateTime(), nullable=False),
        sa.Column('period_seconds', sa.Integer(), nullable=False),
        sa.Column('from_edge_id', sa.String(), nullable=True),
        sa.Column('to_edge_id', sa.String(), nullable=True),
        sa.Column('turn_direction', sa.String(), nullable=True),
        sa.Column('vehicle_count', sa.Integer(), nullable=False),
        sa.Column('avg_speed_mps', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.camera_id'], ),
        sa.ForeignKeyConstraint(['from_edge_id'], ['graph_edges.edge_id'], ),
        sa.ForeignKeyConstraint(['to_edge_id'], ['graph_edges.edge_id'], ),
        sa.PrimaryKeyConstraint('flow_id', 'timestamp_bin')
    )
    op.create_index(op.f('ix_vision_flows_flow_id'), 'vision_flows', ['flow_id'], unique=False)
    op.create_index(op.f('ix_vision_flows_timestamp_bin'), 'vision_flows', ['timestamp_bin'], unique=False)
    op.create_table(
        'vision_tracks',
        sa.Column('track_uuid', sa.String(), nullable=False),
        sa.Column('camera_id', sa.String(), nullable=False),
        sa.Column('entry_timestamp', sa.DateTime(), nullable=False),
        sa.Column('exit_timestamp', sa.DateTime(), nullable=False),
        sa.Column('class_id', sa.Integer(), nullable=False),
        sa.Column('avg_speed_px', sa.Float(), nullable=False),
        sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='LINESTRING', srid=4326, dimension=2, from_text='ST_GeomFromEWKT', name='geometry'), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.camera_id'], ),
        sa.PrimaryKeyConstraint('track_uuid', 'entry_timestamp')
    )
    op.create_index(op.f('ix_vision_tracks_entry_timestamp'), 'vision_tracks', ['entry_timestamp'], unique=False)
    op.create_index(op.f('ix_vision_tracks_track_uuid'), 'vision_tracks', ['track_uuid'], unique=False)
    op.execute(
        "SELECT create_hypertable('vision_tracks', 'entry_timestamp', "
        "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
    )
    op.execute(
        "SELECT create_hypertable('vision_flows', 'timestamp_bin', "
        "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
    )

    op.drop_table('vision_aggregates')
