"""
Configuration for the adaptive control engine (TTH-10, CT-10.4 + CT-10.6).

Extracts the constants previously hardcoded in ``application/`` to a single
``BaseSettings`` so they can be overridden at deploy time via environment
variables, without exposing them to the Administrator in MVP1
(DHU-014 subsección C).

Values are normative (CT-10.6 cites the Manual MTC peruano R.D. N.° 26-2024-MTC/18
for the five MTC constants; CT-10.4 cites SUMO calibration for the 1500 veh/h
peak threshold). They must remain identical to the previous hardcoded
defaults until validation evidence justifies recalibration.

Env var convention: ``CONTROL_<SECTION>__<FIELD>`` (double underscore as
nested delimiter). Example: ``CONTROL_ADAPTIVE__PEAK_THRESHOLD=1500.0``.
"""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebsterSettings(BaseModel):
    min_cycle: float = 30.0
    max_cycle: float = 120.0


class MaxPressureSettings(BaseModel):
    default_cycle: float = 60.0


class MTCSettings(BaseModel):
    min_green: int = 7
    max_green: int = 60
    min_yellow: int = 3
    all_red: int = 2
    min_pedestrian: int = 7
    max_cycle: float = 120.0


class AdaptiveSettings(BaseModel):
    peak_threshold: float = 1500.0


class ControlSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONTROL_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    webster: WebsterSettings = Field(default_factory=WebsterSettings)
    max_pressure: MaxPressureSettings = Field(default_factory=MaxPressureSettings)
    mtc: MTCSettings = Field(default_factory=MTCSettings)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)
