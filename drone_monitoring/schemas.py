from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    lat: float
    lon: float


class NoFlyCircle(BaseModel):
    center: Coordinate
    radius_m: float = Field(default=250.0, ge=50.0, le=3000.0)


class SyntheticMissionRequest(BaseModel):
    city: str = "Volgograd, Russia"
    title: Optional[str] = None
    seed: Optional[int] = None
    start_live: bool = False


class LiveMissionRequest(BaseModel):
    city: str = "Volgograd, Russia"
    title: Optional[str] = None
    start: Coordinate
    delivery: Coordinate
    no_fly_center: Optional[Coordinate] = None
    no_fly_radius_m: float = Field(default=250.0, ge=50.0, le=2000.0)
    no_fly_zones: Optional[list[NoFlyCircle]] = None
    drone_type: str = "cargo"


class LaunchMissionRequest(BaseModel):
    drone_type: str = "cargo"


class ProcessMissionRequest(BaseModel):
    algorithm: str = "kalman_basic"
    signal_gap_seconds: int = Field(default=3, ge=2, le=30)
    interpolate_gap_seconds: int = Field(default=5, ge=1, le=30)
    max_speed_mps: float = Field(default=18.0, ge=5.0, le=50.0)


class UpdateMissionDestinationRequest(BaseModel):
    destination: Coordinate


class CancelMissionRequest(BaseModel):
    reason: Optional[str] = None
