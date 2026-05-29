from math import isclose
from pathlib import Path
from uuid import uuid4

from drone_monitoring.service import MissionMonitoringService


def _build_service(db_name: str) -> tuple[MissionMonitoringService, Path]:
    """Строит сервисную операцию."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    stem = Path(db_name).stem
    db_path = data_dir / f"{stem}_{uuid4().hex[:8]}.sqlite3"
    service = MissionMonitoringService(str(db_path))
    service.init_db()
    return service, db_path


def _cleanup_db(db_path: Path) -> None:
    """Очищает тестовую базу данных."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
        except PermissionError:
            pass


def test_live_mission_supports_multiple_no_fly_zones_and_destination_update():
    """Проверяет live-миссию с несколькими запретными зонами и изменением точки доставки."""
    service, db_path = _build_service("test_monitoring_service_live.sqlite3")
    try:
        mission = service.create_live_mission_draft(
            title="Dispatcher Live Mission",
            city="Volgograd, Russia",
            start=(48.7080, 44.5133),
            delivery=(48.7132, 44.5291),
            no_fly_center=(48.7100, 44.5200),
            no_fly_radius_m=220.0,
            no_fly_zones=[
                {"center": (48.7108, 44.5212), "radius_m": 180.0, "name": "NFZ Alpha"},
                {"center": (48.7165, 44.5258), "radius_m": 240.0, "name": "NFZ Beta"},
            ],
            drone_type="cargo",
        )

        geozones = service.get_geozones(mission["id"])
        no_fly_zones = [zone for zone in geozones if zone["zone_type"] == "no_fly_zone"]
        assert len(no_fly_zones) == 2

        updated = service.update_mission_destination(mission["id"], (48.7194, 44.5384))
        assert isclose(updated["delivery_point"]["lat"], 48.7194, rel_tol=0.0, abs_tol=1e-6)
        assert isclose(updated["delivery_point"]["lon"], 44.5384, rel_tol=0.0, abs_tol=1e-6)

        delivery_zone = next(zone for zone in service.get_geozones(mission["id"]) if zone["zone_type"] == "delivery_zone")
        assert isclose(delivery_zone["center_lat"], 48.7194, rel_tol=0.0, abs_tol=1e-6)
        assert isclose(delivery_zone["center_lon"], 44.5384, rel_tol=0.0, abs_tol=1e-6)

        event_types = [event["event_type"] for event in service.get_events(mission["id"])["events"]]
        assert "MISSION_STARTED" in event_types
        assert "MISSION_UPDATED" in event_types
    finally:
        _cleanup_db(db_path)


def test_synthetic_mission_persists_tracks_events_and_metrics():
    """Проверяет синтетическую миссию с сохранением треков, событий и метрик."""
    service, db_path = _build_service("test_monitoring_service_synth.sqlite3")
    try:
        mission = service.create_synthetic_mission(
            title="Synthetic Monitoring Mission",
            city="Volgograd, Russia",
            seed=42,
        )

        raw_points = service.get_raw_track(mission["id"])["points"]
        processed_points = service.get_processed_track(mission["id"])["points"]
        metrics = service.get_metrics(mission["id"])["metrics"]
        event_types = [event["event_type"] for event in service.get_events(mission["id"])["events"]]

        assert len(raw_points) > 10
        assert len(processed_points) > 10
        assert {"RMSE", "CEP50", "gap_share", "anomaly_count", "avg_hdop"}.issubset(metrics.keys())
        assert "MISSION_COMPLETED" in event_types
    finally:
        _cleanup_db(db_path)
