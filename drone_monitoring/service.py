from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from drone_monitoring.db import DEFAULT_DB_PATH, init_db, session, utc_now_iso
from drone_monitoring.processing import compute_metrics, haversine_m, heading_deg, process_track


logger = logging.getLogger(__name__)

VOLGOGRAD_CENTER = (48.7080, 44.5133)
DEFAULT_DELIVERY_ZONE_RADIUS_M = 120.0
LOW_BATTERY_THRESHOLD_PCT = 20.0
EXCESS_SPEED_THRESHOLD_MPS = 18.0
VOLGOGRAD_DEMO_POINTS = [
    (48.7063, 44.5058),
    (48.7079, 44.5134),
    (48.7107, 44.5186),
    (48.7138, 44.5215),
    (48.7169, 44.5253),
    (48.7197, 44.5282),
    (48.7121, 44.5098),
    (48.7038, 44.5161),
]


class MissionMonitoringService:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._bootstrapped = False

    def init_db(self) -> None:
        init_db(self.db_path)
        with session(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO users (id, username, display_name, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("user_dispatcher", "dispatcher", "Диспетчер", "dispatcher", utc_now_iso()),
            )
        self._bootstrapped = True

    def _ensure_ready(self) -> None:
        if not self._bootstrapped:
            self.init_db()

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:12]}"

    @staticmethod
    def _mission_code() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"MSN-{stamp}-{uuid4().hex[:4].upper()}"

    @staticmethod
    def _json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _loads(raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default

    @staticmethod
    def _point_from_meters(origin: Tuple[float, float], dx_m: float, dy_m: float) -> Tuple[float, float]:
        lat = origin[0] + dy_m / 111320.0
        lon = origin[1] + dx_m / max(1e-6, 111320.0 * math.cos(math.radians((origin[0] + lat) / 2.0)))
        return round(lat, 7), round(lon, 7)

    @staticmethod
    def _zone_contains(zone: Dict[str, Any], point: Tuple[float, float] | None) -> bool:
        if point is None:
            return False
        center_lat = zone.get("center_lat")
        center_lon = zone.get("center_lon")
        radius_m = zone.get("radius_m")
        if center_lat is None or center_lon is None or radius_m is None:
            return False
        return haversine_m((float(center_lat), float(center_lon)), point) <= float(radius_m)

    @staticmethod
    def _build_demo_layout(rng: random.Random) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
        start = rng.choice(VOLGOGRAD_DEMO_POINTS)
        candidates = [point for point in VOLGOGRAD_DEMO_POINTS if haversine_m(start, point) >= 700.0]
        delivery = rng.choice(candidates or VOLGOGRAD_DEMO_POINTS)
        midpoint = ((start[0] + delivery[0]) / 2.0, (start[1] + delivery[1]) / 2.0)
        no_fly = (
            round(midpoint[0] + rng.uniform(-0.0012, 0.0012), 7),
            round(midpoint[1] + rng.uniform(-0.0012, 0.0012), 7),
        )
        radius_m = rng.uniform(160.0, 260.0)
        return start, delivery, no_fly, radius_m

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        mission_id: str,
        drone_id: Optional[str],
        event_type: str,
        severity: str,
        title: str,
        message: str,
        event_at: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO mission_events (mission_id, drone_id, event_type, severity, title, message, event_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mission_id,
                drone_id,
                event_type,
                severity,
                title,
                message,
                event_at or utc_now_iso(),
                self._json(meta or {}),
            ),
        )

    def _next_seq(self, conn: sqlite3.Connection, mission_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq_no), 0) AS seq FROM track_points_raw WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        return int(row["seq"]) + 1

    def _get_mission_base(self, mission_id: str, conn: sqlite3.Connection) -> Dict[str, Any]:
        row = conn.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(f"Mission {mission_id} not found")
        return dict(row)

    def _serialize_drone(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "code": row["code"],
            "display_name": row["display_name"],
            "drone_type": row["drone_type"],
            "status": row["status"],
            "battery_pct": row["battery_pct"],
            "current_position": {"lat": row["current_lat"], "lon": row["current_lon"]},
            "current_alt_m": row["current_alt_m"],
            "last_seen_at": row["last_seen_at"],
            "meta": self._loads(row.get("meta_json"), {}),
        }

    def _mission_payload(self, conn: sqlite3.Connection, mission_row: Dict[str, Any]) -> Dict[str, Any]:
        geozones = self.get_geozones(mission_row["id"], conn=conn)
        latest_metrics = self.get_metrics(mission_row["id"], conn=conn)
        raw_count = conn.execute(
            "SELECT COUNT(*) AS c FROM track_points_raw WHERE mission_id = ?",
            (mission_row["id"],),
        ).fetchone()["c"]
        processed_count = conn.execute(
            "SELECT COUNT(*) AS c FROM track_points_processed WHERE mission_id = ?",
            (mission_row["id"],),
        ).fetchone()["c"]
        event_count = conn.execute(
            "SELECT COUNT(*) AS c FROM mission_events WHERE mission_id = ?",
            (mission_row["id"],),
        ).fetchone()["c"]
        drone = None
        if mission_row.get("drone_id"):
            drone_row = conn.execute("SELECT * FROM drones WHERE id = ?", (mission_row["drone_id"],)).fetchone()
            if drone_row is not None:
                drone = self._serialize_drone(dict(drone_row))
        return {
            "id": mission_row["id"],
            "code": mission_row["code"],
            "title": mission_row["title"],
            "description": mission_row.get("description"),
            "mission_type": mission_row["mission_type"],
            "city": mission_row["city"],
            "status": mission_row["status"],
            "launch_mode": mission_row["launch_mode"],
            "order_id": mission_row.get("order_id"),
            "drone_id": mission_row.get("drone_id"),
            "drone": drone,
            "start_point": {"lat": mission_row["start_lat"], "lon": mission_row["start_lon"]},
            "delivery_point": {"lat": mission_row["delivery_lat"], "lon": mission_row["delivery_lon"]},
            "started_at": mission_row.get("started_at"),
            "completed_at": mission_row.get("completed_at"),
            "created_at": mission_row["created_at"],
            "updated_at": mission_row["updated_at"],
            "meta": self._loads(mission_row.get("meta_json"), {}),
            "counts": {
                "raw_track_points": int(raw_count),
                "processed_track_points": int(processed_count),
                "events": int(event_count),
                "geozones": len(geozones),
            },
            "metrics": latest_metrics.get("metrics", {}),
            "latest_processing_run": latest_metrics.get("processing_run"),
            "geozones": geozones,
        }

    def create_live_mission_draft(
        self,
        title: Optional[str],
        city: str,
        start: Tuple[float, float],
        delivery: Tuple[float, float],
        no_fly_center: Optional[Tuple[float, float]],
        no_fly_radius_m: float,
        no_fly_zones: Optional[List[Dict[str, Any]]] = None,
        drone_type: str = "cargo",
    ) -> Dict[str, Any]:
        self._ensure_ready()
        mission_id = self._new_id("mission")
        now = utc_now_iso()
        title = title or f"Миссия доставки {datetime.now().strftime('%H:%M:%S')}"
        with session(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO missions (
                    id, code, title, description, mission_type, city, status, launch_mode,
                    start_lat, start_lon, delivery_lat, delivery_lon,
                    created_at, updated_at, meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mission_id,
                    self._mission_code(),
                    title,
                    "Миссия доставки для диспетчерского мониторинга.",
                    "delivery_live",
                    city,
                    "draft",
                    "live",
                    start[0],
                    start[1],
                    delivery[0],
                    delivery[1],
                    now,
                    now,
                    self._json({"drone_type": drone_type}),
                ),
            )
            geozones = [
                ("launch_zone", "Точка старта", start, 80.0),
                ("delivery_zone", "Зона выдачи", delivery, DEFAULT_DELIVERY_ZONE_RADIUS_M),
            ]
            zone_specs = list(no_fly_zones or [])
            if not zone_specs and no_fly_center is not None:
                zone_specs = [{"center": no_fly_center, "radius_m": no_fly_radius_m, "name": "Запретная зона"}]
            for zone_idx, zone_spec in enumerate(zone_specs, start=1):
                geozones.append(
                    (
                        "no_fly_zone",
                        str(zone_spec.get("name") or f"Запретная зона {zone_idx}"),
                        tuple(zone_spec.get("center") or no_fly_center),
                        float(zone_spec.get("radius_m") or no_fly_radius_m),
                    )
                )
            for zone_type, name, center, radius in geozones:
                conn.execute(
                    """
                    INSERT INTO geozones (id, mission_id, zone_type, name, center_lat, center_lon, radius_m, polygon_json, created_at, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._new_id("zone"),
                        mission_id,
                        zone_type,
                        name,
                        center[0],
                        center[1],
                        radius,
                        None,
                        now,
                        self._json({}),
                    ),
                )
            self._insert_event(
                conn,
                mission_id,
                None,
                "MISSION_STARTED",
                "info",
                "Миссия подготовлена",
                "Миссия создана и готова к запуску live-симуляции.",
                event_at=now,
                meta={"phase": "draft"},
            )
            mission = self._mission_payload(conn, self._get_mission_base(mission_id, conn))
        return mission

    def _replace_raw_track(
        self,
        conn: sqlite3.Connection,
        mission_id: str,
        drone_id: Optional[str],
        points: List[Dict[str, Any]],
    ) -> None:
        conn.execute("DELETE FROM track_points_raw WHERE mission_id = ?", (mission_id,))
        for point in points:
            conn.execute(
                """
                INSERT INTO track_points_raw (
                    mission_id, drone_id, ts, seq_no, lat, lon, alt_m, speed_mps, heading_deg,
                    hdop, sats, battery_pct, source, is_gap, meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mission_id,
                    drone_id,
                    point["ts"],
                    point["seq_no"],
                    point["lat"],
                    point["lon"],
                    point.get("alt_m"),
                    point.get("speed_mps"),
                    point.get("heading_deg"),
                    point.get("hdop"),
                    point.get("sats"),
                    point.get("battery_pct"),
                    point.get("source", "live"),
                    int(point.get("is_gap", 0)),
                    point.get("meta_json") or self._json({}),
                ),
            )

    def create_synthetic_mission(
        self,
        title: Optional[str] = None,
        city: str = "Volgograd, Russia",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_ready()
        rng = random.Random(seed)
        start, delivery, no_fly, radius_m = self._build_demo_layout(rng)
        mission = self.create_live_mission_draft(
            title=title or "Синтетическая миссия доставки",
            city=city,
            start=start,
            delivery=delivery,
            no_fly_center=no_fly,
            no_fly_radius_m=radius_m,
            no_fly_zones=[{"center": no_fly, "radius_m": radius_m, "name": "Синтетическая запретная зона"}],
            drone_type="cargo",
        )
        mission_id = mission["id"]
        now = datetime.now(timezone.utc).replace(microsecond=0)
        raw_points = []
        path = [
            start,
            self._point_from_meters(start, 350.0, 180.0),
            self._point_from_meters(delivery, -220.0, 150.0),
            delivery,
        ]
        total_seq = 0
        time_offset_s = 0
        battery = 100.0
        prev = None
        for seg_idx in range(len(path) - 1):
            a = path[seg_idx]
            b = path[seg_idx + 1]
            steps = max(18, int(haversine_m(a, b) / 12.0))
            for step in range(steps):
                ratio = step / max(1, steps - 1)
                lat = a[0] + (b[0] - a[0]) * ratio + rng.uniform(-0.00003, 0.00003)
                lon = a[1] + (b[1] - a[1]) * ratio + rng.uniform(-0.00003, 0.00003)
                if total_seq == 22:
                    time_offset_s += 4
                point_time = now + timedelta(seconds=total_seq + time_offset_s)
                total_seq += 1
                if prev is not None:
                    battery -= min(2.0, haversine_m(prev, (lat, lon)) / 650.0)
                prev = (lat, lon)
                raw_points.append(
                    {
                        "ts": point_time.isoformat(),
                        "seq_no": total_seq,
                        "lat": lat,
                        "lon": lon,
                        "alt_m": 55.0 + rng.uniform(-4.0, 6.0),
                        "speed_mps": 11.0 + rng.uniform(-1.8, 1.2),
                        "heading_deg": heading_deg(a, b),
                        "hdop": 0.8 + rng.uniform(0.0, 0.9),
                        "sats": int(rng.uniform(9, 16)),
                        "battery_pct": max(12.0, battery),
                        "source": "synthetic",
                        "meta_json": self._json({"seed": seed}),
                    }
                )
        if len(raw_points) > 35:
            raw_points[35]["lat"] += 0.0025
            raw_points[35]["lon"] -= 0.0020
        with session(self.db_path) as conn:
            self._replace_raw_track(conn, mission_id, None, raw_points)
            conn.execute(
                "UPDATE missions SET status = ?, updated_at = ? WHERE id = ?",
                ("synthetic_ready", utc_now_iso(), mission_id),
            )
            self._insert_event(
                conn,
                mission_id,
                None,
                "MISSION_COMPLETED",
                "info",
                "Синтетический трек подготовлен",
                "Для миссии создан синтетический сырой трек для демонстрации обработки GPS/GNSS.",
                meta={"points": len(raw_points)},
            )
        self.run_processing(mission_id, {"algorithm": "kalman_basic"})
        return self.get_mission(mission_id)

    def create_demo_live_mission(
        self,
        title: Optional[str] = None,
        city: str = "Volgograd, Russia",
        seed: Optional[int] = None,
        drone_type: str = "cargo",
    ) -> Dict[str, Any]:
        self._ensure_ready()
        rng = random.Random(seed)
        start, delivery, no_fly, radius_m = self._build_demo_layout(rng)
        return self.create_live_mission_draft(
            title=title or "Демо-миссия доставки",
            city=city,
            start=start,
            delivery=delivery,
            no_fly_center=no_fly,
            no_fly_radius_m=radius_m,
            no_fly_zones=[{"center": no_fly, "radius_m": radius_m, "name": "Демонстрационная запретная зона"}],
            drone_type=drone_type,
        )

    def register_live_order(self, mission_id: str, internal_order_id: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_ready()
        now = utc_now_iso()
        with session(self.db_path) as conn:
            order_row = conn.execute("SELECT id FROM orders WHERE mission_id = ?", (mission_id,)).fetchone()
            if order_row is None:
                conn.execute(
                    """
                    INSERT INTO orders (id, mission_id, external_order_id, status, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self._new_id("db_order"), mission_id, internal_order_id, "queued", self._json(payload or {}), now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE orders SET external_order_id = ?, status = ?, payload_json = ?, updated_at = ?
                    WHERE mission_id = ?
                    """,
                    (internal_order_id, "queued", self._json(payload or {}), now, mission_id),
                )
            conn.execute(
                "UPDATE missions SET order_id = ?, status = ?, updated_at = ? WHERE id = ?",
                (internal_order_id, "queued", now, mission_id),
            )

    def bind_assignment(self, order_id: str, drone_id: str, runtime_drone: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_ready()
        with session(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT mission_id FROM orders
                WHERE external_order_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (order_id,),
            ).fetchone()
            if row is None:
                return
            mission_id = row["mission_id"]
            now = utc_now_iso()
            conn.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE mission_id = ?",
                ("assigned", now, mission_id),
            )
            conn.execute(
                "UPDATE missions SET drone_id = ?, status = ?, updated_at = ? WHERE id = ?",
                (drone_id, "assigned", now, mission_id),
            )
        if runtime_drone is not None:
            self.upsert_drone_snapshot(drone_id, runtime_drone)

    def _upsert_drone_snapshot_conn(self, conn: sqlite3.Connection, drone_id: str, runtime_drone: Dict[str, Any]) -> None:
        now = utc_now_iso()
        pos = tuple(runtime_drone.get("pos") or (None, None))
        conn.execute(
            """
            INSERT INTO drones (
                id, code, drone_type, display_name, status, battery_pct,
                current_lat, current_lon, current_alt_m, last_seen_at, meta_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                drone_type = excluded.drone_type,
                display_name = excluded.display_name,
                status = excluded.status,
                battery_pct = excluded.battery_pct,
                current_lat = excluded.current_lat,
                current_lon = excluded.current_lon,
                current_alt_m = excluded.current_alt_m,
                last_seen_at = excluded.last_seen_at,
                meta_json = excluded.meta_json,
                updated_at = excluded.updated_at
            """,
            (
                drone_id,
                drone_id,
                runtime_drone.get("type", "cargo"),
                drone_id,
                runtime_drone.get("status", "unknown"),
                float(runtime_drone.get("battery", 100.0)),
                pos[0],
                pos[1],
                float(runtime_drone.get("alt_m", 60.0)),
                now,
                self._json(
                    {
                        "link_quality": runtime_drone.get("link_quality"),
                        "eta_s": runtime_drone.get("eta_s"),
                        "remaining_m": runtime_drone.get("remaining_m"),
                        "active_order_id": runtime_drone.get("active_order_id"),
                        "temp_c": runtime_drone.get("temp_c"),
                    }
                ),
                now,
                now,
            ),
        )

    def upsert_drone_snapshot(self, drone_id: str, runtime_drone: Dict[str, Any]) -> None:
        self._ensure_ready()
        with session(self.db_path) as conn:
            self._upsert_drone_snapshot_conn(conn, drone_id, runtime_drone)

    def record_live_telemetry(self, drone_id: str, runtime_drone: Dict[str, Any]) -> None:
        self._ensure_ready()
        active_order_id = runtime_drone.get("active_order_id")
        pos = runtime_drone.get("pos")
        if not active_order_id or not pos:
            self.upsert_drone_snapshot(drone_id, runtime_drone)
            return
        with session(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT o.mission_id, m.status
                FROM orders o
                JOIN missions m ON m.id = o.mission_id
                WHERE o.external_order_id = ?
                ORDER BY o.updated_at DESC
                LIMIT 1
                """,
                (active_order_id,),
            ).fetchone()
            if row is None:
                return
            mission_id = row["mission_id"]
            now = utc_now_iso()
            self._upsert_drone_snapshot_conn(conn, drone_id, runtime_drone)
            prev = conn.execute(
                """
                SELECT * FROM track_points_raw
                WHERE mission_id = ?
                ORDER BY seq_no DESC
                LIMIT 1
                """,
                (mission_id,),
            ).fetchone()
            prev_point = dict(prev) if prev else None
            speed_mps = float(runtime_drone.get("speed_mps") or 0.0)
            heading = None
            if prev_point is not None:
                prev_coords = (float(prev_point["lat"]), float(prev_point["lon"]))
                curr_coords = (float(pos[0]), float(pos[1]))
                distance = haversine_m(prev_coords, curr_coords)
                time_delta = max(1.0, (datetime.fromisoformat(now) - datetime.fromisoformat(prev_point["ts"])).total_seconds())
                speed_mps = distance / time_delta
                heading = heading_deg(prev_coords, curr_coords)
                if time_delta > 3.0:
                    self._insert_event(
                        conn,
                        mission_id,
                        drone_id,
                        "SIGNAL_GAP",
                        "warning",
                        "Обрыв телеметрии",
                        f"Получен разрыв телеметрии в реальном времени на {time_delta:.1f} с.",
                        event_at=now,
                        meta={"gap_s": round(time_delta, 2)},
                    )
            hdop = round(max(0.7, 2.4 - float(runtime_drone.get("link_quality", 0.7)) * 1.2), 2)
            sats = max(6, min(18, int(round(7 + float(runtime_drone.get("link_quality", 0.7)) * 10))))
            battery_pct = float(runtime_drone.get("battery", 100.0))
            conn.execute(
                """
                INSERT INTO track_points_raw (
                    mission_id, drone_id, ts, seq_no, lat, lon, alt_m, speed_mps, heading_deg,
                    hdop, sats, battery_pct, source, is_gap, meta_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mission_id,
                    drone_id,
                    now,
                    self._next_seq(conn, mission_id),
                    float(pos[0]),
                    float(pos[1]),
                    float(runtime_drone.get("alt_m", 60.0)),
                    speed_mps,
                    heading,
                    hdop,
                    sats,
                    battery_pct,
                    "live",
                    0,
                    self._json({"status": runtime_drone.get("status"), "link_quality": runtime_drone.get("link_quality")}),
                ),
            )
            if row["status"] in ("draft", "queued", "assigned"):
                conn.execute(
                    "UPDATE missions SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?, drone_id = ? WHERE id = ?",
                    ("in_progress", now, now, drone_id, mission_id),
                )
                self._insert_event(
                    conn,
                    mission_id,
                    drone_id,
                    "MISSION_STARTED",
                    "info",
                    "Миссия началась",
                    "Дрон начал выполнение миссии и передаёт телеметрию в реальном времени.",
                    event_at=now,
                )
            if prev_point is not None:
                prev_battery = float(prev_point.get("battery_pct") or 100.0)
                if prev_battery > LOW_BATTERY_THRESHOLD_PCT >= battery_pct:
                    self._insert_event(
                        conn,
                        mission_id,
                        drone_id,
                        "LOW_BATTERY",
                        "warning",
                        "Низкий заряд",
                        "Заряд дрона упал ниже 20%, дрон должен уйти на зарядку.",
                        event_at=now,
                        meta={"battery_pct": round(battery_pct, 2)},
                    )
                prev_speed = float(prev_point.get("speed_mps") or 0.0)
                if prev_speed <= EXCESS_SPEED_THRESHOLD_MPS < speed_mps:
                    self._insert_event(
                        conn,
                        mission_id,
                        drone_id,
                        "EXCESS_SPEED",
                        "warning",
                        "Превышение скорости",
                        f"Зафиксирована скорость {speed_mps:.1f} м/с.",
                        event_at=now,
                        meta={"speed_mps": round(speed_mps, 2)},
                    )
            geozones = self.get_geozones(mission_id, conn=conn)
            current_point = (float(pos[0]), float(pos[1]))
            previous_point = None
            if prev_point is not None:
                previous_point = (float(prev_point["lat"]), float(prev_point["lon"]))
            for zone in geozones:
                previous_inside = self._zone_contains(zone, previous_point)
                current_inside = self._zone_contains(zone, current_point)
                if zone["zone_type"] == "delivery_zone":
                    if not previous_inside and current_inside:
                        self._insert_event(
                            conn,
                            mission_id,
                            drone_id,
                            "ENTER_DELIVERY_ZONE",
                            "info",
                            "Вход в зону выдачи",
                            "Дрон вошёл в зону выдачи.",
                            event_at=now,
                        )
                    elif previous_inside and not current_inside:
                        self._insert_event(
                            conn,
                            mission_id,
                            drone_id,
                            "EXIT_DELIVERY_ZONE",
                            "info",
                            "Выход из зоны выдачи",
                            "Дрон покинул зону выдачи.",
                            event_at=now,
                        )
                if zone["zone_type"] == "no_fly_zone":
                    if not previous_inside and current_inside:
                        self._insert_event(
                            conn,
                            mission_id,
                            drone_id,
                            "ENTER_NO_FLY_ZONE",
                            "critical",
                            "Вход в запретную зону",
                            "Маршрут дрона пересёк запретную зону.",
                            event_at=now,
                        )
                    elif previous_inside and not current_inside:
                        self._insert_event(
                            conn,
                            mission_id,
                            drone_id,
                            "EXIT_NO_FLY_ZONE",
                            "warning",
                            "Выход из запретной зоны",
                            "Дрон покинул запретную зону.",
                            event_at=now,
                        )

    def complete_live_mission_for_order(self, order_id: str, runtime_drone: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_ready()
        with session(self.db_path) as conn:
            row = conn.execute(
                "SELECT mission_id FROM orders WHERE external_order_id = ? ORDER BY updated_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            if row is None:
                return
            mission_id = row["mission_id"]
            now = utc_now_iso()
            conn.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE mission_id = ?",
                ("completed", now, mission_id),
            )
            conn.execute(
                "UPDATE missions SET status = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                ("completed", now, now, mission_id),
            )
            self._insert_event(
                conn,
                mission_id,
                runtime_drone.get("id") if runtime_drone else None,
                "MISSION_COMPLETED",
                "info",
                "Миссия завершена",
                "Заказ завершён, миссия закрыта в мониторинге.",
                event_at=now,
            )

    def fail_live_mission_for_order(self, order_id: str, runtime_drone: Optional[Dict[str, Any]] = None, reason: str = "unknown") -> None:
        self._ensure_ready()
        with session(self.db_path) as conn:
            row = conn.execute(
                "SELECT mission_id FROM orders WHERE external_order_id = ? ORDER BY updated_at DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            if row is None:
                return
            mission_id = row["mission_id"]
            now = utc_now_iso()
            mission_row = conn.execute("SELECT meta_json FROM missions WHERE id = ?", (mission_id,)).fetchone()
            meta = self._loads(mission_row["meta_json"] if mission_row else None, {})
            meta["failure_reason"] = reason
            conn.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE mission_id = ?",
                ("failed", now, mission_id),
            )
            conn.execute(
                "UPDATE missions SET status = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                ("failed", now, now, mission_id),
            )
            conn.execute(
                "UPDATE missions SET meta_json = ? WHERE id = ?",
                (self._json(meta), mission_id),
            )
            self._insert_event(
                conn,
                mission_id,
                runtime_drone.get("id") if runtime_drone else None,
                "EMERGENCY_LANDING",
                "critical",
                "Аварийная посадка",
                "Дрон полностью разрядился и выполнил аварийную посадку.",
                event_at=now,
                meta={"reason": reason},
            )

    def list_missions(self, scope: str = "active") -> List[Dict[str, Any]]:
        self._ensure_ready()
        with session(self.db_path) as conn:
            scope_value = (scope or "active").strip().lower()
            if scope_value == "archive":
                rows = conn.execute(
                    """
                    SELECT * FROM missions
                    WHERE status IN ('completed', 'cancelled', 'failed', 'synthetic_ready')
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            elif scope_value == "all":
                rows = conn.execute("SELECT * FROM missions ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM missions
                    WHERE status NOT IN ('completed', 'cancelled', 'failed', 'synthetic_ready')
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            return [self._mission_payload(conn, dict(row)) for row in rows]

    def get_mission(self, mission_id: str) -> Dict[str, Any]:
        self._ensure_ready()
        with session(self.db_path) as conn:
            mission_row = self._get_mission_base(mission_id, conn)
            return self._mission_payload(conn, mission_row)

    def get_raw_track(self, mission_id: str) -> Dict[str, Any]:
        self._ensure_ready()
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM track_points_raw WHERE mission_id = ? ORDER BY seq_no ASC",
                (mission_id,),
            ).fetchall()
            points = []
            for row in rows:
                item = dict(row)
                item["meta"] = self._loads(item.pop("meta_json", None), {})
                points.append(item)
            return {"mission_id": mission_id, "points": points}

    def get_geozones(self, mission_id: str, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        self._ensure_ready()
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            close_conn = True
        try:
            rows = conn.execute(
                "SELECT * FROM geozones WHERE mission_id = ? ORDER BY created_at ASC",
                (mission_id,),
            ).fetchall()
            zones = []
            for row in rows:
                item = dict(row)
                item["meta"] = self._loads(item.pop("meta_json", None), {})
                item["polygon"] = self._loads(item.pop("polygon_json", None), None)
                zones.append(item)
            return zones
        finally:
            if close_conn:
                conn.close()

    def get_events(self, mission_id: str) -> Dict[str, Any]:
        self._ensure_ready()
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM mission_events WHERE mission_id = ? ORDER BY event_at DESC, id DESC",
                (mission_id,),
            ).fetchall()
            events = []
            for row in rows:
                item = dict(row)
                item["meta"] = self._loads(item.pop("meta_json", None), {})
                events.append(item)
            return {"mission_id": mission_id, "events": events}

    def get_latest_processing_run_id(self, conn: sqlite3.Connection, mission_id: str) -> Optional[str]:
        row = conn.execute(
            """
            SELECT id FROM processing_runs
            WHERE mission_id = ? AND status = 'completed'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (mission_id,),
        ).fetchone()
        return row["id"] if row is not None else None

    def get_processed_track(self, mission_id: str) -> Dict[str, Any]:
        self._ensure_ready()
        self.ensure_processed_track(mission_id)
        with session(self.db_path) as conn:
            run_id = self.get_latest_processing_run_id(conn, mission_id)
            if run_id is None:
                return {"mission_id": mission_id, "processing_run_id": None, "points": []}
            rows = conn.execute(
                """
                SELECT * FROM track_points_processed
                WHERE mission_id = ? AND processing_run_id = ?
                ORDER BY seq_no ASC
                """,
                (mission_id, run_id),
            ).fetchall()
            points = []
            for row in rows:
                item = dict(row)
                item["flags"] = self._loads(item.pop("flags_json", None), {})
                item["meta"] = self._loads(item.pop("meta_json", None), {})
                points.append(item)
            return {"mission_id": mission_id, "processing_run_id": run_id, "points": points}

    def ensure_processed_track(self, mission_id: str) -> None:
        self._ensure_ready()
        with session(self.db_path) as conn:
            existing = conn.execute(
                "SELECT COUNT(*) AS c FROM track_points_processed WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()["c"]
            raw_count = conn.execute(
                "SELECT COUNT(*) AS c FROM track_points_raw WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()["c"]
        if existing == 0 and raw_count >= 2:
            self.run_processing(mission_id, {"algorithm": "kalman_basic"})

    def run_processing(self, mission_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_ready()
        with session(self.db_path) as conn:
            raw_rows = conn.execute(
                "SELECT * FROM track_points_raw WHERE mission_id = ? ORDER BY seq_no ASC",
                (mission_id,),
            ).fetchall()
            raw_points = [dict(row) for row in raw_rows]
            if len(raw_points) < 2:
                return {"mission_id": mission_id, "processing_run_id": None, "summary": {"raw_points": len(raw_points)}}
            run_id = self._new_id("proc")
            started_at = utc_now_iso()
            conn.execute(
                """
                INSERT INTO processing_runs (id, mission_id, algorithm, status, started_at, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    mission_id,
                    str(config.get("algorithm", "kalman_basic")),
                    "running",
                    started_at,
                    self._json(config),
                ),
            )
            result = process_track(raw_points, config)
            summary = result["summary"]
            processed_points = result["processed_points"]
            metrics = compute_metrics(raw_points, processed_points, summary)
            conn.execute("DELETE FROM track_points_processed WHERE mission_id = ?", (mission_id,))
            conn.execute("DELETE FROM mission_metrics WHERE mission_id = ?", (mission_id,))
            for point in processed_points:
                conn.execute(
                    """
                    INSERT INTO track_points_processed (
                        mission_id, processing_run_id, ts, seq_no, lat, lon, alt_m, speed_mps,
                        heading_deg, filter_name, flags_json, source_raw_id, meta_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mission_id,
                        run_id,
                        point["ts"],
                        point["seq_no"],
                        point["lat"],
                        point["lon"],
                        point.get("alt_m"),
                        point.get("speed_mps"),
                        point.get("heading_deg"),
                        point["filter_name"],
                        self._json(point.get("flags") or {}),
                        point.get("source_raw_id"),
                        self._json(point.get("meta") or {}),
                    ),
                )
            for event in result["events"]:
                self._insert_event(
                    conn,
                    mission_id,
                    None,
                    event["event_type"],
                    event["severity"],
                    event["title"],
                    event["message"],
                    event_at=event["event_at"],
                    meta=event.get("meta") or {},
                )
            created_at = utc_now_iso()
            metric_units = {
                "RMSE": "m",
                "CEP50": "m",
                "gap_share": "ratio",
                "anomaly_count": "count",
                "avg_hdop": "hdop",
            }
            for name, value in metrics.items():
                conn.execute(
                    """
                    INSERT INTO mission_metrics (mission_id, processing_run_id, metric_name, metric_value, metric_unit, created_at, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mission_id, run_id, name, float(value), metric_units.get(name), created_at, self._json({})),
                )
            conn.execute(
                """
                UPDATE processing_runs
                SET status = ?, finished_at = ?, summary_json = ?
                WHERE id = ?
                """,
                ("completed", utc_now_iso(), self._json(summary), run_id),
            )
            conn.execute(
                "UPDATE missions SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), mission_id),
            )
            return {
                "mission_id": mission_id,
                "processing_run_id": run_id,
                "summary": summary,
                "metrics": metrics,
            }

    def get_metrics(self, mission_id: str, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        self._ensure_ready()
        close_conn = False
        if conn is None:
            self.ensure_processed_track(mission_id)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            close_conn = True
        try:
            run_id = self.get_latest_processing_run_id(conn, mission_id)
            rows = conn.execute(
                """
                SELECT * FROM mission_metrics
                WHERE mission_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (mission_id,),
            ).fetchall()
            metrics: Dict[str, float] = {}
            units: Dict[str, str | None] = {}
            for row in rows:
                if row["metric_name"] not in metrics:
                    metrics[row["metric_name"]] = float(row["metric_value"])
                    units[row["metric_name"]] = row["metric_unit"]
            return {"mission_id": mission_id, "processing_run": run_id, "metrics": metrics, "units": units}
        finally:
            if close_conn:
                conn.close()

    def get_drone_monitoring(self, drone_id: str, runtime_drone: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_ready()
        if runtime_drone is not None:
            self.upsert_drone_snapshot(drone_id, runtime_drone)
        with session(self.db_path) as conn:
            drone_row = conn.execute("SELECT * FROM drones WHERE id = ?", (drone_id,)).fetchone()
            mission_row = conn.execute(
                """
                SELECT * FROM missions
                WHERE drone_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (drone_id,),
            ).fetchone()
            recent_events = conn.execute(
                """
                SELECT * FROM mission_events
                WHERE drone_id = ?
                ORDER BY event_at DESC, id DESC
                LIMIT 20
                """,
                (drone_id,),
            ).fetchall()
            payload = {
                "drone": self._serialize_drone(dict(drone_row)) if drone_row is not None else None,
                "mission": self._mission_payload(conn, dict(mission_row)) if mission_row is not None else None,
                "events": [
                    {
                        **dict(row),
                        "meta": self._loads(row["meta_json"], {}),
                    }
                    for row in recent_events
                ],
            }
            if runtime_drone is not None:
                payload["runtime"] = {
                    "status": runtime_drone.get("status"),
                    "battery": runtime_drone.get("battery"),
                    "eta_s": runtime_drone.get("eta_s"),
                    "remaining_m": runtime_drone.get("remaining_m"),
                    "route": runtime_drone.get("route") or [],
                    "history": runtime_drone.get("history") or [],
                    "link_quality": runtime_drone.get("link_quality"),
            }
            return payload

    def cancel_mission(self, mission_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_ready()
        now = utc_now_iso()
        with session(self.db_path) as conn:
            mission = self._get_mission_base(mission_id, conn)
            meta = self._loads(mission.get("meta_json"), {})
            if reason:
                meta["cancel_reason"] = reason
            conn.execute(
                "UPDATE missions SET status = ?, completed_at = ?, updated_at = ?, drone_id = NULL, meta_json = ? WHERE id = ?",
                ("cancelled", now, now, self._json(meta), mission_id),
            )
            conn.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE mission_id = ?",
                ("cancelled", now, mission_id),
            )
            message = "Диспетчер отменил миссию во время выполнения или ожидания."
            if reason:
                message = f"{message} Причина: {reason}"
            self._insert_event(
                conn,
                mission_id,
                None,
                "MISSION_CANCELLED",
                "warning",
                "Миссия отменена",
                message,
                event_at=now,
                meta={"reason": reason} if reason else {},
            )
            return self._mission_payload(conn, self._get_mission_base(mission_id, conn))

    def update_mission_destination(self, mission_id: str, destination: Tuple[float, float]) -> Dict[str, Any]:
        self._ensure_ready()
        now = utc_now_iso()
        with session(self.db_path) as conn:
            conn.execute(
                """
                UPDATE missions
                SET delivery_lat = ?, delivery_lon = ?, updated_at = ?
                WHERE id = ?
                """,
                (float(destination[0]), float(destination[1]), now, mission_id),
            )
            conn.execute(
                """
                UPDATE geozones
                SET center_lat = ?, center_lon = ?
                WHERE mission_id = ? AND zone_type = 'delivery_zone'
                """,
                (float(destination[0]), float(destination[1]), mission_id),
            )
            self._insert_event(
                conn,
                mission_id,
                None,
                "MISSION_UPDATED",
                "info",
                "Точка доставки изменена",
                "Диспетчер изменил точку назначения миссии во время выполнения.",
                event_at=now,
                meta={"delivery_lat": float(destination[0]), "delivery_lon": float(destination[1])},
            )
            return self._mission_payload(conn, self._get_mission_base(mission_id, conn))

    def reconcile_runtime_assignments(self, active_order_ids: set[str]) -> int:
        self._ensure_ready()
        now = utc_now_iso()
        with session(self.db_path) as conn:
            stale_rows = conn.execute(
                """
                SELECT id, order_id FROM missions
                WHERE status IN ('draft', 'queued', 'assigned', 'in_progress')
                """
            ).fetchall()
            stale_ids = [row["id"] for row in stale_rows if (row["order_id"] or "") not in active_order_ids]
            if not stale_ids:
                return 0
            for mission_id in stale_ids:
                conn.execute(
                    "UPDATE missions SET status = ?, completed_at = ?, updated_at = ?, drone_id = NULL WHERE id = ?",
                    ("cancelled", now, now, mission_id),
                )
                conn.execute(
                    "UPDATE orders SET status = ?, updated_at = ? WHERE mission_id = ?",
                    ("cancelled", now, mission_id),
                )
                self._insert_event(
                    conn,
                    mission_id,
                    None,
                    "MISSION_CANCELLED",
                    "warning",
                    "Live-сеанс завершён",
                    "После перезапуска приложения активный runtime-заказ для миссии не найден, поэтому миссия перенесена в архив.",
                    event_at=now,
                    meta={"reason": "runtime_reconciled"},
                )
            return len(stale_ids)

    def upsert_mission_no_fly_zone(
        self,
        mission_id: str,
        zone_id: str,
        center: Tuple[float, float],
        radius_m: float,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_ready()
        now = utc_now_iso()
        with session(self.db_path) as conn:
            exists = conn.execute(
                "SELECT id FROM geozones WHERE id = ?",
                (zone_id,),
            ).fetchone()
            if exists is None:
                conn.execute(
                    """
                    INSERT INTO geozones (id, mission_id, zone_type, name, center_lat, center_lon, radius_m, polygon_json, created_at, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        zone_id,
                        mission_id,
                        "no_fly_zone",
                        name or "Запретная зона",
                        float(center[0]),
                        float(center[1]),
                        float(radius_m),
                        None,
                        now,
                        self._json({"created_by": "dispatcher"}),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE geozones
                    SET center_lat = ?, center_lon = ?, radius_m = ?, name = ?, meta_json = ?
                    WHERE id = ?
                    """,
                    (
                        float(center[0]),
                        float(center[1]),
                        float(radius_m),
                        name or "Запретная зона",
                        self._json({"updated_by": "dispatcher"}),
                        zone_id,
                    ),
                )
            self._insert_event(
                conn,
                mission_id,
                None,
                "MISSION_UPDATED",
                "warning",
                "Команда диспетчера: новая запретная зона",
                "Диспетчер добавил или обновил запретную зону для активной миссии.",
                event_at=now,
                meta={"zone_id": zone_id, "center_lat": float(center[0]), "center_lon": float(center[1]), "radius_m": float(radius_m)},
            )
            return self._mission_payload(conn, self._get_mission_base(mission_id, conn))

    def remove_mission_zone(self, zone_id: str) -> Optional[str]:
        self._ensure_ready()
        with session(self.db_path) as conn:
            row = conn.execute(
                "SELECT mission_id, name, zone_type FROM geozones WHERE id = ?",
                (zone_id,),
            ).fetchone()
            if row is None:
                return None
            mission_id = row["mission_id"]
            zone_name = row["name"]
            zone_type = row["zone_type"]
            conn.execute("DELETE FROM geozones WHERE id = ?", (zone_id,))
            if zone_type == "no_fly_zone":
                self._insert_event(
                    conn,
                    mission_id,
                    None,
                    "MISSION_UPDATED",
                    "info",
                    "Команда диспетчера: удалена запретная зона",
                    f"Диспетчер удалил запретную зону '{zone_name}'.",
                    event_at=utc_now_iso(),
                    meta={"zone_id": zone_id},
                )
            return mission_id
