from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DEFAULT_DB_PATH = os.environ.get("MONITORING_DB_PATH", os.path.join("data", "monitoring.sqlite3"))


def utc_now_iso() -> str:
    """Возвращает текущую дату и время в UTC в ISO-формате."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_db_dir(db_path: str = DEFAULT_DB_PATH) -> None:
    """Создает каталог для файла базы данных при необходимости."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Открывает подключение к SQLite и включает параметры, нужные для устойчивой работы."""
    ensure_db_dir(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    # WAL dramatically improves concurrent reads during live telemetry writes.
    # If the filesystem does not support it, fall back to MEMORY journal.
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def session(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Предоставляет контекстную сессию SQLite с автоматическим commit или rollback."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Создает таблицы базы данных мониторинга, если они еще не были созданы."""
    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS drones (
        id TEXT PRIMARY KEY,
        code TEXT NOT NULL UNIQUE,
        drone_type TEXT NOT NULL,
        display_name TEXT NOT NULL,
        status TEXT NOT NULL,
        battery_pct REAL NOT NULL DEFAULT 100,
        current_lat REAL,
        current_lon REAL,
        current_alt_m REAL,
        last_seen_at TEXT,
        meta_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS missions (
        id TEXT PRIMARY KEY,
        code TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        description TEXT,
        mission_type TEXT NOT NULL,
        city TEXT NOT NULL,
        status TEXT NOT NULL,
        launch_mode TEXT NOT NULL,
        drone_id TEXT,
        order_id TEXT,
        start_lat REAL NOT NULL,
        start_lon REAL NOT NULL,
        delivery_lat REAL NOT NULL,
        delivery_lon REAL NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        meta_json TEXT,
        FOREIGN KEY (drone_id) REFERENCES drones(id)
    );

    CREATE TABLE IF NOT EXISTS orders (
        id TEXT PRIMARY KEY,
        mission_id TEXT NOT NULL,
        external_order_id TEXT,
        status TEXT NOT NULL,
        payload_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS geozones (
        id TEXT PRIMARY KEY,
        mission_id TEXT NOT NULL,
        zone_type TEXT NOT NULL,
        name TEXT NOT NULL,
        center_lat REAL,
        center_lon REAL,
        radius_m REAL,
        polygon_json TEXT,
        created_at TEXT NOT NULL,
        meta_json TEXT,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS processing_runs (
        id TEXT PRIMARY KEY,
        mission_id TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        config_json TEXT,
        summary_json TEXT,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS track_points_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mission_id TEXT NOT NULL,
        drone_id TEXT,
        ts TEXT NOT NULL,
        seq_no INTEGER NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        alt_m REAL,
        speed_mps REAL,
        heading_deg REAL,
        hdop REAL,
        sats INTEGER,
        battery_pct REAL,
        source TEXT NOT NULL,
        is_gap INTEGER NOT NULL DEFAULT 0,
        meta_json TEXT,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE,
        FOREIGN KEY (drone_id) REFERENCES drones(id)
    );

    CREATE TABLE IF NOT EXISTS track_points_processed (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mission_id TEXT NOT NULL,
        processing_run_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        seq_no INTEGER NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        alt_m REAL,
        speed_mps REAL,
        heading_deg REAL,
        filter_name TEXT NOT NULL,
        flags_json TEXT,
        source_raw_id INTEGER,
        meta_json TEXT,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE,
        FOREIGN KEY (processing_run_id) REFERENCES processing_runs(id) ON DELETE CASCADE,
        FOREIGN KEY (source_raw_id) REFERENCES track_points_raw(id)
    );

    CREATE TABLE IF NOT EXISTS mission_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mission_id TEXT NOT NULL,
        drone_id TEXT,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        event_at TEXT NOT NULL,
        meta_json TEXT,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE,
        FOREIGN KEY (drone_id) REFERENCES drones(id)
    );

    CREATE TABLE IF NOT EXISTS mission_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mission_id TEXT NOT NULL,
        processing_run_id TEXT,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metric_unit TEXT,
        created_at TEXT NOT NULL,
        meta_json TEXT,
        FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE,
        FOREIGN KEY (processing_run_id) REFERENCES processing_runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status);
    CREATE INDEX IF NOT EXISTS idx_orders_mission_id ON orders(mission_id);
    CREATE INDEX IF NOT EXISTS idx_geozones_mission_id ON geozones(mission_id);
    CREATE INDEX IF NOT EXISTS idx_raw_mission_ts ON track_points_raw(mission_id, ts);
    CREATE INDEX IF NOT EXISTS idx_raw_mission_seq ON track_points_raw(mission_id, seq_no);
    CREATE INDEX IF NOT EXISTS idx_processed_mission_seq ON track_points_processed(mission_id, seq_no);
    CREATE INDEX IF NOT EXISTS idx_events_mission_time ON mission_events(mission_id, event_at);
    CREATE INDEX IF NOT EXISTS idx_metrics_mission_name ON mission_metrics(mission_id, metric_name);
    """
    with session(db_path) as conn:
        conn.executescript(schema)
