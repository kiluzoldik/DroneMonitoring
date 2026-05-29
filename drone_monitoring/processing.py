from __future__ import annotations

import math
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Tuple


def parse_ts(value: str | datetime) -> datetime:
    """Преобразует строковую временную метку в объект datetime."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Вычисляет расстояние между двумя географическими координатами в метрах."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    s = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
    )
    return 6371000.0 * 2.0 * math.atan2(math.sqrt(s), math.sqrt(max(1e-12, 1.0 - s)))


def heading_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Вычисляет направление движения между двумя точками в градусах."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def project_m(point: Tuple[float, float], origin: Tuple[float, float]) -> Tuple[float, float]:
    """Проецирует географическую точку в локальную метрическую систему координат."""
    lat, lon = point
    o_lat, o_lon = origin
    y = (lat - o_lat) * 111320.0
    x = (lon - o_lon) * 111320.0 * math.cos(math.radians((lat + o_lat) / 2.0))
    return x, y


def unproject_m(xy: Tuple[float, float], origin: Tuple[float, float]) -> Tuple[float, float]:
    """Преобразует локальные метрические координаты обратно в широту и долготу."""
    x, y = xy
    o_lat, o_lon = origin
    lat = o_lat + y / 111320.0
    lon = o_lon + x / max(1e-6, 111320.0 * math.cos(math.radians((lat + o_lat) / 2.0)))
    return lat, lon


class ScalarKalman:
    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 18.0):
        """Инициализирует."""
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = 0.0
        self.error = 1.0
        self.initialized = False

    def update(self, measurement: float) -> float:
        """Обновляет."""
        if not self.initialized:
            self.estimate = measurement
            self.error = 1.0
            self.initialized = True
            return measurement
        self.error += self.process_noise
        gain = self.error / (self.error + self.measurement_noise)
        self.estimate = self.estimate + gain * (measurement - self.estimate)
        self.error = (1.0 - gain) * self.error
        return self.estimate


def process_track(raw_points: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Обрабатывает сырой GNSS-трек: сортирует точки, ищет разрывы, фильтрует выбросы и строит сглаженный трек."""
    signal_gap_seconds = int(config.get("signal_gap_seconds", 3))
    interpolate_gap_seconds = int(config.get("interpolate_gap_seconds", 5))
    max_speed_mps = float(config.get("max_speed_mps", 18.0))
    ordered = sorted(raw_points, key=lambda item: (parse_ts(item["ts"]), float(item.get("seq_no", 0))))
    cleaned: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    filtered_outliers = 0
    missing_seconds = 0
    interpolated_points = 0
    prev_actual: Dict[str, Any] | None = None

    for point in ordered:
        current = dict(point)
        current["ts"] = parse_ts(current["ts"]).isoformat()
        current["flags"] = {}
        if prev_actual is not None:
            prev_ts = parse_ts(prev_actual["ts"])
            curr_ts = parse_ts(current["ts"])
            dt = max(1.0, (curr_ts - prev_ts).total_seconds())
            distance = haversine_m(
                (float(prev_actual["lat"]), float(prev_actual["lon"])),
                (float(current["lat"]), float(current["lon"])),
            )
            derived_speed = distance / max(1.0, dt)
            if dt > signal_gap_seconds:
                missing_seconds += max(0, int(round(dt)) - 1)
                events.append(
                    {
                        "event_type": "SIGNAL_GAP",
                        "severity": "warning",
                        "title": "Разрыв GPS-сигнала",
                        "message": f"Обнаружен временной разрыв телеметрии длительностью {dt:.1f} с.",
                        "event_at": current["ts"],
                        "meta": {"duration_s": round(dt, 2)},
                    }
                )
            if derived_speed > max_speed_mps * 1.8:
                filtered_outliers += 1
                events.append(
                    {
                        "event_type": "FILTER_OUTLIER",
                        "severity": "warning",
                        "title": "Выброс координаты",
                        "message": f"Точка seq={current.get('seq_no')} исключена как выброс GPS.",
                        "event_at": current["ts"],
                        "meta": {"derived_speed_mps": round(derived_speed, 2)},
                    }
                )
                continue
            if 1.0 < dt <= interpolate_gap_seconds + 1:
                for step in range(1, int(dt)):
                    ratio = step / dt
                    interpolated_points += 1
                    cleaned.append(
                        {
                            "id": None,
                            "ts": (prev_ts + (curr_ts - prev_ts) * ratio).isoformat(),
                            "seq_no": float(prev_actual.get("seq_no", 0)) + ratio,
                            "lat": float(prev_actual["lat"]) + (float(current["lat"]) - float(prev_actual["lat"])) * ratio,
                            "lon": float(prev_actual["lon"]) + (float(current["lon"]) - float(prev_actual["lon"])) * ratio,
                            "alt_m": float(prev_actual.get("alt_m") or 0.0) + (float(current.get("alt_m") or 0.0) - float(prev_actual.get("alt_m") or 0.0)) * ratio,
                            "speed_mps": None,
                            "heading_deg": None,
                            "hdop": float(prev_actual.get("hdop") or current.get("hdop") or 1.2),
                            "sats": int(round((int(prev_actual.get("sats") or 10) + int(current.get("sats") or 10)) / 2.0)),
                            "battery_pct": float(prev_actual.get("battery_pct") or current.get("battery_pct") or 100.0),
                            "source": "interpolated",
                            "flags": {"interpolated": True},
                            "source_raw_id": None,
                        }
                    )
        current["source_raw_id"] = current.get("id")
        cleaned.append(current)
        prev_actual = current

    processed: List[Dict[str, Any]] = []
    if cleaned:
        origin = (float(cleaned[0]["lat"]), float(cleaned[0]["lon"]))
        x_filter = ScalarKalman()
        y_filter = ScalarKalman()
        prev_point: Dict[str, Any] | None = None
        for idx, point in enumerate(cleaned, start=1):
            x, y = project_m((float(point["lat"]), float(point["lon"])), origin)
            sx = x_filter.update(x)
            sy = y_filter.update(y)
            lat, lon = unproject_m((sx, sy), origin)
            processed_point = {
                "ts": point["ts"],
                "seq_no": idx,
                "lat": lat,
                "lon": lon,
                "alt_m": point.get("alt_m"),
                "speed_mps": point.get("speed_mps"),
                "heading_deg": point.get("heading_deg"),
                "filter_name": str(config.get("algorithm", "kalman_basic")),
                "flags": dict(point.get("flags") or {}),
                "source_raw_id": point.get("source_raw_id"),
                "meta": {"source": point.get("source")},
            }
            if prev_point is not None:
                dt = max(1.0, (parse_ts(processed_point["ts"]) - parse_ts(prev_point["ts"])).total_seconds())
                processed_point["speed_mps"] = haversine_m(
                    (float(prev_point["lat"]), float(prev_point["lon"])),
                    (float(processed_point["lat"]), float(processed_point["lon"])),
                ) / dt
                processed_point["heading_deg"] = heading_deg(
                    (float(prev_point["lat"]), float(prev_point["lon"])),
                    (float(processed_point["lat"]), float(processed_point["lon"])),
                )
            prev_point = processed_point
            processed.append(processed_point)

    duration_seconds = 0.0
    if len(ordered) >= 2:
        duration_seconds = max(
            1.0,
            (parse_ts(ordered[-1]["ts"]) - parse_ts(ordered[0]["ts"])).total_seconds(),
        )
    return {
        "processed_points": processed,
        "events": events,
        "summary": {
            "raw_points": len(ordered),
            "processed_points": len(processed),
            "filtered_outliers": filtered_outliers,
            "interpolated_points": interpolated_points,
            "missing_seconds": missing_seconds,
            "gap_share": min(1.0, missing_seconds / duration_seconds) if duration_seconds else 0.0,
        },
    }


def compute_metrics(raw_points: List[Dict[str, Any]], processed_points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, float]:
    """Вычисляет метрики качества обработанного GNSS-трека."""
    processed_by_raw = {
        int(point["source_raw_id"]): point
        for point in processed_points
        if point.get("source_raw_id") is not None
    }
    errors = []
    hdops = []
    for raw in raw_points:
        raw_id = raw.get("id")
        if raw.get("hdop") is not None:
            hdops.append(float(raw["hdop"]))
        if raw_id in processed_by_raw:
            errors.append(
                haversine_m(
                    (float(raw["lat"]), float(raw["lon"])),
                    (float(processed_by_raw[raw_id]["lat"]), float(processed_by_raw[raw_id]["lon"])),
                )
            )
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors)) if errors else 0.0
    cep50 = float(median(errors)) if errors else 0.0
    return {
        "RMSE": round(rmse, 3),
        "CEP50": round(cep50, 3),
        "gap_share": round(float(summary.get("gap_share", 0.0)), 6),
        "anomaly_count": float(summary.get("filtered_outliers", 0)),
        "avg_hdop": round(sum(hdops) / len(hdops), 3) if hdops else 0.0,
    }
