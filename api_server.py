from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
import math
import networkx as nx
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import os
import base64
import json
import time
from datetime import datetime
from redis import Redis

from data_service import DataService
from drone_monitoring import MissionMonitoringService
from drone_monitoring.api import register_monitoring_routes
from graph_service import GraphService
from routing_service import RoutingService, MODE_EMPTY, MODE_LOADED

logger = logging.getLogger(__name__)

# Planning constants (single source: routing_service has defaults)
# 20% резерв: нельзя использовать больше 80% на полезный маршрут (требование бизнес-логики)
RESERVE_PCT = 20.0
# Запас при планировании заказа (для грузового уже учтён CARGO_RESERVE_AFTER_DELIVERY_PCT)
PLAN_RESERVE_PCT = 20.0
# В тестовом режиме — высокий запас, маршрут через одну или несколько станций
PLAN_RESERVE_PCT_TEST = 55.0
# Макс. доля батареи на один сегмент без зарядки (80% usable): если сегмент «съедает» больше — строим через станции
MAX_SEGMENT_BATTERY_PCT = 80.0
# Допустимая доля батареи на перелёт до зарядки: можно «дотянуть» до станции с меньшим запасом (после посадки — 100%)
MAX_BATTERY_PCT_TO_REACH_CHARGER = 85.0
CHARGER_ARRIVAL_MIN_PCT = 5.0
# Порог заряда (%): при достижении дрон летит на станцию зарядки (если свободен или держит груз)
FLY_TO_CHARGER_AT_PCT = 20.0
# У грузового дрона после доставки должен оставаться минимум 20% заряда
CARGO_RESERVE_AFTER_DELIVERY_PCT = 20.0
STATION_NEAR_METERS = 20.0
# Станция зарядки: запас заряженных аккумуляторов (смена вместо зарядки на месте)
STATION_CHARGED_BATTERIES_MAX = 20
# Тиков на зарядку одного аккумулятора после смены
STATION_BATTERY_CHARGE_TICKS = 25
# Минимальное число городских станций, чтобы хватало на дальние сценарии
MIN_CITY_STATION_COUNT = 24
# Оценка: тиков на одну остановку на зарядку (база: 4% за тик, с 20% до 100% ≈ 20 тиков). 1 тик ≈ 1 сек.
CHARGE_TICKS_ESTIMATE_PER_STOP = 20
# Порог завершения зарядки: избегаем зависаний на 99.x% из-за плавающей точности.
CHARGE_COMPLETE_PCT = 99.0
# Скорость по умолчанию для оценки ETA (м/с), если ветер неизвестен
DEFAULT_SPEED_MPS = 12.0
# Текущая демонстрационная скорость увеличена в 2.5 раза относительно прежнего множителя 2.0.
SIMULATION_SPEED_MULTIPLIER = 5.0
AIR_ROUTE_STEP_M = 140.0
AIR_ZONE_PADDING_M = 35.0
EMERGENCY_LANDING_BATTERY_PCT = 0.5
LIVE_TELEMETRY_QUEUE_MAX = 4096
PUBLIC_HISTORY_LIMIT = 90
ASSIGN_PLAN_BACKOFF_MIN_S = 2.0
ASSIGN_PLAN_BACKOFF_MAX_S = 20.0
MAX_PLAN_ATTEMPTS_PER_TICK = 2
MISSION_POINT_NEAR_METERS = 18.0
SERVICE_STOP_TICKS = 5

# Failure reasons for plan_order_trip
NO_PATH_TO_PICKUP = "NO_PATH_TO_PICKUP"
NO_FEASIBLE_CHARGING_CHAIN_TO_PICKUP = "NO_FEASIBLE_CHARGING_CHAIN_TO_PICKUP"
NO_FEASIBLE_CHAIN_PICKUP_TO_DROPOFF_LOADED = "NO_FEASIBLE_CHAIN_PICKUP_TO_DROPOFF_LOADED"
NO_ESCAPE_AFTER_DROPOFF = "NO_ESCAPE_AFTER_DROPOFF"

app = FastAPI(title="Drone Planner API", version="0.2.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Минимальная 1×1 PNG (прозрачный пиксель) — чтобы не было 404 на /favicon.ico
FAVICON_PNG = base64.b64decode(
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
	"""Возвращает минимальную иконку сайта, чтобы браузер не получал ошибку 404."""
	return Response(content=FAVICON_PNG, media_type="image/png")

# Services
_data_service = DataService()
_graph_service = GraphService()
_routing_service = RoutingService(_graph_service)
_monitoring_service = MissionMonitoringService()

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
try:
    _redis: Redis | None = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    _redis.ping()
except Exception:
    _redis = None
    logger.warning("Redis is not available; running without persistence")

# Models
class LoadCityRequest(BaseModel):
	city: str
	drone_type: str = "cargo"

class AddOrderRequest(BaseModel):
	address_from: Optional[str] = None
	address_to: Optional[str] = None
	coords_from: Optional[List[float]] = None  # [lat, lon]
	coords_to: Optional[List[float]] = None
	type_hint: Optional[str] = None  # delivery|shooting|work
	# Явный тип дрона: cargo | operator | cleaner. Если не задан — выводится из type_hint.
	drone_type: Optional[str] = None
	# Приоритет (больше = выше приоритет в очереди). По умолчанию 5.
	priority: int = 5
	battery_level: float = 100
	waypoints: Optional[List[List[float]]] = None  # optional waypoint coords [ [lat,lon], ... ]
	# Для операторского дрона: область облёта (лассо) — список точек [ [lat,lon], ... ]
	area_polygon: Optional[List[List[float]]] = None  # [ [lat,lon], ... ]

class NoFlyZone(BaseModel):
	# Rectangle or circle zone for runtime rerouting.
	zone_type: str = "rectangle"
	lat_min: Optional[float] = None
	lat_max: Optional[float] = None
	lon_min: Optional[float] = None
	lon_max: Optional[float] = None
	center_lat: Optional[float] = None
	center_lon: Optional[float] = None
	radius_m: Optional[float] = None
	id: Optional[str] = None
	mission_id: Optional[str] = None

# State
STATE: Dict[str, Any] = {
	"city": None,
	"city_graph": None,
	"orders": [],
	"drones": {},  # drone_id -> {pos, type, battery, route, target_idx, status}
	"no_fly_zones": [],
	"clients": set(),  # websockets
	"zone_version": 0,
	"weather": {"wind_mps": 3.0},
    "base": None,  # base location lat, lon
	"inventory": {},  # type -> count available at base
	"stations": [],  # list of (lat, lon)
    "station_queues": {},  # station_index -> {charging:[], queue:[], capacity:int} (legacy for base) or {charged_batteries:int, charging_queue:[ticks], queue:[drone_id]}
    "base_queue": {"charging": [], "queue": [], "capacity": 2},
	"charger_nodes": {"base": None, "stations": []},
	"primary_component_nodes": set(),
	"battery_mode": "reality",  # "reality" | "test" — в тесте укороченная дальность для проверки маршрутов
}

LIVE_TELEMETRY_QUEUE: Optional[asyncio.Queue[Tuple[str, Dict[str, Any]]]] = None
LIVE_TELEMETRY_LAST_SENT: Dict[str, Dict[str, Any]] = {}

DEMO_CITY = "Volgograd, Russia"
DEMO_BASE = (48.7080, 44.5133)
DEMO_STATIONS = [
	(48.7146, 44.5198),
	(48.7062, 44.5064),
	(48.7208, 44.5276),
]


def _available_graph_view(G) -> nx.Graph:
	"""Выполняет служебную операцию: доступный граф представление."""
	view = nx.Graph()
	if G is None:
		return view
	for node, attr in G.nodes(data=True):
		try:
			node_weight = float(attr.get("weight", 1.0))
		except Exception:
			node_weight = 1.0
		if math.isfinite(node_weight):
			view.add_node(node)
	for u, v, data in G.edges(data=True):
		if u not in view or v not in view:
			continue
		try:
			edge_weight = float(data.get("weight", 1.0))
		except Exception:
			edge_weight = 1.0
		if math.isfinite(edge_weight):
			view.add_edge(u, v)
	return view


def _nearest_node_from_candidates(G, point: Tuple[float, float], candidates: List[Any]) -> Optional[Any]:
	"""Находит ближайшую узел из candidates."""
	if G is None or not point or len(point) != 2 or not candidates:
		return None
	best_node = None
	best_dist = float("inf")
	lat, lon = float(point[0]), float(point[1])
	for node in candidates:
		try:
			pos = G.nodes[node].get("pos")
			if not isinstance(pos, (list, tuple)) or len(pos) != 2:
				continue
			lat_diff = (lat - float(pos[0])) * 111000.0
			avg_lat = (lat + float(pos[0])) / 2.0
			lon_diff = (lon - float(pos[1])) * 111000.0 * max(0.01, math.cos(math.radians(avg_lat)))
			dist = math.sqrt(lat_diff ** 2 + lon_diff ** 2)
			if dist < best_dist:
				best_dist = dist
				best_node = node
		except Exception:
			continue
	return best_node


def _refresh_primary_component_nodes() -> None:
	"""Обновляет основной компонент узлы."""
	G = STATE.get("city_graph")
	if G is None or len(G.nodes) == 0:
		STATE["primary_component_nodes"] = set()
		return
	view = _available_graph_view(G)
	if len(view.nodes) == 0:
		STATE["primary_component_nodes"] = set()
		return
	components = list(nx.connected_components(view))
	if not components:
		STATE["primary_component_nodes"] = set()
		return
	largest_component = max(components, key=len)
	base = STATE.get("base")
	base_node = None
	if base and isinstance(base, (list, tuple)) and len(base) == 2:
		base_node = _nearest_node_from_candidates(G, tuple(base), list(largest_component))
	if base_node is not None:
		for component in components:
			if base_node in component:
				STATE["primary_component_nodes"] = set(component)
				return
	STATE["primary_component_nodes"] = set(largest_component)


def _find_graph_node_for_point(point: Tuple[float, float], prefer_primary: bool = True) -> Optional[Any]:
	"""Ищет узел графа, соответствующий заданной точке."""
	G = STATE.get("city_graph")
	if G is None or not point or len(point) != 2:
		return None
	if prefer_primary:
		primary_nodes = list(STATE.get("primary_component_nodes") or [])
		node = _nearest_node_from_candidates(G, tuple(point), primary_nodes)
		if node is not None:
			return node
	return _routing_service._find_nearest_node(G, tuple(point))


def _generate_city_stations(count: int = MIN_CITY_STATION_COUNT) -> List[Tuple[float, float]]:
	"""Выполняет служебную операцию: generate город станции."""
	G = STATE.get("city_graph")
	primary_nodes = list(STATE.get("primary_component_nodes") or [])
	if G is None or len(primary_nodes) < 5:
		return list(DEMO_STATIONS)
	candidates: List[Tuple[Any, Tuple[float, float]]] = []
	for node in primary_nodes:
		pos = G.nodes[node].get("pos")
		if isinstance(pos, (list, tuple)) and len(pos) == 2:
			candidates.append((node, (float(pos[0]), float(pos[1]))))
	if len(candidates) < 5:
		return list(DEMO_STATIONS)
	latitudes = [pos[0] for _, pos in candidates]
	longitudes = [pos[1] for _, pos in candidates]
	lat_min, lat_max = min(latitudes), max(latitudes)
	lon_min, lon_max = min(longitudes), max(longitudes)
	cols = 4
	rows = max(1, math.ceil(count / cols))
	selected: List[Tuple[float, float]] = []
	used_nodes = set()
	for row in range(rows):
		for col in range(cols):
			if len(selected) >= count:
				break
			target = (
				lat_min + (row + 0.5) * (lat_max - lat_min) / rows,
				lon_min + (col + 0.5) * (lon_max - lon_min) / cols,
			)
			available = [(node, pos) for node, pos in candidates if node not in used_nodes]
			if not available:
				break
			best = min(available, key=lambda item: haversine_m(target, item[1]))
			if selected and min(haversine_m(best[1], point) for point in selected) < 900.0:
				continue
			selected.append(best[1])
			used_nodes.add(best[0])
	if len(selected) < count:
		remaining = [(node, pos) for node, pos in candidates if node not in used_nodes]
		if not selected and remaining:
			selected.append(remaining[0][1])
			used_nodes.add(remaining[0][0])
			remaining = remaining[1:]
		while len(selected) < count and remaining:
			best_idx = 0
			best_score = -1.0
			for idx, (node, pos) in enumerate(remaining):
				score = min(haversine_m(pos, chosen) for chosen in selected) if selected else 0.0
				if score > best_score:
					best_score = score
					best_idx = idx
			node, pos = remaining.pop(best_idx)
			if selected and min(haversine_m(pos, chosen) for chosen in selected) < 700.0:
				continue
			selected.append(pos)
			used_nodes.add(node)
	return selected[:count] if selected else list(DEMO_STATIONS)


def _normalize_runtime_zone_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Нормализует runtime-состояние зону данные ответа."""
	zone = dict(payload or {})
	zone_type = (zone.get("zone_type") or "").strip().lower()
	if not zone_type:
		zone_type = "circle" if zone.get("center_lat") is not None or zone.get("center_lon") is not None else "rectangle"
	if zone_type == "circle":
		center_lat = float(zone.get("center_lat"))
		center_lon = float(zone.get("center_lon"))
		radius_m = max(10.0, float(zone.get("radius_m") or 50.0))
		return {
			"id": str(zone.get("id") or f"zone_{len(STATE['no_fly_zones'])+1}"),
			"zone_type": "circle",
			"center_lat": center_lat,
			"center_lon": center_lon,
			"radius_m": radius_m,
			"mission_id": zone.get("mission_id"),
		}
	lat_min = float(zone.get("lat_min"))
	lat_max = float(zone.get("lat_max"))
	lon_min = float(zone.get("lon_min"))
	lon_max = float(zone.get("lon_max"))
	return {
		"id": str(zone.get("id") or f"zone_{len(STATE['no_fly_zones'])+1}"),
		"zone_type": "rectangle",
		"lat_min": lat_min,
		"lat_max": lat_max,
		"lon_min": lon_min,
		"lon_max": lon_max,
		"mission_id": zone.get("mission_id"),
	}


def _zone_center(zone: Dict[str, Any]) -> Tuple[float, float]:
	"""Выполняет служебную операцию: зону центр."""
	if zone.get("zone_type") == "circle":
		return (float(zone.get("center_lat")), float(zone.get("center_lon")))
	lat_min = float(zone.get("lat_min"))
	lat_max = float(zone.get("lat_max"))
	lon_min = float(zone.get("lon_min"))
	lon_max = float(zone.get("lon_max"))
	return ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)


def _snap_point_to_graph(point: Tuple[float, float]) -> Tuple[float, float]:
	"""Привязывает точку к ближайшему узлу графа."""
	G = STATE.get("city_graph")
	if G is None or not point or len(point) != 2:
		return tuple(point)
	try:
		node = _find_graph_node_for_point(tuple(point), prefer_primary=True)
		if node is None:
			return tuple(point)
		pos = G.nodes[node].get("pos")
		if isinstance(pos, (list, tuple)) and len(pos) == 2:
			return (float(pos[0]), float(pos[1]))
	except Exception:
		logger.exception("Failed to snap point %s to graph", point)
	return tuple(point)


async def _ensure_demo_environment(city: str = DEMO_CITY):
	"""Проверяет и подготавливает демонстрационный сценарий демосреду."""
	target_city = city or DEMO_CITY
	if STATE.get("city_graph") is None or STATE.get("city") != target_city:
		city_data = _data_service.get_city_data(target_city)
		city_data["no_fly_zones"] = list(STATE["no_fly_zones"]) or city_data.get("no_fly_zones", [])
		city_graph = _graph_service.build_city_graph(city_data, "cargo")
		_routing_service.city_graphs[target_city] = city_graph
		STATE["city"] = target_city
		STATE["city_graph"] = city_graph
		STATE["drone_type"] = "cargo"
	if not STATE.get("base"):
		STATE["base"] = DEMO_BASE
	_refresh_primary_component_nodes()
	inventory = dict(STATE.get("inventory") or {})
	for drone_type, count in {"cargo": 4, "operator": 2, "cleaner": 1}.items():
		try:
			current = int(inventory.get(drone_type, 0))
		except Exception:
			current = 0
		if current < count:
			inventory[drone_type] = count
	STATE["inventory"] = inventory
	if not STATE.get("stations") or len(STATE.get("stations") or []) < MIN_CITY_STATION_COUNT:
		STATE["stations"] = _generate_city_stations(MIN_CITY_STATION_COUNT)
	if (not STATE.get("station_queues")) or len(STATE.get("station_queues") or {}) != len(STATE.get("stations") or []):
		STATE["station_queues"] = {
			str(i): {"charged_batteries": STATION_CHARGED_BATTERIES_MAX, "charging_queue": [], "queue": []}
			for i in range(len(STATE["stations"]))
		}
	refresh_charger_nodes()
	ensure_base_drones()
	await persist_state()


async def _enqueue_live_order(
	mission: Dict[str, Any],
	drone_type: str = "cargo",
	start_override: Optional[Tuple[float, float]] = None,
	end_override: Optional[Tuple[float, float]] = None,
) -> str:
	"""Выполняет служебную операцию: enqueue live-заказ."""
	order_id = f"ord_{len(STATE['orders'])+1}"
	start = mission.get("start_point") or {}
	end = mission.get("delivery_point") or {}
	start_point = tuple(start_override) if start_override is not None else (float(start.get("lat")), float(start.get("lon")))
	end_point = tuple(end_override) if end_override is not None else (float(end.get("lat")), float(end.get("lon")))
	entry = {
		"id": order_id,
		"type": "delivery",
		"drone_type": drone_type or mission.get("meta", {}).get("drone_type") or "cargo",
		"priority": 10,
		"start": start_point,
		"end": end_point,
		"battery_level": 100.0,
		"status": "queued",
		"waypoints": [],
		"mission_id": mission.get("id"),
		"mission_title": mission.get("title"),
	}
	STATE["orders"].append(entry)
	await persist_state()
	return order_id


async def _upsert_runtime_no_fly_zone(zone_payload: Dict[str, Any]) -> str:
	"""Создает или обновляет runtime-состояние запретную зону."""
	zone = _normalize_runtime_zone_payload(zone_payload)
	if zone.get("mission_id") and zone.get("zone_type") == "circle":
		_monitoring_service.upsert_mission_no_fly_zone(
			str(zone["mission_id"]),
			str(zone["id"]),
			(float(zone["center_lat"]), float(zone["center_lon"])),
			float(zone["radius_m"]),
			name=f"Запретная зона {len([z for z in STATE['no_fly_zones'] if z.get('mission_id') == zone.get('mission_id')]) + 1}",
		)
	replaced = False
	for idx, existing in enumerate(STATE["no_fly_zones"]):
		if existing.get("id") == zone["id"]:
			STATE["no_fly_zones"][idx] = zone
			replaced = True
			break
	if not replaced:
		STATE["no_fly_zones"].append(zone)
	STATE["zone_version"] += 1
	await rebuild_graph_with_zones()
	_reroute_active_drones_after_zone_change()
	await persist_state()
	return zone["id"]


async def _update_runtime_destination(mission: Dict[str, Any], destination: Tuple[float, float]):
	"""Обновляет runtime-состояние точку доставки."""
	order_id = mission.get("order_id")
	if not order_id:
		for order in STATE.get("orders", []):
			if order.get("mission_id") == mission.get("id"):
				order_id = order.get("id")
				break
	if not order_id:
		return None
	return await _apply_order_destination_update(order_id, tuple(destination))


async def _remove_runtime_zones_for_mission(mission_id: str) -> int:
	"""Удаляет runtime-зоны, связанные с миссией."""
	before = len(STATE["no_fly_zones"])
	STATE["no_fly_zones"] = [zone for zone in STATE.get("no_fly_zones", []) if zone.get("mission_id") != mission_id]
	removed = before - len(STATE["no_fly_zones"])
	if removed:
		STATE["zone_version"] += 1
		await rebuild_graph_with_zones()
		_reroute_active_drones_after_zone_change()
	return removed


async def _cancel_runtime_order(order_id: str) -> Optional[Dict[str, Any]]:
	"""Отменяет runtime-состояние заказ."""
	for order in STATE.get("orders", []):
		if order.get("id") != order_id:
			continue
		if order.get("status") in ("completed", "cancelled"):
			return order
		order["status"] = "cancelled"
		drone_id = order.get("drone_id")
		order.pop("drone_id", None)
		if drone_id and drone_id in STATE.get("drones", {}):
			drone = STATE["drones"][drone_id]
			_remove_drone_from_charge_queues(drone_id)
			drone["route"] = []
			drone["resume_route"] = []
			drone.pop("post_delivery_route", None)
			drone["target_idx"] = 0
			drone["waypoints_completed"] = 0
			drone["loaded_after_waypoint_count"] = 0
			drone.pop("saved_order_id", None)
			drone.pop("saved_route_for_charge", None)
			drone.pop("saved_target_idx", None)
			drone.pop("service_pause_ticks", None)
			drone.pop("service_pause_reason", None)
			drone.pop("service_resume_status", None)
			drone.pop("active_order_id", None)
			drone.pop("mission_mode", None)
			drone["force_charge_after_route"] = False
			if float(drone.get("battery", 100.0)) <= FLY_TO_CHARGER_AT_PCT:
				drone["status"] = "low_battery"
				maybe_route_to_base_or_station(drone)
			else:
				drone["status"] = "idle"
		await persist_state()
		return order
	return None


async def _cancel_runtime_mission(mission_id: str) -> Optional[Dict[str, Any]]:
	"""Отменяет runtime-состояние миссию."""
	cancelled_order = None
	for order in STATE.get("orders", []):
		if order.get("mission_id") == mission_id and order.get("status") not in ("completed", "cancelled"):
			cancelled_order = await _cancel_runtime_order(str(order.get("id")))
			break
	await _remove_runtime_zones_for_mission(mission_id)
	await persist_state()
	return cancelled_order


register_monitoring_routes(
	app,
	_monitoring_service,
	{
		"state": STATE,
		"ensure_demo_environment": _ensure_demo_environment,
		"enqueue_live_order": _enqueue_live_order,
		"upsert_runtime_no_fly_zone": _upsert_runtime_no_fly_zone,
		"update_runtime_destination": _update_runtime_destination,
		"cancel_runtime_mission": _cancel_runtime_mission,
		"snap_point_to_graph": _snap_point_to_graph,
	},
)

# Helpers
# Краткое описание погоды по коду WMO (Open-Meteo)
def _weather_code_to_desc(code: int) -> str:
	"""Преобразует код погоды в текстовое описание."""
	codes = {
		0: "Ясно", 1: "Преим. ясно", 2: "Переменная облачность", 3: "Пасмурно",
		45: "Туман", 48: "Изморозь", 51: "Морось", 53: "Морось", 55: "Морось",
		61: "Дождь", 63: "Дождь", 65: "Сильный дождь", 66: "Ледяной дождь", 67: "Ливень",
		71: "Снег", 73: "Снег", 75: "Снегопад", 77: "Снежные зёрна",
		80: "Ливень", 81: "Ливень", 82: "Ливень", 85: "Снег", 86: "Снегопад",
		95: "Гроза", 96: "Гроза с градом", 99: "Гроза с градом",
	}
	return codes.get(code, "—")


def _fetch_weather_sync(lat: float, lon: float) -> Optional[Dict[str, Any]]:
	"""Синхронный запрос погоды с Open-Meteo (бесплатный API, без ключа)."""
	try:
		import requests
		url = "https://api.open-meteo.com/v1/forecast"
		params = {
			"latitude": lat, "longitude": lon,
			"current": "relative_humidity_2m,weather_code,wind_speed_10m",
		}
		r = requests.get(url, params=params, timeout=5.0)
		r.raise_for_status()
		data = r.json()
		cur = data.get("current") or {}
		wind_kmh = float(cur.get("wind_speed_10m", 0) or 0)
		wind_mps = round(wind_kmh / 3.6, 1)
		humidity = int(cur.get("relative_humidity_2m", 0) or 0)
		code = int(cur.get("weather_code", 0) or 0)
		return {
			"wind_mps": max(0, min(40, wind_mps)),
			"humidity": max(0, min(100, humidity)),
			"weather_code": code,
			"description": _weather_code_to_desc(code),
		}
	except Exception as e:
		logger.warning("fetch_weather_from_api failed: %s", e)
		return None


async def fetch_weather_from_api(lat: float, lon: float) -> Optional[Dict[str, Any]]:
	"""Запрашивает погоду из api."""
	return await asyncio.to_thread(_fetch_weather_sync, lat, lon)


def _station_states_for_ui() -> List[Dict[str, Any]]:
	"""Список по индексам станций: {charged, charging} для отображения в UI."""
	stations = STATE.get("stations") or []
	sqs = STATE.get("station_queues") or {}
	result = []
	for i in range(len(stations)):
		sq = sqs.get(str(i), {})
		if isinstance(sq, dict) and "charged_batteries" in sq:
			result.append({
				"charged": int(sq.get("charged_batteries", 0)),
				"charging": len(sq.get("charging_queue", [])),
			})
		else:
			result.append({"charged": STATION_CHARGED_BATTERIES_MAX, "charging": 0})
	return result


def _public_order_snapshot(order: Dict[str, Any]) -> Dict[str, Any]:
	"""Формирует публичный снимок состояния заказа."""
	snapshot = dict(order or {})
	snapshot.pop("segments", None)
	snapshot.pop("battery_plan", None)
	return snapshot


def _public_drone_snapshot(drone: Dict[str, Any]) -> Dict[str, Any]:
	"""Формирует публичный снимок состояния дрона."""
	snapshot = dict(drone or {})
	snapshot.pop("history", None)
	return snapshot


def _public_histories() -> Dict[str, List[Tuple[float, float]]]:
	"""Формирует публичное представление истории."""
	result: Dict[str, List[Tuple[float, float]]] = {}
	for drone_id, drone in (STATE.get("drones") or {}).items():
		history = list(drone.get("history") or [])
		result[drone_id] = history[-PUBLIC_HISTORY_LIMIT:]
	return result


def _public_state_payload() -> Dict[str, Any]:
	"""Формирует публичное представление состояние данные ответа."""
	return {
		"city": STATE["city"],
		"orders": [_public_order_snapshot(order) for order in STATE.get("orders", [])],
		"drones": {drone_id: _public_drone_snapshot(drone) for drone_id, drone in (STATE.get("drones") or {}).items()},
		"histories": _public_histories(),
		"no_fly_zones": STATE["no_fly_zones"],
		"stations": STATE.get("stations", []),
		"station_states": _station_states_for_ui(),
		"weather": STATE.get("weather", {}),
		"battery_mode": STATE.get("battery_mode", "reality"),
	}


def _queue_live_telemetry(drone_id: str, drone: Dict[str, Any]) -> None:
	"""Ставит в очередь live-телеметрию."""
	global LIVE_TELEMETRY_QUEUE, LIVE_TELEMETRY_LAST_SENT
	queue = LIVE_TELEMETRY_QUEUE
	if queue is None:
		return
	snapshot = dict(drone or {})
	history = list(snapshot.get("history") or [])
	if history:
		snapshot["history"] = history[-PUBLIC_HISTORY_LIMIT:]
	now = time.time()
	pos = tuple(snapshot.get("pos") or ())
	status = str(snapshot.get("status") or "")
	battery = float(snapshot.get("battery", 0.0) or 0.0)
	last = LIVE_TELEMETRY_LAST_SENT.get(drone_id) or {}
	last_pos = tuple(last.get("pos") or ())
	moved_m = haversine_m(last_pos, pos) if len(pos) == 2 and len(last_pos) == 2 else float("inf")
	if last:
		if (
			status == last.get("status")
			and abs(battery - float(last.get("battery", battery))) < 3.0
			and moved_m < 18.0
			and (now - float(last.get("ts", 0.0))) < 2.0
		):
			return
	LIVE_TELEMETRY_LAST_SENT[drone_id] = {"ts": now, "pos": pos, "status": status, "battery": battery}
	try:
		queue.put_nowait((drone_id, snapshot))
	except asyncio.QueueFull:
		try:
			_ = queue.get_nowait()
		except asyncio.QueueEmpty:
			return
		try:
			queue.put_nowait((drone_id, snapshot))
		except asyncio.QueueFull:
			pass


async def telemetry_writer_loop():
	"""Фоново забирает live-телеметрию из очереди и сохраняет ее в базу данных мониторинга."""
	global LIVE_TELEMETRY_QUEUE
	while True:
		queue = LIVE_TELEMETRY_QUEUE
		if queue is None:
			await asyncio.sleep(0.2)
			continue
		drone_id, snapshot = await queue.get()
		try:
			await asyncio.to_thread(_monitoring_service.record_live_telemetry, drone_id, snapshot)
		except Exception:
			logger.exception("telemetry_writer_loop error")
		finally:
			queue.task_done()


async def broadcast_state():
	"""Отправляет всем WebSocket-клиентам актуальный снимок состояния симуляции."""
	payload = _public_state_payload()
	dead: List[WebSocket] = []
	for ws in list(STATE["clients"]):
		try:
			await ws.send_json(payload)
		except Exception:
			dead.append(ws)
	for ws in dead:
		STATE["clients"].discard(ws)

async def broadcaster_loop():
	"""Периодически запускает рассылку состояния активным клиентам интерфейса."""
	while True:
		await broadcast_state()
		await asyncio.sleep(1.0)

async def scheduler_loop():
	# very simple loop: assign queued orders, move drones, reroute on zones
	"""Выполняет основной цикл симуляции: двигает дроны, назначает заказы и обновляет зарядку."""
	while True:
		try:
			simulate_step()
			await asyncio.to_thread(assign_orders)
		except Exception:
			logger.exception("scheduler_loop error")
		# periodic save
		try:
			await persist_state()
		except Exception:
			logger.exception("persist_state error")
		await asyncio.sleep(1.0)

@app.on_event("startup")
async def on_startup():
	"""Готовит сервисы при запуске FastAPI-приложения и поднимает фоновые задачи."""
	global LIVE_TELEMETRY_QUEUE
	_monitoring_service.init_db()
	LIVE_TELEMETRY_QUEUE = asyncio.Queue(maxsize=LIVE_TELEMETRY_QUEUE_MAX)
	await restore_state()
	# Попытка автозагрузки погоды на базе, чтобы UI не показывал только "ветер=по умолчанию".
	try:
		w = STATE.get("weather") or {}
		base = STATE.get("base")
		needs_weather = (w.get("description") is None) or (w.get("humidity") is None)
		if needs_weather and base and isinstance(base, (list, tuple)) and len(base) == 2:
			lat, lon = float(base[0]), float(base[1])
			data = await fetch_weather_from_api(lat, lon)
			if data:
				STATE["weather"] = data
				await persist_state()
	except Exception:
		logger.exception("auto weather fetch failed")
	# initialize drones at base from inventory if needed
	try:
		ensure_base_drones()
	except Exception:
		logger.exception("ensure_base_drones on startup failed")
	try:
		active_order_ids = {
			str(order.get("id"))
			for order in STATE.get("orders", [])
			if order.get("status") not in ("completed", "cancelled", "failed")
		}
		await asyncio.to_thread(_monitoring_service.reconcile_runtime_assignments, active_order_ids)
	except Exception:
		logger.exception("reconcile_runtime_assignments failed")
	asyncio.create_task(broadcaster_loop())
	asyncio.create_task(telemetry_writer_loop())
	asyncio.create_task(scheduler_loop())

@app.on_event("shutdown")
async def on_shutdown():
	"""Останавливает фоновые задачи и сохраняет runtime-состояние перед завершением сервера."""
	if LIVE_TELEMETRY_QUEUE is not None:
		await LIVE_TELEMETRY_QUEUE.join()
	await persist_state()

# API endpoints
@app.post("/api/load_city")
async def load_city(body: LoadCityRequest):
	"""Загружает город."""
	try:
		city_data = _data_service.get_city_data(body.city)
		# inject current API no-fly zones into data prior to build
		city_data['no_fly_zones'] = list(STATE["no_fly_zones"]) or city_data.get('no_fly_zones', [])
		city_graph = _graph_service.build_city_graph(city_data, body.drone_type)
		_routing_service.city_graphs[body.city] = city_graph
		STATE["city"] = body.city
		STATE["city_graph"] = city_graph
		STATE["drone_type"] = body.drone_type
		_refresh_primary_component_nodes()
		if not STATE.get("stations") or len(STATE.get("stations") or []) < MIN_CITY_STATION_COUNT:
			STATE["stations"] = _generate_city_stations(MIN_CITY_STATION_COUNT)
			STATE["station_queues"] = {
				str(i): {"charged_batteries": STATION_CHARGED_BATTERIES_MAX, "charging_queue": [], "queue": []}
				for i in range(len(STATE["stations"]))
			}
		refresh_charger_nodes()
		await persist_state()
		return {"ok": True, "stats": {
			"nodes": len(city_graph.nodes),
			"edges": len(city_graph.edges)
		}}
	except Exception as e:
		logger.exception("load_city failed")
		return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.get("/api/state")
async def get_state():
	"""Получает состояние."""
	payload = _public_state_payload()
	payload.update({
		"orders_count": len(STATE["orders"]),
		"drones_count": len(STATE["drones"]),
		"base": STATE["base"],
		"inventory": STATE["inventory"],
	})
	return payload

@app.post("/api/reverse")
async def reverse_geocode(body: Dict[str, float]):
	"""Выполняет обратное геокодирование координат."""
	lat = body.get("lat") ; lon = body.get("lon")
	if lat is None or lon is None:
		return JSONResponse(status_code=400, content={"ok": False, "error": "lat/lon required"})
	addr = _data_service.coords_to_address((float(lat), float(lon)), language='ru')
	return {"ok": True, "address": addr}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
	"""Обслуживает WebSocket-подключение клиента."""
	await ws.accept()
	STATE["clients"].add(ws)
	try:
		while True:
			# keep alive / receive optional messages later
			_ = await ws.receive_text()
	except WebSocketDisconnect:
		pass
	finally:
		STATE["clients"].discard(ws)

@app.post("/api/no_fly_zones")
async def add_no_fly_zone(zone: NoFlyZone):
	"""Добавляет запретную зону."""
	z = zone.model_dump()
	await _upsert_runtime_no_fly_zone(z)
	return {"ok": True, "zone": _normalize_runtime_zone_payload(z)}

@app.get("/api/no_fly_zones")
async def list_no_fly_zones():
	"""Возвращает список запретные зоны."""
	return STATE["no_fly_zones"]

@app.delete("/api/no_fly_zones/{zone_id}")
async def remove_no_fly_zone(zone_id: str):
	"""Удаляет запретную зону."""
	zone_to_remove = next((z for z in STATE["no_fly_zones"] if z.get("id") == zone_id), None)
	before = len(STATE["no_fly_zones"])
	STATE["no_fly_zones"] = [z for z in STATE["no_fly_zones"] if z.get("id") != zone_id]
	removed = before - len(STATE["no_fly_zones"])
	if removed:
		if zone_to_remove and zone_to_remove.get("mission_id"):
			_monitoring_service.remove_mission_zone(zone_id)
		STATE["zone_version"] += 1
		await rebuild_graph_with_zones()
		_reroute_active_drones_after_zone_change()
		await persist_state()
	return {"ok": True, "removed": removed }

def _centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
	"""Вычисляет центр набора точек."""
	if not points:
		return (0.0, 0.0)
	n = len(points)
	return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


@app.post("/api/orders")
async def add_order(order: AddOrderRequest):
	# Classify order type
	"""Добавляет заказ."""
	order_type = classify_order(order)
	req_drone_type = (order.drone_type or "").strip().lower() or map_order_to_drone_type(order_type)
	if req_drone_type not in ("cargo", "operator", "cleaner"):
		req_drone_type = map_order_to_drone_type(order_type)
	start = await resolve_point(order.coords_from, order.address_from)
	end = await resolve_point(order.coords_to, order.address_to)
	# Операторский/сервисный: точка назначения может быть без «откуда» — старт с базы
	if not start and req_drone_type in ("operator", "cleaner"):
		base = STATE.get("base")
		if isinstance(base, (list, tuple)) and len(base) == 2:
			start = tuple(base)
	if not end and getattr(order, "area_polygon", None):
		pts = [tuple(p) for p in order.area_polygon if isinstance(p, (list, tuple)) and len(p) == 2]
		if pts:
			end = _centroid(pts)
	if not start and end and req_drone_type in ("operator", "cleaner"):
		start = end
	if not start or not end:
		return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid start or end"})
	order_id = f"ord_{len(STATE['orders'])+1}"
	priority = int(getattr(order, "priority", 5))
	priority = max(0, min(10, priority))
	entry = {
		"id": order_id,
		"type": order_type,
		"drone_type": req_drone_type,
		"priority": priority,
		"start": start,
		"end": end,
		"battery_level": max(1.0, min(100.0, float(order.battery_level))),
		"status": "queued",
		"waypoints": [tuple(wp) for wp in (order.waypoints or []) if isinstance(wp, (list, tuple)) and len(wp)==2],
	}
	if getattr(order, "area_polygon", None):
		entry["area_polygon"] = [tuple(p) for p in order.area_polygon if isinstance(p, (list, tuple)) and len(p)==2]
	STATE["orders"].append(entry)
	await persist_state()
	return {"ok": True, "order": entry, "queue_size": len(STATE["orders"]) }

@app.get("/api/orders")
async def list_orders():
	"""Возвращает список заказы."""
	return STATE["orders"]

@app.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Отменяет заказ."""
    order = await _cancel_runtime_order(order_id)
    if order is not None:
        return {"ok": True, "order": order}
    return JSONResponse(status_code=404, content={"ok": False, "error": "Order not found or not cancellable"})

class UpdateOrderRequest(BaseModel):
    new_address_to: Optional[str] = None
    new_coords_to: Optional[List[float]] = None


async def _apply_order_destination_update(order_id: str, new_end: Tuple[float, float]):
	"""Применяет изменение точки доставки к заказу."""
	for o in STATE["orders"]:
		if o.get("id") == order_id:
			o["end"] = tuple(new_end)
			_reset_order_plan_backoff(o)
			mission_id = o.get("mission_id")
			if mission_id:
				_monitoring_service.update_mission_destination(mission_id, tuple(new_end))
			did = o.get("drone_id")
			if did and did in STATE["drones"]:
				d = STATE["drones"][did]
				if d.get("type") == "cargo":
					res = _plan_active_cargo_order_from_drone(d, o, destination=tuple(new_end))
					coords = res.get("coords") if res and res.get("ok") else None
				else:
					_, coords, _ = plan_route_for(d.get("pos", o["start"]), tuple(new_end), d["type"], d["battery"])
				if coords:
					apply_midroute_charging(d, coords)
			await persist_state()
			return {"ok": True, "order": o}
	return JSONResponse(status_code=404, content={"ok": False, "error": "Order not found"})


@app.post("/api/orders/{order_id}/update_destination")
async def update_order_destination(order_id: str, body: UpdateOrderRequest):
    """Обновляет заказ точку доставки."""
    new_end = await resolve_point(body.new_coords_to, body.new_address_to)
    if not new_end:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid destination"})
    return await _apply_order_destination_update(order_id, tuple(new_end))

@app.get("/")
async def root_page():
	"""Возвращает главную HTML-страницу приложения."""
	try:
		with open("static/index.html", "r", encoding="utf-8") as f:
			return HTMLResponse(f.read())
	except Exception:
		return HTMLResponse("<h3>Client not found. Ensure static/index.html exists.</h3>", status_code=404)

@app.post("/api/weather")
async def set_weather(w: Dict[str, Any]):
	"""Устанавливает погоду."""
	wind = float(w.get("wind_mps", 3.0))
	STATE["weather"] = {
		"wind_mps": max(0.0, min(40.0, wind)),
		"humidity": STATE["weather"].get("humidity"),
		"description": STATE["weather"].get("description"),
		"weather_code": STATE["weather"].get("weather_code"),
	}
	await persist_state()
	return {"ok": True, "weather": STATE["weather"]}


@app.get("/api/weather/fetch")
async def fetch_weather(lat: Optional[float] = None, lon: Optional[float] = None):
	"""Загрузить погоду из Open-Meteo по координатам (или по базе). При недоступности API — 200, текущее состояние."""
	if lat is None or lon is None:
		base = STATE.get("base")
		if base and len(base) == 2:
			lat, lon = float(base[0]), float(base[1])
		else:
			lat, lon = 48.7080, 44.5133
	data = await fetch_weather_from_api(lat, lon)
	if data:
		STATE["weather"] = data
		await persist_state()
		return {"ok": True, "fetched": True, "weather": STATE["weather"]}
	# API недоступен — не ломаем UI: 200, текущая погода, fetched: False
	return {
		"ok": True,
		"fetched": False,
		"weather": STATE.get("weather", {"wind_mps": 3.0}),
		"error": "Weather API unavailable",
	}


@app.get("/api/battery_mode")
async def get_battery_mode():
	"""Текущий режим расхода: reality — реальные данные, test — укороченная дальность для тестов."""
	return {"mode": STATE.get("battery_mode", "reality")}


@app.post("/api/battery_mode")
async def set_battery_mode(body: Dict[str, str]):
	"""Переключить режим: {"mode": "reality"} или {"mode": "test"}."""
	mode = (body.get("mode") or "").strip().lower()
	if mode not in ("reality", "test"):
		return JSONResponse(status_code=400, content={"ok": False, "error": "mode must be 'reality' or 'test'"})
	STATE["battery_mode"] = mode
	_routing_service.set_battery_mode(mode)
	# При переключении в «Тест» сбрасываем назначенные заказы, чтобы маршруты пересчитались через зарядки
	if mode == "test":
		for o in STATE.get("orders", []):
			if o.get("status") == "assigned":
				did = o.get("drone_id")
				if did and did in STATE.get("drones", {}):
					d = STATE["drones"][did]
					d["route"] = []
					d["target_idx"] = 0
					d["status"] = "idle"
				o["status"] = "queued"
				o.pop("drone_id", None)
		logger.info("battery_mode=test: assigned orders reset to queued for replan with chargers")
	return {"ok": True, "mode": mode}


# Base and inventory management
class BaseConfig(BaseModel):
    base: Optional[Tuple[float, float]] = None
    inventory: Optional[Dict[str, int]] = None  # type->count

@app.post("/api/base")
async def set_base(cfg: BaseConfig):
    """Устанавливает базу."""
    if cfg.base:
        STATE["base"] = tuple(cfg.base)
        # Update weather by base coordinates (best effort).
        try:
            lat, lon = float(STATE["base"][0]), float(STATE["base"][1])
            data = await fetch_weather_from_api(lat, lon)
            if data:
                STATE["weather"] = data
        except Exception:
            logger.exception("fetch weather on set_base failed")
    if cfg.inventory is not None:
        # sanitize counts
        inv: Dict[str, int] = {}
        for k, v in (cfg.inventory or {}).items():
            try:
                inv[str(k)] = max(0, int(v))
            except Exception:
                continue
        STATE["inventory"] = inv
    refresh_charger_nodes()
    ensure_base_drones()
    await persist_state()
    return {"ok": True, "base": STATE["base"], "inventory": STATE["inventory"]}

@app.get("/api/base")
async def get_base():
    """Получает базу."""
    return {"base": STATE["base"], "inventory": STATE["inventory"], "stations": STATE["stations"]}

class StationsConfig(BaseModel):
    stations: List[Tuple[float, float]]
    capacity: Optional[int] = 2

@app.post("/api/stations")
async def set_stations(cfg: StationsConfig):
    """Устанавливает станции."""
    try:
        stations = []
        for s in cfg.stations:
            if isinstance(s, (list, tuple)) and len(s) == 2:
                stations.append((float(s[0]), float(s[1])))
        STATE["stations"] = stations
        # Станции: смена аккумулятора, запас 20 заряженных; очередь на зарядку
        STATE["station_queues"] = {
            str(i): {
                "charged_batteries": STATION_CHARGED_BATTERIES_MAX,
                "charging_queue": [],  # список тиков до готовности каждого аккумулятора
                "queue": [],  # drone_id, ожидающие заряженный аккумулятор
            }
            for i in range(len(stations))
        }
        refresh_charger_nodes()
        await persist_state()
        return {"ok": True, "stations": stations}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

def refresh_charger_nodes() -> None:
	"""Привязывает базу и станции к ближайшим узлам графа после загрузки города или изменения настроек."""
	G = STATE.get("city_graph")
	if G is None:
		STATE["charger_nodes"] = {"base": None, "stations": []}
		return
	_refresh_primary_component_nodes()
	cn = {"base": None, "stations": []}
	base = STATE.get("base")
	if base and isinstance(base, (list, tuple)) and len(base) == 2:
		cn["base"] = _find_graph_node_for_point(tuple(base), prefer_primary=True)
	stations = STATE.get("stations") or []
	cn["stations"] = []
	for s in stations:
		if isinstance(s, (list, tuple)) and len(s) == 2:
			node = _find_graph_node_for_point(tuple(s), prefer_primary=True)
			cn["stations"].append(node)
		else:
			cn["stations"].append(None)
	STATE["charger_nodes"] = cn
	logger.debug("charger_nodes refreshed: base=%s stations=%s", cn["base"], cn["stations"])


# Utilities
async def resolve_point(coords: Optional[List[float]], address: Optional[str]):
	"""Определяет точку."""
	if coords and len(coords) == 2:
		return tuple(coords)
	if address:
		try:
			city = STATE["city"]
			return _data_service.address_to_coords(address, city)
		except Exception:
			return None
	return None

def classify_order(order: AddOrderRequest) -> str:
	"""Определяет тип заказ."""
	if order.type_hint in ("delivery", "shooting", "work"):
		return order.type_hint
	text = " ".join(filter(None, [order.address_from, order.address_to]))
	text = text.lower() if text else ""
	if any(k in text for k in ["достав", "deliver", "посыл", "parcel"]):
		return "delivery"
	if any(k in text for k in ["съём", "съем", "photo", "video", "aerial"]):
		return "shooting"
	return "work"

def _route_length(coords: List[Tuple[float, float]]) -> float:
	"""Приблизительная длина маршрута в метрах по списку координат."""
	if not coords or len(coords) < 2:
		return 0.0
	return sum(haversine_m(tuple(coords[i]), tuple(coords[i + 1])) for i in range(len(coords) - 1))


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
	"""Проверка: точка (lat, lon) внутри полигона (ray casting)."""
	if not polygon or len(polygon) < 3:
		return False
	lat, lon = point
	n = len(polygon)
	inside = False
	j = n - 1
	for i in range(n):
		lat_i, lon_i = polygon[i]
		lat_j, lon_j = polygon[j]
		if ((lon_i > lon) != (lon_j > lon)) and (lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i + 1e-20) + lat_i):
			inside = not inside
		j = i
	return inside


# Базовый шаг сетки облёта внутри зоны (м). Для городского осмотра держим плотнее.
OPERATOR_AREA_GRID_STEP_M = 30.0


def _lawnmower_waypoints_inside_polygon(polygon: List[Tuple[float, float]], step_m: float = OPERATOR_AREA_GRID_STEP_M) -> List[Tuple[float, float]]:
	"""
	Генерирует точки облёта внутри полигона (лаунмower): сетка с шагом step_m,
	только точки внутри полигона, порядок «змейкой» по строкам (lat).
	"""
	if not polygon or len(polygon) < 3:
		return []
	lats = [p[0] for p in polygon]
	lons = [p[1] for p in polygon]
	lat_min, lat_max = min(lats), max(lats)
	lon_min, lon_max = min(lons), max(lons)
	# Приближение: 1° широты ≈ 111 км, 1° долготы ≈ 111*cos(lat) км
	lat_mid = (lat_min + lat_max) / 2.0
	deg_per_m_lat = 1.0 / (111_000.0)
	deg_per_m_lon = 1.0 / (111_000.0 * max(0.01, math.cos(math.radians(lat_mid))))
	step_lat = step_m * deg_per_m_lat
	step_lon = step_m * deg_per_m_lon
	if step_lat <= 0 or step_lon <= 0:
		return [tuple(p) for p in polygon]
	# Сетка по широте (ряды), по долготе (столбцы)
	rows = []
	lat = lat_min
	while lat <= lat_max:
		row = []
		lon = lon_min
		while lon <= lon_max:
			pt = (lat, lon)
			if _point_in_polygon(pt, polygon):
				row.append(pt)
			lon += step_lon
		if row:
			rows.append(row)
		lat += step_lat
	# Змейка: чётные ряды по возрастанию lon, нечётные — по убыванию
	out = []
	for i, row in enumerate(rows):
		if i % 2 == 1:
			row = list(reversed(row))
		out.extend(row)
	return out


def _polygon_bbox_dims_m(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
	"""Вычисляет размеры ограничивающего прямоугольника полигона в метрах."""
	if not polygon:
		return (0.0, 0.0)
	lats = [p[0] for p in polygon]
	lons = [p[1] for p in polygon]
	lat_mid = (min(lats) + max(lats)) / 2.0
	h_m = abs(max(lats) - min(lats)) * 111_000.0
	w_m = abs(max(lons) - min(lons)) * 111_000.0 * max(0.01, math.cos(math.radians(lat_mid)))
	return (w_m, h_m)


def _operator_area_grid_step_m(polygon: List[Tuple[float, float]]) -> float:
	"""Адаптивный шаг змейки: целимся в 20..40 м в зависимости от размера зоны."""
	w_m, h_m = _polygon_bbox_dims_m(polygon)
	min_dim = max(1.0, min(w_m, h_m))
	adaptive = min_dim / 8.0
	return float(max(20.0, min(40.0, adaptive)))


def _operator_area_waypoints(order: Dict[str, Any]) -> List[Tuple[float, float]]:
	"""Строит точки облета внутри выделенной области по схеме регулярного прохода, а не только по контуру."""
	# Для continuation-задач используем уже оставшиеся внутренние точки,
	# чтобы не генерировать всю сетку заново и не терять прогресс.
	if order.get("remaining_waypoints"):
		return [
			tuple(p) for p in (order.get("remaining_waypoints") or [])
			if isinstance(p, (list, tuple)) and len(p) == 2
		]
	if order.get("rest_waypoints"):
		return [
			tuple(p) for p in (order.get("rest_waypoints") or [])
			if isinstance(p, (list, tuple)) and len(p) == 2
		]
	wp = []
	polygon = []
	if order.get("area_polygon"):
		polygon = [tuple(p) for p in order["area_polygon"] if isinstance(p, (list, tuple)) and len(p) == 2]
	if not polygon and order.get("start") and order.get("end"):
		s, e = order["start"], order["end"]
		lat_min, lat_max = min(s[0], e[0]), max(s[0], e[0])
		lon_min, lon_max = min(s[1], e[1]), max(s[1], e[1])
		polygon = [(lat_min, lon_min), (lat_max, lon_min), (lat_max, lon_max), (lat_min, lon_max)]
	if len(polygon) >= 3:
		step_m = _operator_area_grid_step_m(polygon)
		wp = _lawnmower_waypoints_inside_polygon(polygon, step_m)
	if not wp:
		# Fallback: только вершины контура (как раньше)
		wp = list(polygon) if polygon else []
	return wp


def _operator_area_root_order_id(order: Dict[str, Any]) -> str:
	"""Возвращает идентификатор корневого заказа для цепочки работ операторского дрона."""
	return str(order.get("root_order_id") or order.get("id") or "")


def _segment_intersects_no_fly_zone(a: Tuple[float, float], b: Tuple[float, float], samples: int = 12) -> bool:
	"""Проверяет прямой сегмент на пересечение запретных зон дискретной аппроксимацией."""
	if is_point_in_any_zone(a) or is_point_in_any_zone(b):
		return True
	for i in range(1, max(2, samples)):
		t = i / float(samples)
		p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
		if is_point_in_any_zone(p):
			return True
	return False


def _build_operator_area_flight_path(start_pos: Tuple[float, float], polygon: List[Tuple[float, float]], waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
	"""
	Маршрут осмотра operator-area в координатах полёта (без road graph):
	точки только внутри полигона; сегменты, пересекающие no-fly, пропускаются локально.
	"""
	if not waypoints:
		return []
	coords: List[Tuple[float, float]] = []
	prev = tuple(start_pos)
	for wp in waypoints:
		wp = tuple(wp)
		if polygon and not _point_in_polygon(wp, polygon):
			continue
		if _segment_intersects_no_fly_zone(prev, wp):
			# Локально пропускаем недопустимую точку, не переводя всю миссию на дорожный граф.
			continue
		if not coords or coords[-1] != wp:
			coords.append(wp)
			prev = wp
	return coords


def _can_operator_continue_with_reserve(battery_pct: float, segment_m: float, drone_type: str = "operator") -> bool:
	"""Проверка, что после следующего сегмента у дрона останется минимум 20% батареи."""
	after = _routing_service.compute_battery_after(segment_m, battery_pct, MODE_EMPTY, drone_type)
	return after >= 20.0


def _split_operator_area_by_battery(start_pos: Tuple[float, float], battery_pct: float, area_path: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]:
	"""
	Делит area-path на доступную сейчас часть и остаток по правилу резерва 20%.
	Возвращает (current_part, remaining_part, battery_after_current_part).
	"""
	current: List[Tuple[float, float]] = []
	if not area_path:
		return (current, [], battery_pct)
	prev = tuple(start_pos)
	bat = float(battery_pct)
	for i, wp in enumerate(area_path):
		dist_m = haversine_m(prev, tuple(wp))
		if not _can_operator_continue_with_reserve(bat, dist_m, "operator"):
			return (current, [tuple(x) for x in area_path[i:]], bat)
		bat = _routing_service.compute_battery_after(dist_m, bat, MODE_EMPTY, "operator")
		current.append(tuple(wp))
		prev = tuple(wp)
	return (current, [], bat)


def _build_operator_area_route(
	drone_pos: Tuple[float, float],
	battery_pct: float,
	polygon: List[Tuple[float, float]],
	coverage_waypoints: List[Tuple[float, float]],
	drone_type: str = "operator",
) -> Dict[str, Any]:
	"""
	Смешанный маршрут operator-area:
	A) approach по graph до входа в зону,
	B) coverage free-flight внутри полигона,
	C) exit по graph до зарядки.
	"""
	if not coverage_waypoints:
		return {"ok": False, "reason": "NO_PATH", "details": "empty coverage"}
	area_path = _build_operator_area_flight_path(drone_pos, polygon, coverage_waypoints)
	if not area_path:
		return {"ok": False, "reason": "NO_PATH", "details": "no valid coverage path"}

	entry = tuple(area_path[0])
	_, approach_coords_raw, approach_len = plan_route_for(tuple(drone_pos), entry, drone_type, battery_pct, reserve_pct=PLAN_RESERVE_PCT)
	approach_coords = _to_coord_list(approach_coords_raw or [tuple(drone_pos)])
	if not approach_coords:
		approach_coords = [tuple(drone_pos)]
	approach_last = tuple(approach_coords[-1])
	approach_bat = _routing_service.compute_battery_after(approach_len, battery_pct, MODE_EMPTY, drone_type)

	current_cov, remaining_cov, bat_after_cov = _split_operator_area_by_battery(approach_last, approach_bat, area_path)
	if not current_cov:
		return {
			"ok": True,
			"approach_coords": approach_coords,
			"coverage_coords": [],
			"exit_coords": [],
			"combined_coords": approach_coords,
			"approach_length": float(approach_len),
			"coverage_length": 0.0,
			"exit_length": 0.0,
			"remaining_waypoints": area_path,
			"handover_point": approach_last,
			"coverage_start_idx": len(approach_coords),
			"coverage_end_idx": len(approach_coords),
		}

	coverage_len = _route_length([approach_last] + current_cov)
	last_cov = tuple(current_cov[-1])
	exit_coords, exit_len = _best_escape_graph_path(last_cov, bat_after_cov, drone_type)
	combined = _concat_coords(approach_coords, current_cov)
	combined = _concat_coords(combined, exit_coords)
	coverage_start_idx = max(0, len(approach_coords) - 1)
	coverage_end_idx = coverage_start_idx + len(current_cov)
	return {
		"ok": True,
		"approach_coords": approach_coords,
		"coverage_coords": current_cov,
		"exit_coords": exit_coords,
		"combined_coords": combined,
		"approach_length": float(approach_len),
		"coverage_length": float(coverage_len),
		"exit_length": float(exit_len),
		"remaining_waypoints": remaining_cov,
		"handover_point": last_cov,
		"coverage_start_idx": coverage_start_idx,
		"coverage_end_idx": coverage_end_idx,
	}


def _advance_area_order_progress(order: Dict[str, Any], flown_points: List[Tuple[float, float]]) -> None:
	"""Обновляет прогресс area-миссии: completed/remaining для текущего заказа."""
	remaining = [tuple(w) for w in (order.get("remaining_waypoints") or _operator_area_waypoints(order))]
	if not remaining:
		return
	n = min(len(flown_points), len(remaining))
	order["completed_waypoints_count"] = int(order.get("completed_waypoints_count", 0)) + n
	order["remaining_waypoints"] = [tuple(w) for w in remaining[n:]]


def _to_coord_list(coords: List[Any]) -> List[Tuple[float, float]]:
	"""Преобразует список координат к единому формату."""
	out: List[Tuple[float, float]] = []
	for p in coords or []:
		if isinstance(p, (list, tuple)) and len(p) == 2:
			out.append((float(p[0]), float(p[1])))
	return out


def _best_escape_graph_path(start: Tuple[float, float], battery_pct: float, drone_type: str = "operator") -> Tuple[List[Tuple[float, float]], float]:
	"""Строит путь по графу до ближайшей точки зарядки: базы или станции."""
	chargers: List[Tuple[float, float]] = []
	if STATE.get("base"):
		chargers.append(tuple(STATE["base"]))
	chargers += [tuple(s) for s in STATE.get("stations", []) if isinstance(s, (list, tuple)) and len(s) == 2]
	best_coords: List[Tuple[float, float]] = []
	best_len = float("inf")
	for c in chargers:
		_, coords, length = plan_route_for(tuple(start), tuple(c), drone_type, battery_pct, reserve_pct=PLAN_RESERVE_PCT)
		if coords and length < best_len:
			best_len = float(length)
			best_coords = _to_coord_list(coords)
	if best_coords:
		return (best_coords, best_len)
	return ([], 0.0)


def _concat_coords(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
	"""Объединяет координаты."""
	if not a:
		return list(b or [])
	if not b:
		return list(a or [])
	if a[-1] == b[0]:
		return list(a) + list(b[1:])
	return list(a) + list(b)


def _get_active_order_for_drone(drone_id: str) -> Optional[Dict[str, Any]]:
	"""Получает активный заказ, назначенный дрону."""
	for o in STATE.get("orders", []):
		if o.get("drone_id") == drone_id and o.get("status") in ("assigned", "in_progress", "waiting_continuation"):
			return o
	return None


def _is_near_point(a: Tuple[float, float], b: Tuple[float, float], threshold_m: float = MISSION_POINT_NEAR_METERS) -> bool:
	"""Проверяет близость двух точек с заданным порогом."""
	try:
		return haversine_m(tuple(a), tuple(b)) <= float(threshold_m)
	except Exception:
		return False


def _begin_service_stop(drone: Dict[str, Any], reason: str, ticks: int = SERVICE_STOP_TICKS, resume_status: str = "enroute") -> None:
	"""Запускает сервисную остановку дрона в точке забора или выдачи."""
	drone["service_pause_ticks"] = max(1, int(ticks))
	drone["service_pause_reason"] = str(reason)
	drone["service_resume_status"] = str(resume_status)
	drone["status"] = "service_stop"
	drone["speed_mps"] = 0.0


def _process_service_stop(drone_id: str, drone: Dict[str, Any]) -> bool:
	"""Обрабатывает оставшееся время сервисной остановки."""
	if drone.get("status") != "service_stop":
		return False
	ticks_left = int(drone.get("service_pause_ticks", 0) or 0)
	if ticks_left <= 0:
		drone.pop("service_pause_ticks", None)
		drone.pop("service_pause_reason", None)
		drone.pop("service_resume_status", None)
		drone["status"] = "idle"
		return True
	drone["speed_mps"] = 0.0
	ticks_left -= 1
	drone["service_pause_ticks"] = ticks_left
	if ticks_left > 0:
		return True
	reason = str(drone.pop("service_pause_reason", "") or "")
	resume_status = str(drone.pop("service_resume_status", "enroute") or "enroute")
	drone.pop("service_pause_ticks", None)
	if reason == "delivery":
		mark_order_completed_if_any(drone_id)
		post_route = list(drone.pop("post_delivery_route", []) or [])
		if post_route:
			drone["route"] = post_route
			drone["target_idx"] = _first_meaningful_target_idx(tuple(drone.get("pos", (0.0, 0.0))), post_route)
			drone["status"] = "return_charge"
		else:
			drone["route"] = []
			drone["target_idx"] = 0
			drone["status"] = "idle"
		return True
	route = drone.get("route") or []
	idx = int(drone.get("target_idx", 0))
	drone["status"] = resume_status if route and idx < len(route) else "idle"
	return True


def _remove_drone_from_charge_queues(drone_id: str) -> None:
	"""Гарантированно убирает drone_id из всех списков очередей зарядки (база и станции)."""
	bq = STATE.setdefault("base_queue", {"charging": [], "queue": [], "capacity": 2})
	for key in ("charging", "queue"):
		lst = bq.get(key) or []
		while drone_id in lst:
			lst.remove(drone_id)
	STATE["base_queue"] = bq
	sqs = STATE.setdefault("station_queues", {})
	for _key, sq in list(sqs.items()):
		for key2 in ("charging", "queue"):
			lst = sq.get(key2) or []
		while drone_id in lst:
			lst.remove(drone_id)


def _try_close_order_if_no_flight_after_charge(drone_id: str, drone: Dict[str, Any]) -> bool:
	"""
	После зарядки replan мог не построить маршрут; для operator area без оставшихся точек
	закрываем заказ, чтобы не зависать in_progress.
	"""
	order = _get_active_order_for_drone(drone_id)
	if not order:
		return False
	required_type = order.get("drone_type") or map_order_to_drone_type(order.get("type", "delivery"))
	if required_type != "operator" or not order.get("area_polygon"):
		return False
	if order.get("remaining_waypoints"):
		return False
	oid = order.get("id")
	# Order завершается после зарядки / передаётся в continuation — см. mark_order_completed_if_any.
	mark_order_completed_if_any(drone_id)
	o2 = _get_active_order_for_drone(drone_id)
	return o2 is None or o2.get("id") != oid


def _force_exit_charging_if_complete(drone_id: str, drone: Dict[str, Any]) -> None:
	"""
	Исправление инконсистентности: battery >= CHARGE_COMPLETE_PCT, status == charging.
	Дрон выходит из charging, покидает очереди; далее resume миссии или idle + завершение заказа.
	"""
	if drone.get("status") != "charging":
		return
	if float(drone.get("battery", 0.0)) < CHARGE_COMPLETE_PCT:
		return
	_remove_drone_from_charge_queues(drone_id)
	drone["battery"] = 100.0
	_resume_after_charge_or_hold(drone_id, drone)
	if drone.get("status") == "charging":
		rt = drone.get("route") or []
		ti = int(drone.get("target_idx", 0))
		if rt and ti < len(rt):
			drone["status"] = "enroute"
		else:
			mark_order_completed_if_any(drone_id)
			drone["status"] = "idle"


def _repair_invalid_charging_state(drone_id: str, drone: Dict[str, Any]) -> None:
	"""Чинит зависшее состояние charging, если дрон не находится у базы или станции."""
	if drone.get("status") != "charging":
		return
	pos = tuple(drone.get("pos") or ())
	if len(pos) != 2:
		return
	if _is_at_charge_location(pos):
		return
	_remove_drone_from_charge_queues(drone_id)
	if float(drone.get("battery", 0.0)) >= CHARGE_COMPLETE_PCT:
		_resume_after_charge_or_hold(drone_id, drone)
		return
	order = _get_active_order_for_drone(drone_id)
	if order:
		drone["status"] = "holding"
		if not (drone.get("route") or []):
			_rebuild_active_order_route(drone_id, drone, order)
		return
	drone["status"] = "low_battery"
	maybe_route_to_base_or_station(drone)


def _mark_order_in_progress_if_started(drone_id: str, drone: Dict[str, Any]) -> None:
	"""Переводит заказ в статус выполнения, когда дрон реально начинает движение по маршруту миссии."""
	order = _get_active_order_for_drone(drone_id)
	if not order:
		return
	route = drone.get("route") or []
	if not route:
		return
	if int(drone.get("target_idx", 0)) >= len(route):
		return
	if drone.get("status") in ("enroute", "return_charge", "return_base"):
		if order.get("status") == "assigned":
			order["status"] = "in_progress"


def _sanitize_active_drone_state(drone_id: str, drone: Dict[str, Any]) -> None:
	"""
	Инвариант state-machine:
	если есть active_order и непустой route, дрон не должен оставаться idle/holding.
	"""
	order = _get_active_order_for_drone(drone_id)
	route = drone.get("route") or []
	idx = int(drone.get("target_idx", 0))
	if order and route and idx < len(route) and drone.get("status") in ("idle", "holding"):
		drone["status"] = "enroute"
	# Дрон в holding без маршрута после неудачного replan — пробуем восстановить миссию или закрыть заказ.
	if drone.get("status") == "holding" and order and (not route or idx >= len(route)):
		if _rebuild_active_order_route(drone_id, drone, order):
			pass
		elif _restore_route_after_charging(drone):
			pass
		elif _try_close_order_if_no_flight_after_charge(drone_id, drone):
			pass
	_repair_invalid_charging_state(drone_id, drone)
	# Защита: battery >= порога завершения зарядки, но статус всё ещё charging — выходим из зависшего состояния.
	_force_exit_charging_if_complete(drone_id, drone)


def _find_existing_continuation(parent_order: Dict[str, Any], remaining_waypoints: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
	"""Ищет уже созданное продолжение заказа, чтобы не создавать дубликаты для той же миссии и остатка маршрута."""
	mission_id = str(parent_order.get("mission_id") or _operator_area_root_order_id(parent_order))
	fingerprint = f"{mission_id}:{len(remaining_waypoints)}:{tuple(remaining_waypoints[:1] or [])}:{tuple(remaining_waypoints[-1:] or [])}"
	for o in STATE.get("orders", []):
		if o.get("status") not in ("queued", "assigned", "in_progress", "waiting_continuation"):
			continue
		if str(o.get("continuation_fingerprint") or "") == fingerprint:
			return o
	return None


def _new_operator_area_continuation(order: Dict[str, Any], handover_point: Tuple[float, float], remaining_waypoints: List[Tuple[float, float]]) -> Dict[str, Any]:
	"""Выполняет служебную операцию: новый операторский дрон зону работ продолжение."""
	root_id = _operator_area_root_order_id(order)
	history = list(order.get("handover_history") or [])
	history.append(tuple(handover_point))
	cont_idx = int(order.get("continuation_index", 0)) + 1
	mission_id = str(order.get("mission_id") or root_id)
	fingerprint = f"{mission_id}:{len(remaining_waypoints)}:{tuple(remaining_waypoints[:1] or [])}:{tuple(remaining_waypoints[-1:] or [])}"
	return {
		"id": f"ord_cont_{root_id}_{cont_idx}_{len(STATE.get('orders', []))}",
		"type": order.get("type", "shooting"),
		"drone_type": "operator",
		"priority": order.get("priority", 5),
		"start": tuple(handover_point),
		"end": tuple(remaining_waypoints[-1]) if remaining_waypoints else tuple(handover_point),
		"battery_level": 100.0,
		"status": "queued",
		"area_polygon": order.get("area_polygon"),
		"mission_mode": "operator_area",
		"remaining_waypoints": [tuple(w) for w in remaining_waypoints],
		"completed_waypoints_count": int(order.get("completed_waypoints_count", 0)),
		"handover_history": history,
		"handover_point": tuple(handover_point),
		"root_order_id": root_id,
		"continuation_index": cont_idx,
		"is_area_continuation": True,
		"parent_order_id": order.get("id"),
		"continuation_of": order.get("id"),
		"mission_id": mission_id,
		"continuation_fingerprint": fingerprint,
		"continuation_spawned": False,
	}


def plan_operator_area_trip(
	drone: Dict[str, Any], order: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
	"""
	Планирование облёта области операторским дроном. Если на одном аккумуляторе не облететь,
	возвращает маршрут до точки «передачи» и заказ-продолжение для второго дрона.
	Возвращает (result, continuation_order). result как у plan_order_trip; continuation_order или None.
	"""
	city = STATE.get("city")
	if not city:
		return ({"ok": False, "reason": "no city", "details": "no selected city"}, None)
	_routing_service.set_battery_mode(STATE.get("battery_mode", "reality"))
	waypoints = _operator_area_waypoints(order)
	if not waypoints:
		return ({"ok": False, "reason": "no area", "details": "no area_polygon or bounds"}, None)
	battery_pct = float(drone.get("battery", 100.0))
	pos = tuple(drone.get("pos", order.get("start", (0, 0))))
	polygon = [tuple(p) for p in (order.get("area_polygon") or []) if isinstance(p, (list, tuple)) and len(p) == 2]
	continuation_order = None
	route_info = _build_operator_area_route(pos, battery_pct, polygon, waypoints, "operator")
	if not route_info.get("ok"):
		return ({"ok": False, "reason": route_info.get("reason", "NO_PATH"), "details": route_info.get("details", "area route failed")}, None)
	coverage_coords = _to_coord_list(route_info.get("coverage_coords") or [])
	remaining_part = _to_coord_list(route_info.get("remaining_waypoints") or [])
	if not coverage_coords and remaining_part:
		# Даже coverage не начат: оставляем continuation в ожидании другого оператора.
		existing = _find_existing_continuation(order, remaining_part)
		continuation_order = existing or _new_operator_area_continuation(order, tuple(route_info.get("handover_point") or pos), remaining_part)
		return ({
			"ok": True,
			"coords": _to_coord_list(route_info.get("approach_coords") or [pos]),
			"route_length": float(route_info.get("approach_length", 0.0)),
			"pickup_waypoint_count": 0,
			"area_completed_waypoints_count": 0,
			"area_total_waypoints_count": len(_to_coord_list(_build_operator_area_flight_path(pos, polygon, waypoints))),
			"chargers_used": [],
			"battery_plan": [{"at": "start", "battery": battery_pct}],
			"segments": [{"type": "approach_graph", "coords": _to_coord_list(route_info.get("approach_coords") or []), "length": float(route_info.get("approach_length", 0.0)), "chargers": []}],
			"mission_mode": "operator_area",
			"force_charge_after_route": True,
			"coverage_start_idx": int(route_info.get("coverage_start_idx", 0)),
			"coverage_end_idx": int(route_info.get("coverage_end_idx", 0)),
		}, continuation_order)

	coords = _to_coord_list(route_info.get("combined_coords") or [])
	length = float(route_info.get("approach_length", 0.0)) + float(route_info.get("coverage_length", 0.0)) + float(route_info.get("exit_length", 0.0))
	if remaining_part:
		# Передача operator area-задачи: фиксируем handover и оставшиеся внутренние точки.
		handover = tuple(route_info.get("handover_point") or coverage_coords[-1])
		existing = _find_existing_continuation(order, remaining_part)
		continuation_order = existing or _new_operator_area_continuation(order, handover, remaining_part)
	result = {
		"ok": True,
		"coords": coords,
		"route_length": length,
		"pickup_waypoint_count": len(coords),
		"area_completed_waypoints_count": len(coverage_coords),
		"area_total_waypoints_count": len(coverage_coords) + len(remaining_part),
		"chargers_used": [],
		"battery_plan": [{"at": "start", "battery": battery_pct}],
		# Комбинированный маршрут: до зоны по graph, внутри coverage змейкой, далее выход по graph.
		"segments": [
			{"type": "approach_graph", "coords": _to_coord_list(route_info.get("approach_coords") or []), "length": float(route_info.get("approach_length", 0.0)), "chargers": []},
			{"type": "coverage_area", "coords": coverage_coords, "length": float(route_info.get("coverage_length", 0.0)), "chargers": []},
			{"type": "exit_graph", "coords": _to_coord_list(route_info.get("exit_coords") or []), "length": float(route_info.get("exit_length", 0.0)), "chargers": []},
		],
		"mission_mode": "operator_area",
		"force_charge_after_route": bool(continuation_order),
		"coverage_start_idx": int(route_info.get("coverage_start_idx", 0)),
		"coverage_end_idx": int(route_info.get("coverage_end_idx", 0)),
	}
	return (result, continuation_order)


def plan_operator_point_trip(
	drone_pos: Tuple[float, float],
	point: Tuple[float, float],
	drone_type: str,
	battery_pct: float,
) -> Dict[str, Any]:
	"""
	Маршрут операторского/сервисного дрона до точки: дрон → точка → зарядка.
	Оба сегмента в режиме MODE_EMPTY, с возможными остановками на зарядку.
	"""
	G = STATE.get("city_graph")
	city = STATE.get("city")
	if not city or G is None:
		return {"ok": False, "reason": "no graph", "details": "no city graph"}
	_routing_service.set_battery_mode(STATE.get("battery_mode", "reality"))
	_routing_service.city_graphs[city] = G
	charger_nodes = STATE.get("charger_nodes") or {"base": None, "stations": []}
	plan_reserve = PLAN_RESERVE_PCT_TEST if STATE.get("battery_mode") == "test" else PLAN_RESERVE_PCT
	start_node = _find_graph_node_for_point(drone_pos, prefer_primary=True)
	point_node = _find_graph_node_for_point(point, prefer_primary=True)
	if not start_node or not point_node:
		return {"ok": False, "reason": NO_PATH_TO_PICKUP, "details": "nearest node not found"}
	# Сегмент 1: дрон → точка (с зарядками по пути)
	path_a, coords_a, len_a, chargers_a = _routing_service.plan_with_chargers(
		G, start_node, point_node, battery_pct, MODE_EMPTY, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
		max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
	)
	if not path_a or not coords_a:
		return {"ok": False, "reason": NO_FEASIBLE_CHARGING_CHAIN_TO_PICKUP, "details": "no path to point"}
	battery_after_a = _routing_service.compute_battery_after(len_a, battery_pct, MODE_EMPTY, drone_type)
	if chargers_a:
		battery_after_a = 100.0
	# Сегмент 2: точка → зарядка (escape)
	base_node = charger_nodes.get("base")
	station_nodes = charger_nodes.get("stations") or []
	path_c, coords_c, len_c, chargers_c = None, None, 0.0, []
	best_c = None
	best_len_c = float("inf")
	for name, goal_n in [("base", base_node)] + [(f"station_{i}", sn) for i, sn in enumerate(station_nodes) if sn]:
		if goal_n is None or goal_n not in G.nodes:
			continue
		p_c, co_c, l_c, ch_c = _routing_service.plan_with_chargers(
			G, point_node, goal_n, battery_after_a, MODE_EMPTY, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
			max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
		)
		if p_c and co_c and l_c < best_len_c:
			best_len_c = l_c
			best_c = (p_c, co_c, l_c, ch_c)
	if best_c:
		path_c, coords_c, len_c, chargers_c = best_c
	full_coords = list(coords_a)
	if coords_c:
		full_coords.extend(coords_c[1:])
	total_length = len_a + len_c
	chargers_used = list(dict.fromkeys(chargers_a + chargers_c))
	battery_after_escape = 100.0 if chargers_c else _routing_service.compute_battery_after(len_c, battery_after_a, MODE_EMPTY, drone_type)
	return {
		"ok": True,
		"coords": full_coords,
		"segments": [
			{"type": "to_point", "coords": coords_a, "length": len_a, "chargers": chargers_a},
			{"type": "escape", "coords": coords_c or [], "length": len_c, "chargers": chargers_c},
		],
		"chargers_used": chargers_used,
		"battery_plan": [
			{"at": "start", "battery": battery_pct},
			{"at": "after_point", "battery": battery_after_a},
			{"at": "after_escape", "battery": battery_after_escape},
		],
		"pickup_waypoint_count": len(coords_a),
		"route_length": total_length,
	}


def plan_operator_continuation_trip(drone: Dict[str, Any], order: Dict[str, Any]) -> Dict[str, Any]:
	"""Создает продолжение операторской задачи: перелет к точке передачи и последующий облёт области."""
	remaining = _operator_area_waypoints(order)
	if not remaining:
		return {"ok": False, "reason": "NO_PATH", "details": "no remaining waypoints"}
	handover = order.get("handover_point")
	if not handover:
		handover = remaining[0]
	handover = tuple(handover) if isinstance(handover, (list, tuple)) and len(handover) == 2 else remaining[0]
	if not remaining or tuple(remaining[0]) != tuple(handover):
		remaining = [tuple(handover)] + [tuple(w) for w in remaining]
	# Передаём в общий area-планировщик только хвост после handover.
	# Это обеспечивает многошаговую передачу без отдельной одноразовой логики.
	tmp_order = dict(order)
	tmp_order["remaining_waypoints"] = list(remaining)
	tmp_order["start"] = handover
	return plan_operator_area_trip(drone, tmp_order)[0]


async def rebuild_graph_with_zones():
	"""Перестраивает граф с учетом зоны."""
	city = STATE.get("city")
	if not city:
		return
	try:
		city_data = _data_service.get_city_data(city)
		city_data['no_fly_zones'] = list(STATE["no_fly_zones"]) or city_data.get('no_fly_zones', [])
		drone_type = STATE.get("drone_type") or "cargo"
		city_graph = _graph_service.build_city_graph(city_data, drone_type)
		_routing_service.city_graphs[city] = city_graph
		STATE["city_graph"] = city_graph
		_refresh_primary_component_nodes()
		refresh_charger_nodes()
		await persist_state()
	except Exception:
		logger.exception("Failed to rebuild graph with zones")


def _reset_order_plan_backoff(order: Dict[str, Any]) -> None:
	"""Сбрасывает задержку повторного планирования заказа."""
	for key in (
		"last_plan_attempt_at",
		"next_plan_attempt_at",
		"plan_fail_count",
		"plan_last_reason",
		"plan_last_details",
		"plan_skip_drone_ids",
		"plan_last_drone_id",
	):
		order.pop(key, None)


def _schedule_order_plan_retry(order: Dict[str, Any], reason: str, details: str, min_delay_s: float = ASSIGN_PLAN_BACKOFF_MIN_S) -> None:
	"""Назначает повторную попытку планирования заказа с задержкой."""
	now = time.time()
	fail_count = int(order.get("plan_fail_count", 0) or 0) + 1
	delay = max(min_delay_s, ASSIGN_PLAN_BACKOFF_MIN_S * (2 ** min(fail_count - 1, 3)))
	delay = min(ASSIGN_PLAN_BACKOFF_MAX_S, delay)
	order["plan_fail_count"] = fail_count
	order["plan_last_reason"] = reason
	order["plan_last_details"] = details
	order["last_plan_attempt_at"] = now
	order["next_plan_attempt_at"] = now + delay


def assign_orders():
	# Очередь заказов: назначаем только на свободный дрон нужной категории; приоритет — по полю priority (больше = раньше).
	"""Назначает ожидающие заказы подходящим дронам с учетом типа, заряда и доступного маршрута."""
	city = STATE.get("city")
	G = STATE.get("city_graph")
	if not city or G is None:
		return
	queue = [o for o in STATE["orders"] if o.get("status") == "queued"]
	# Сортируем по приоритету (убывание), при равном — по порядку в списке (FIFO)
	order_indices = {o.get("id"): i for i, o in enumerate(STATE["orders"])}
	queue = sorted(queue, key=lambda o: (-o.get("priority", 0), order_indices.get(o.get("id"), 0)))
	now = time.time()
	plan_attempts = 0
	for order in queue:
		next_attempt_at = float(order.get("next_plan_attempt_at", 0.0) or 0.0)
		if next_attempt_at > now:
			continue
		if plan_attempts >= MAX_PLAN_ATTEMPTS_PER_TICK:
			break
		plan_attempts += 1
		required_type = order.get("drone_type") or map_order_to_drone_type(order.get("type", "delivery"))
		drone_id = pick_drone_for_order(order, required_type)
		if drone_id is None:
			_schedule_order_plan_retry(order, "NO_AVAILABLE_DRONE", "no idle drone or feasible new launch", min_delay_s=2.0)
			continue
		drone = STATE["drones"][drone_id]
		continuation = None
		# Грузовой: три фазы (забор → доставка → зарядка). Операторский/сервисный: точка или область.
		if required_type == "cargo":
			result = plan_order_trip(
				drone["pos"], order["start"], order["end"], drone["type"], drone["battery"]
			)
		elif required_type == "operator" and (order.get("remaining_waypoints") or (order.get("handover_point") and order.get("rest_waypoints"))):
			tmp_order = dict(order)
			remaining = _operator_area_waypoints(order)
			handover = order.get("handover_point")
			if handover and (not remaining or tuple(remaining[0]) != tuple(handover)):
				tmp_order["remaining_waypoints"] = [tuple(handover)] + [tuple(w) for w in remaining]
			result, continuation = plan_operator_area_trip(drone, tmp_order)
			if result and result.get("ok") and continuation and not any(o.get("id") == continuation.get("id") for o in STATE.get("orders", [])):
				# Многошаговая передача area-задачи: каждый этап может создать следующий continuation.
				STATE["orders"].append(continuation)
				order["continuation_spawned"] = True
				order["status"] = "waiting_continuation"
			logger.info("assign_orders: area continuation order %s -> drone %s", order.get("id"), drone_id)
		elif required_type == "operator" and order.get("area_polygon"):
			order["mission_mode"] = "operator_area"
			order["mission_id"] = str(order.get("mission_id") or order.get("id"))
			if not order.get("area_waypoints"):
				order["area_waypoints"] = [tuple(w) for w in _operator_area_waypoints(order)]
			if not order.get("remaining_waypoints"):
				order["remaining_waypoints"] = list(order.get("area_waypoints") or [])
			result, continuation = plan_operator_area_trip(drone, order)
			if result and result.get("ok") and continuation and not any(o.get("id") == continuation.get("id") for o in STATE.get("orders", [])):
				STATE["orders"].append(continuation)
				order["continuation_spawned"] = True
				order["status"] = "waiting_continuation"
				logger.info("assign_orders: created continuation order %s (handover at %s), first drone %s", continuation.get("id"), continuation.get("handover_point"), drone_id)
			if not result or not result.get("ok"):
				result = result or {"ok": False, "reason": "NO_PATH", "details": "operator area plan failed"}
		else:
			# Операторская точка или сервисный: дрон → точка → зарядка (отдельное планирование)
			result = plan_operator_point_trip(
				drone["pos"], order["end"], drone["type"], drone["battery"]
			)
		if not result.get("ok"):
			skip_ids = list(order.get("plan_skip_drone_ids") or [])
			if drone_id not in skip_ids:
				skip_ids.append(drone_id)
			order["plan_skip_drone_ids"] = skip_ids
			order["plan_last_drone_id"] = drone_id
			_schedule_order_plan_retry(
				order,
				str(result.get("reason") or "PLAN_FAILED"),
				str(result.get("details") or "route planning failed"),
			)
			logger.info(
				"assign_orders: order %s not assigned to %s reason=%s details=%s",
				order.get("id"), drone_id, result.get("reason"), result.get("details")
			)
			continue
		_reset_order_plan_backoff(order)
		full_coords = result["coords"]
		apply_midroute_charging(drone, full_coords)
		chargers_used = result.get("chargers_used", [])
		pickup_wp = result.get("pickup_waypoint_count", len(full_coords))
		logger.info(
			"assign_orders: order %s -> %s, route_length=%.0fm, chargers_used=%s (%s), pickup_waypoints=%s",
			order.get("id"), drone_id, result["route_length"], chargers_used, len(chargers_used), pickup_wp
		)
		order["status"] = "assigned"
		order["drone_id"] = drone_id
		order["pickup_serviced"] = bool(order.get("pickup_serviced", False))
		order["delivery_serviced"] = bool(order.get("delivery_serviced", False))
		order["route_length"] = result["route_length"]
		order["pickup_waypoint_count"] = pickup_wp
		if required_type == "operator" and (order.get("area_polygon") or order.get("remaining_waypoints")):
			order["mission_mode"] = "operator_area"
			_advance_area_order_progress(order, [tuple(p) for p in (result.get("coords") or [])])
			total_wp = int(result.get("area_total_waypoints_count", 0)) or len(_operator_area_waypoints(order))
			order["total_waypoints_count"] = total_wp
			if not order.get("area_waypoints"):
				order["area_waypoints"] = [tuple(w) for w in _operator_area_waypoints(order)]
			if continuation:
				order["remaining_waypoints"] = list(continuation.get("remaining_waypoints") or [])
				order["handover_history"] = list(continuation.get("handover_history") or order.get("handover_history") or [])
				order["has_continuation"] = True
				order["root_order_id"] = _operator_area_root_order_id(order)
				order["continuation_spawned"] = True
			else:
				order["remaining_waypoints"] = []
				order["has_continuation"] = False
				order["continuation_spawned"] = False
		order["chargers_used"] = result.get("chargers_used", [])
		order["battery_plan"] = result.get("battery_plan", [])
		order["segments"] = result.get("segments", [])
		drone["loaded_after_waypoint_count"] = pickup_wp
		drone["waypoints_completed"] = 0
		drone.pop("post_delivery_route", None)
		drone.pop("service_pause_ticks", None)
		drone.pop("service_pause_reason", None)
		drone.pop("service_resume_status", None)
		drone["active_order_id"] = order.get("id")
		drone["mission_mode"] = result.get("mission_mode")
		drone["force_charge_after_route"] = bool(result.get("force_charge_after_route"))
		_monitoring_service.bind_assignment(order.get("id"), drone_id, drone)

def _estimate_speed_mps() -> float:
	"""Скорость дрона (м/с) для оценки ETA с учётом ветра."""
	wind = float(STATE.get("weather", {}).get("wind_mps", 3.0))
	base_speed = max(5.0, 15.0 - 0.3 * wind)
	return base_speed * SIMULATION_SPEED_MULTIPLIER


def _order_completion_time_seconds(result: Dict[str, Any]) -> float:
	"""Оценка времени выполнения заказа в секундах: полёт + остановки на зарядку."""
	if not result or not result.get("ok"):
		return float("inf")
	route_m = float(result.get("route_length", 0))
	chargers = result.get("chargers_used") or []
	speed = _estimate_speed_mps()
	flight_sec = route_m / speed if speed > 0 else 0
	charge_sec = len(chargers) * CHARGE_TICKS_ESTIMATE_PER_STOP
	return flight_sec + charge_sec


def _run_plan_for_order(drone_or_base_pos: Tuple[float, float], battery_pct: float, order: Dict[str, Any], required_type: str) -> Optional[Dict[str, Any]]:
	"""Запускает планировщик для заказа с заданной позиции и заряда. Не изменяет STATE (кроме графа)."""
	if required_type == "cargo":
		return plan_order_trip(
			drone_or_base_pos, order["start"], order["end"], required_type, battery_pct
		)
	if required_type == "operator" and (order.get("remaining_waypoints") or (order.get("handover_point") and order.get("rest_waypoints"))):
		# continuation — нужен объект дрона с pos и battery
		drone = {"pos": drone_or_base_pos, "battery": battery_pct, "type": "operator"}
		return plan_operator_continuation_trip(drone, order)
	if required_type == "operator" and order.get("area_polygon"):
		drone = {"pos": drone_or_base_pos, "battery": battery_pct, "type": "operator"}
		result, _ = plan_operator_area_trip(drone, order)
		return result
	# operator point / cleaner
	point = order.get("end") or order.get("start")
	if not point:
		return None
	point = tuple(point) if isinstance(point, (list, tuple)) and len(point) == 2 else None
	if not point:
		return None
	return plan_operator_point_trip(drone_or_base_pos, point, required_type, battery_pct)


def estimate_order_completion_time_seconds(drone: Dict[str, Any], order: Dict[str, Any], required_type: str) -> Tuple[float, Optional[Dict[str, Any]]]:
	"""Оценка времени (сек) до завершения заказа этим дроном (с учётом зарядки). Возвращает (секунды, result или None)."""
	pos = drone.get("pos")
	if not pos:
		return (float("inf"), None)
	pos = tuple(pos) if isinstance(pos, (list, tuple)) and len(pos) == 2 else None
	if not pos:
		return (float("inf"), None)
	battery = float(drone.get("battery", 100.0))
	result = _run_plan_for_order(pos, battery, order, required_type)
	if not result or not result.get("ok"):
		return (float("inf"), None)
	return (_order_completion_time_seconds(result), result)


def estimate_new_drone_completion_time_seconds(order: Dict[str, Any], required_type: str) -> Tuple[float, Optional[Dict[str, Any]]]:
	"""Оценка времени (сек) до завершения заказа «новым» дроном с базы (100% заряд)."""
	base = STATE.get("base")
	if not base:
		return (float("inf"), None)
	base = tuple(base) if isinstance(base, (list, tuple)) and len(base) == 2 else None
	if not base:
		return (float("inf"), None)
	result = _run_plan_for_order(base, 100.0, order, required_type)
	if not result or not result.get("ok"):
		return (float("inf"), None)
	return (_order_completion_time_seconds(result), result)


def pick_drone_for_order(order: Dict[str, Any], required_drone_type: str) -> Optional[str]:
	"""Быстрый выбор дрона по близости и заряду без полного перепланирования всех кандидатов."""
	order_start = order.get("handover_point")
	if not order_start and order.get("remaining_waypoints"):
		order_start = (order.get("remaining_waypoints") or [None])[0]
	if not order_start:
		order_start = order.get("start") or order.get("end")
	if not order_start:
		return None
	order_start = tuple(order_start) if isinstance(order_start, (list, tuple)) and len(order_start) == 2 else None
	if not order_start:
		return None

	skipped_ids = set(order.get("plan_skip_drone_ids") or [])
	eligible: List[Tuple[float, float, str]] = []
	for drone_id, d in STATE["drones"].items():
		if d.get("type") != required_drone_type:
			continue
		if d.get("status") not in (None, "idle", "holding"):
			continue
		if d.get("status") == "holding" and _get_active_order_for_drone(drone_id):
			continue
		if not d.get("pos"):
			continue
		battery = float(d.get("battery", 100.0))
		if battery <= FLY_TO_CHARGER_AT_PCT:
			continue
		dist = haversine_m(tuple(d["pos"]), order_start)
		eligible.append((dist, -battery, drone_id))

	eligible.sort()
	if skipped_ids:
		filtered = [item for item in eligible if item[2] not in skipped_ids]
		if filtered:
			eligible = filtered

	# Для continuation/handover задач не спавним новый дрон автоматически:
	# если свободных нет — заказ ждёт в очереди.
	is_continuation = bool(order.get("remaining_waypoints") or order.get("handover_point") or order.get("is_area_continuation"))
	if is_continuation and not eligible:
		return None

	inv = STATE.get("inventory") or {}
	has_inventory = (inv.get(required_drone_type, 0) or 0) > 0
	deployment_points: List[Tuple[str, Tuple[float, float]]] = []
	base = STATE.get("base")
	if base and isinstance(base, (list, tuple)) and len(base) == 2:
		deployment_points.append(("base", tuple(base)))
	for idx, station in enumerate(STATE.get("stations") or []):
		if isinstance(station, (list, tuple)) and len(station) == 2:
			deployment_points.append((f"station_{idx}", tuple(station)))
	if has_inventory and deployment_points:
		deploy_name, deploy_pos = min(deployment_points, key=lambda item: haversine_m(item[1], order_start))
		deploy_dist = haversine_m(deploy_pos, order_start)
		best_existing = eligible[0][0] if eligible else float("inf")
		if not eligible or deploy_dist + 250.0 < best_existing:
			new_id = spawn_drone_from_inventory(
				required_drone_type,
				order_id=str(order.get("id")),
				caller="pick_drone_for_order",
				spawn_pos=deploy_pos,
			)
			if new_id:
				logger.info(
					"pick_drone_for_order: order=%s -> new drone %s from %s (deploy_dist=%.0fm, best_existing=%.0fm)",
					order.get("id"), new_id, deploy_name, deploy_dist, best_existing,
				)
				return new_id

	if eligible:
		chosen_id = eligible[0][2]
		logger.info(
			"pick_drone_for_order: order=%s -> drone %s (straight distance %.0fm)",
			order.get("id"), chosen_id, eligible[0][0],
		)
		return chosen_id
	return None


def map_order_to_drone_type(order_type: str) -> str:
	"""Сопоставляет тип заказа с требуемым типом дрона."""
	if order_type == "delivery":
		return "cargo"
	if order_type == "shooting":
		return "operator"
	return "cleaner"


def plan_route_for(
	start: Tuple[float, float],
	end: Tuple[float, float],
	drone_type: str,
	battery_level: float,
	waypoints: Optional[List[Tuple[float, float]]] = None,
	reserve_pct: Optional[float] = None,
):
	"""reserve_pct: при None используется RESERVE_PCT; при 0 — весь заряд на путь (для экстренного вылета на зарядку)."""
	city = STATE.get("city")
	G = STATE.get("city_graph")
	if not city or G is None:
		return None, None, 0.0
	_routing_service.city_graphs[city] = G
	res = reserve_pct if reserve_pct is not None else RESERVE_PCT
	max_range = _routing_service.max_reachable_distance(battery_level, MODE_EMPTY, drone_type, reserve_pct=res)
	path, coords, length = _routing_service.plan_direct_path(G, start, end, max_range, waypoints=waypoints)
	if path:
		return path, coords, length
	# fallback to node-to-node
	start_node = _find_graph_node_for_point(start, prefer_primary=True)
	end_node = _find_graph_node_for_point(end, prefer_primary=True)
	if not start_node or not end_node:
		return None, None, 0.0
	path = _routing_service._find_safe_path(G, start_node, end_node, max_range)
	if not path:
		return None, None, 0.0
	coords = [G.nodes[n]['pos'] for n in path]
	length = _routing_service._calculate_path_length(G, path)
	return path, coords, length


def _plan_order_trip_via_charger_first(
	G, drone_pos: Tuple[float, float], pickup: Tuple[float, float], dropoff: Tuple[float, float],
	drone_type: str, battery_pct: float, charger_nodes: Dict, pickup_node, dropoff_node, start_node,
) -> Optional[Dict[str, Any]]:
	"""Строит маршрут: дрон → ближайшая станция зарядки → забор → доставка → зарядка. Для грузового с запасом 20% после доставки."""
	base_node = charger_nodes.get("base")
	station_nodes = charger_nodes.get("stations") or []
	candidates = []
	if base_node and base_node in G.nodes:
		candidates.append(("base", base_node))
	for i, sn in enumerate(station_nodes):
		if sn and sn in G.nodes:
			candidates.append((f"station_{i}", sn))
	if not candidates:
		return None
	# Ближайшая зарядка от текущей позиции
	best_name, best_node = None, None
	best_len = float("inf")
	for name, node in candidates:
		p, co, l, _ = _routing_service.plan_with_chargers(
			G, start_node, node, battery_pct, MODE_EMPTY, drone_type, reserve_pct=CHARGER_ARRIVAL_MIN_PCT, charger_nodes=charger_nodes,
			max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
		)
		if p and co and l < best_len:
			best_len = l
			best_name, best_node = name, node
	if not best_node:
		return None
	# Сегмент до зарядки
	path_to_ch, coords_to_ch, len_to_ch, ch_to = _routing_service.plan_with_chargers(
		G, start_node, best_node, battery_pct, MODE_EMPTY, drone_type, reserve_pct=CHARGER_ARRIVAL_MIN_PCT, charger_nodes=charger_nodes,
		max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
	)
	if not path_to_ch or not coords_to_ch:
		return None
	plan_reserve = CARGO_RESERVE_AFTER_DELIVERY_PCT
	# От зарядки (100%) до забора
	path_a, coords_a, len_a, chargers_a = _routing_service.plan_with_chargers(
		G, best_node, pickup_node, 100.0, MODE_EMPTY, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
		max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
	)
	if not path_a or not coords_a:
		return None
	battery_after_a = 100.0 if chargers_a else _routing_service.compute_battery_after(len_a, 100.0, MODE_EMPTY, drone_type)
	# Забор → доставка
	path_b, coords_b, len_b, chargers_b = _routing_service.plan_with_chargers(
		G, pickup_node, dropoff_node, battery_after_a, MODE_LOADED, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
		max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
	)
	if not path_b or not coords_b:
		return None
	battery_after_b = _routing_service.compute_battery_after(len_b, battery_after_a, MODE_LOADED, drone_type)
	if chargers_b:
		battery_after_b = 100.0
	if battery_after_b < CARGO_RESERVE_AFTER_DELIVERY_PCT:
		return None
	# Доставка → зарядка (escape)
	path_c, coords_c, len_c, chargers_c = None, None, 0.0, []
	station_list = charger_nodes.get("stations") or []
	for name, goal_n in [("base", charger_nodes.get("base"))] + [(f"station_{i}", sn) for i, sn in enumerate(station_list) if sn]:
		if goal_n is None or goal_n not in G.nodes:
			continue
		p_c, co_c, l_c, ch_c = _routing_service.plan_with_chargers(
			G, dropoff_node, goal_n, battery_after_b, MODE_EMPTY, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
			max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
		)
		if p_c and co_c and l_c < (len_c if path_c else float("inf")):
			path_c, coords_c, len_c, chargers_c = p_c, co_c, l_c, ch_c
	if not path_c or not coords_c:
		return None
	full_coords = list(coords_to_ch)
	if coords_a:
		full_coords.extend(coords_a[1:])
	if coords_b:
		full_coords.extend(coords_b[1:])
	if coords_c:
		full_coords.extend(coords_c[1:])
	pickup_waypoint_count = len(coords_to_ch) + len(coords_a) - 1
	total_length = len_to_ch + len_a + len_b + len_c
	chargers_used = list(dict.fromkeys(ch_to + chargers_a + chargers_b + chargers_c))
	return {
		"ok": True,
		"coords": full_coords,
		"segments": [
			{"type": "to_charger", "coords": coords_to_ch, "length": len_to_ch, "chargers": ch_to},
			{"type": "to_pickup", "coords": coords_a, "length": len_a, "chargers": chargers_a},
			{"type": "to_dropoff", "coords": coords_b, "length": len_b, "chargers": chargers_b},
			{"type": "escape", "coords": coords_c, "length": len_c, "chargers": chargers_c},
		],
		"chargers_used": chargers_used,
		"battery_plan": [
			{"at": "start", "battery": battery_pct},
			{"at": "after_charger", "battery": 100.0},
			{"at": "after_pickup", "battery": battery_after_a},
			{"at": "after_dropoff", "battery": battery_after_b},
			{"at": "after_escape", "battery": 100.0},
		],
		"pickup_waypoint_count": pickup_waypoint_count,
		"route_length": total_length,
	}


def _direct_charger_points() -> List[Tuple[str, Tuple[float, float]]]:
	"""Выполняет служебную операцию: прямой маршрут зарядную станцию точки."""
	points: List[Tuple[str, Tuple[float, float]]] = []
	base = STATE.get("base")
	if base and isinstance(base, (list, tuple)) and len(base) == 2:
		points.append(("base", tuple(base)))
	for idx, station in enumerate(STATE.get("stations") or []):
		if isinstance(station, (list, tuple)) and len(station) == 2:
			points.append((f"station_{idx}", tuple(station)))
	return points


def _interpolate_air_segment(start: Tuple[float, float], end: Tuple[float, float], step_m: float = 250.0) -> List[Tuple[float, float]]:
	"""Разбивает воздушный сегмент на промежуточные точки."""
	dist = haversine_m(start, end)
	if dist <= step_m:
		return [tuple(start), tuple(end)]
	steps = max(1, int(math.ceil(dist / step_m)))
	points = [tuple(start)]
	for idx in range(1, steps):
		points.append(move_towards(tuple(start), tuple(end), idx / steps))
	points.append(tuple(end))
	return points


def _dedupe_route_points(points: List[Tuple[float, float]], min_distance_m: float = 2.0) -> List[Tuple[float, float]]:
	"""Удаляет слишком близкие дубли точек маршрута."""
	result: List[Tuple[float, float]] = []
	for point in points:
		pt = (float(point[0]), float(point[1]))
		if not result or haversine_m(result[-1], pt) >= min_distance_m:
			result.append(pt)
	return result


def _to_local_xy(point: Tuple[float, float], origin: Tuple[float, float]) -> Tuple[float, float]:
	"""Преобразует географическую точку в локальные координаты XY."""
	avg_lat = math.radians((float(point[0]) + float(origin[0])) / 2.0)
	x = (float(point[1]) - float(origin[1])) * 111320.0 * math.cos(avg_lat)
	y = (float(point[0]) - float(origin[0])) * 111320.0
	return (x, y)


def _from_local_xy(point_xy: Tuple[float, float], origin: Tuple[float, float]) -> Tuple[float, float]:
	"""Преобразует локальные координаты XY обратно в географическую точку."""
	lat = float(origin[0]) + (float(point_xy[1]) / 111320.0)
	cos_lat = max(0.25, math.cos(math.radians((float(origin[0]) + lat) / 2.0)))
	lon = float(origin[1]) + (float(point_xy[0]) / (111320.0 * cos_lat))
	return (lat, lon)


def _zone_circle_radius(zone: Dict[str, Any], padding_m: float = AIR_ZONE_PADDING_M) -> float:
	"""Возвращает радиус круговой зоны с учетом защитного отступа."""
	return max(1.0, float(zone.get("radius_m") or 0.0) + float(padding_m))


def _point_in_zone_with_padding(point: Tuple[float, float], zone: Dict[str, Any], padding_m: float = AIR_ZONE_PADDING_M) -> bool:
	"""Проверяет попадание точки в зону с учетом защитного отступа."""
	zone_type = (zone.get("zone_type") or "rectangle").lower()
	if zone_type == "circle":
		center = (float(zone.get("center_lat")), float(zone.get("center_lon")))
		return haversine_m(tuple(point), center) < _zone_circle_radius(zone, padding_m)
	lat_pad = float(padding_m) / 111320.0
	lon_pad = float(padding_m) / max(0.25, 111320.0 * math.cos(math.radians(float(point[0]))))
	lat_min = min(float(zone.get("lat_min")), float(zone.get("lat_max"))) - lat_pad
	lat_max = max(float(zone.get("lat_min")), float(zone.get("lat_max"))) + lat_pad
	lon_min = min(float(zone.get("lon_min")), float(zone.get("lon_max"))) - lon_pad
	lon_max = max(float(zone.get("lon_min")), float(zone.get("lon_max"))) + lon_pad
	return lat_min <= float(point[0]) <= lat_max and lon_min <= float(point[1]) <= lon_max


def _segment_intersects_zone_with_padding(
	start: Tuple[float, float],
	end: Tuple[float, float],
	zone: Dict[str, Any],
	padding_m: float = AIR_ZONE_PADDING_M,
) -> bool:
	"""Проверяет пересечение сегмента с зоной с учетом защитного отступа."""
	if _point_in_zone_with_padding(start, zone, padding_m) or _point_in_zone_with_padding(end, zone, padding_m):
		return True
	zone_type = (zone.get("zone_type") or "rectangle").lower()
	if zone_type == "circle":
		center = (float(zone.get("center_lat")), float(zone.get("center_lon")))
		radius_m = _zone_circle_radius(zone, padding_m)
		origin = center
		sx, sy = _to_local_xy(tuple(start), origin)
		ex, ey = _to_local_xy(tuple(end), origin)
		dx, dy = ex - sx, ey - sy
		denom = dx * dx + dy * dy
		if denom <= 1e-9:
			return math.hypot(sx, sy) < radius_m
		t = max(0.0, min(1.0, -((sx * dx) + (sy * dy)) / denom))
		px, py = sx + t * dx, sy + t * dy
		return math.hypot(px, py) < radius_m
	samples = max(8, int(math.ceil(haversine_m(tuple(start), tuple(end)) / 120.0)))
	for idx in range(samples + 1):
		point = move_towards(tuple(start), tuple(end), idx / max(1, samples))
		if _point_in_zone_with_padding(point, zone, padding_m):
			return True
	return False


def _blocking_zones_for_segment(
	start: Tuple[float, float],
	end: Tuple[float, float],
	zones: List[Dict[str, Any]],
) -> List[Tuple[float, Dict[str, Any]]]:
	"""Находит запретные зоны, блокирующие заданный сегмент маршрута."""
	result: List[Tuple[float, Dict[str, Any]]] = []
	for zone in zones:
		if not _segment_intersects_zone_with_padding(start, end, zone):
			continue
		center = _zone_center(zone)
		score = haversine_m(tuple(start), tuple(center))
		result.append((score, zone))
	result.sort(key=lambda item: item[0])
	return result


def _circle_tangent_points(point_xy: Tuple[float, float], radius_m: float) -> List[Tuple[float, float]]:
	"""Вычисляет касательные точки к круговой зоне."""
	dist = math.hypot(float(point_xy[0]), float(point_xy[1]))
	if dist <= radius_m + 1.0:
		return []
	base_angle = math.atan2(float(point_xy[1]), float(point_xy[0]))
	offset = math.acos(max(-1.0, min(1.0, radius_m / dist)))
	return [
		(radius_m * math.cos(base_angle + offset), radius_m * math.sin(base_angle + offset)),
		(radius_m * math.cos(base_angle - offset), radius_m * math.sin(base_angle - offset)),
	]


def _sample_arc_points(
	center: Tuple[float, float],
	radius_m: float,
	start_angle: float,
	end_angle: float,
	direction: str,
) -> List[Tuple[float, float]]:
	"""Строит набор точек вдоль дуги обхода круговой зоны."""
	if direction == "ccw":
		delta = (end_angle - start_angle) % (2.0 * math.pi)
	else:
		delta = -((start_angle - end_angle) % (2.0 * math.pi))
	steps = max(2, int(math.ceil((abs(delta) * radius_m) / AIR_ROUTE_STEP_M)))
	points: List[Tuple[float, float]] = []
	for idx in range(steps + 1):
		angle = start_angle + delta * (idx / max(1, steps))
		points.append(
			_from_local_xy(
				(radius_m * math.cos(angle), radius_m * math.sin(angle)),
				center,
			)
		)
	return points


def _fallback_detour_candidates(
	start: Tuple[float, float],
	end: Tuple[float, float],
	zone: Dict[str, Any],
) -> List[List[Tuple[float, float]]]:
	"""Формирует резервные варианты обхода препятствия."""
	center = _zone_center(zone)
	origin = center
	sx, sy = _to_local_xy(tuple(start), origin)
	ex, ey = _to_local_xy(tuple(end), origin)
	dx, dy = ex - sx, ey - sy
	length = math.hypot(dx, dy)
	if length <= 1.0:
		length = 1.0
	ux, uy = dx / length, dy / length
	px, py = -uy, ux
	zone_type = (zone.get("zone_type") or "rectangle").lower()
	radius_m = _zone_circle_radius(zone) if zone_type == "circle" else max(140.0, AIR_ZONE_PADDING_M * 3.0)
	candidates: List[List[Tuple[float, float]]] = []
	for sign in (-1.0, 1.0):
		waypoint_xy = (px * radius_m * sign, py * radius_m * sign)
		waypoint = _from_local_xy(waypoint_xy, origin)
		candidates.append([tuple(start), waypoint, tuple(end)])
	return candidates


def _circle_detour_candidates(
	start: Tuple[float, float],
	end: Tuple[float, float],
	zone: Dict[str, Any],
) -> List[List[Tuple[float, float]]]:
	"""Формирует варианты обхода круговой запретной зоны."""
	center = (float(zone.get("center_lat")), float(zone.get("center_lon")))
	radius_m = _zone_circle_radius(zone)
	origin = center
	start_xy = _to_local_xy(tuple(start), origin)
	end_xy = _to_local_xy(tuple(end), origin)
	start_tangents = _circle_tangent_points(start_xy, radius_m)
	end_tangents = _circle_tangent_points(end_xy, radius_m)
	if not start_tangents or not end_tangents:
		return _fallback_detour_candidates(start, end, zone)
	candidates: List[List[Tuple[float, float]]] = []
	for tangent_start in start_tangents:
		for tangent_end in end_tangents:
			start_angle = math.atan2(tangent_start[1], tangent_start[0])
			end_angle = math.atan2(tangent_end[1], tangent_end[0])
			for direction in ("cw", "ccw"):
				arc = _sample_arc_points(center, radius_m, start_angle, end_angle, direction)
				points = [tuple(start), _from_local_xy(tangent_start, origin)]
				points.extend(arc[1:-1])
				points.append(_from_local_xy(tangent_end, origin))
				points.append(tuple(end))
				candidates.append(_dedupe_route_points(points))
	return candidates or _fallback_detour_candidates(start, end, zone)


def _build_air_route(
	start: Tuple[float, float],
	end: Tuple[float, float],
	zones: Optional[List[Dict[str, Any]]] = None,
	depth: int = 0,
) -> Optional[List[Tuple[float, float]]]:
	"""Строит воздушный маршрут."""
	start = tuple(start)
	end = tuple(end)
	zone_list = list(zones if zones is not None else (STATE.get("no_fly_zones") or []))
	if haversine_m(start, end) <= AIR_ROUTE_STEP_M:
		return [start, end]
	if depth >= 5 or not zone_list:
		return _interpolate_air_segment(start, end, step_m=AIR_ROUTE_STEP_M)
	blocking = _blocking_zones_for_segment(start, end, zone_list)
	if not blocking:
		return _interpolate_air_segment(start, end, step_m=AIR_ROUTE_STEP_M)
	_, first_zone = blocking[0]
	zone_type = (first_zone.get("zone_type") or "rectangle").lower()
	if zone_type == "circle":
		candidates = _circle_detour_candidates(start, end, first_zone)
	else:
		candidates = _fallback_detour_candidates(start, end, first_zone)
	best_route: Optional[List[Tuple[float, float]]] = None
	best_length = float("inf")
	for candidate in candidates[:12]:
		assembled: List[Tuple[float, float]] = []
		valid = True
		for idx in range(len(candidate) - 1):
			segment = _build_air_route(candidate[idx], candidate[idx + 1], zone_list, depth + 1)
			if not segment:
				valid = False
				break
			assembled = _concat_coords(assembled, segment)
		if not valid or not assembled:
			continue
		assembled = _dedupe_route_points(assembled, min_distance_m=3.0)
		length = _route_length(assembled)
		if length < best_length:
			best_route = assembled
			best_length = length
	return best_route


def _plan_air_escape_to_charger(
	start: Tuple[float, float],
	battery_pct: float,
	drone_type: str,
	reserve_pct: float,
) -> Tuple[Optional[List[Tuple[float, float]]], List[str], float]:
	"""Планирует воздушный выход из зоны к зарядной станции."""
	best_coords: Optional[List[Tuple[float, float]]] = None
	best_chargers: List[str] = []
	best_length = float("inf")
	for charger_name, charger_point in _direct_charger_points():
		coords, chargers, length = _plan_direct_air_route_with_stations(
			start,
			charger_point,
			battery_pct,
			MODE_EMPTY,
			drone_type,
			reserve_pct,
			force_goal_charger=charger_name,
		)
		if coords and length < best_length:
			best_coords = coords
			best_chargers = list(dict.fromkeys(chargers + [charger_name]))
			best_length = length
	return best_coords, best_chargers, (best_length if best_coords else 0.0)


def _plan_direct_air_route_with_stations(
	start: Tuple[float, float],
	goal: Tuple[float, float],
	battery_pct: float,
	mode: str,
	drone_type: str,
	reserve_pct: float,
	force_goal_charger: Optional[str] = None,
) -> Tuple[Optional[List[Tuple[float, float]]], List[str], float]:
	"""Планирует прямой маршрут воздушный маршрут с учетом станции."""
	chargers = _direct_charger_points()
	current = tuple(start)
	goal = tuple(goal)
	current_limit = _routing_service.max_reachable_distance(battery_pct, mode, drone_type, reserve_pct=reserve_pct)
	full_limit = _routing_service.max_reachable_distance(100.0, mode, drone_type, reserve_pct=reserve_pct)
	route: List[Tuple[float, float]] = [current]
	used: List[str] = []
	visited: set[str] = set()
	for _ in range(12):
		direct_route = _build_air_route(current, goal)
		direct_len = _route_length(direct_route or [])
		if direct_route and direct_len <= current_limit:
			return _concat_coords(route, direct_route), used, _route_length(_concat_coords(route, direct_route))
		current_goal_dist = direct_len if direct_route else haversine_m(current, goal)
		candidates = []
		for name, point in chargers:
			if name in visited:
				continue
			if force_goal_charger and name != force_goal_charger:
				continue
			segment_to_charger = _build_air_route(current, point)
			leg_dist = _route_length(segment_to_charger or [])
			if not segment_to_charger or leg_dist > current_limit:
				continue
			goal_route = _build_air_route(point, goal)
			goal_dist = _route_length(goal_route or [])
			if not goal_route:
				continue
			if not force_goal_charger and goal_dist >= current_goal_dist - 80.0:
				continue
			candidates.append((goal_dist, leg_dist, name, point, segment_to_charger))
		if not candidates:
			return None, used, 0.0
		candidates.sort(key=lambda item: (item[0], item[1]))
		_, _, charger_name, charger_point, segment = candidates[0]
		route = _concat_coords(route, segment)
		used.append(charger_name)
		visited.add(charger_name)
		current = tuple(charger_point)
		current_limit = full_limit
	return None, used, 0.0


def _plan_cargo_direct_air_trip(
	drone_pos: Tuple[float, float],
	pickup: Tuple[float, float],
	dropoff: Tuple[float, float],
	battery_pct: float,
	plan_reserve: float,
) -> Optional[Dict[str, Any]]:
	"""Планирует грузовой дрон прямой маршрут воздушный поездку/полет."""
	coords_a, chargers_a, len_a = _plan_direct_air_route_with_stations(
		drone_pos, pickup, battery_pct, MODE_EMPTY, "cargo", plan_reserve
	)
	if not coords_a:
		return None
	battery_after_a = 100.0 if chargers_a else _routing_service.compute_battery_after(len_a, battery_pct, MODE_EMPTY, "cargo")
	coords_b, chargers_b, len_b = _plan_direct_air_route_with_stations(
		pickup, dropoff, battery_after_a, MODE_LOADED, "cargo", plan_reserve
	)
	if not coords_b:
		return None
	battery_after_b = 100.0 if chargers_b else _routing_service.compute_battery_after(len_b, battery_after_a, MODE_LOADED, "cargo")
	if battery_after_b < CARGO_RESERVE_AFTER_DELIVERY_PCT:
		return None
	coords_c, chargers_c, len_c = _plan_air_escape_to_charger(
		dropoff,
		battery_after_b,
		"cargo",
		plan_reserve,
	)
	if not coords_c and _direct_charger_points():
		return None
	full_coords = list(coords_a)
	if coords_b:
		full_coords.extend(coords_b[1:])
	if coords_c:
		full_coords.extend(coords_c[1:])
	chargers_used = list(dict.fromkeys(chargers_a + chargers_b + chargers_c))
	return {
		"ok": True,
		"coords": full_coords,
		"segments": [
			{"type": "to_pickup", "coords": coords_a, "length": len_a, "chargers": chargers_a},
			{"type": "to_dropoff", "coords": coords_b, "length": len_b, "chargers": chargers_b},
			{"type": "escape", "coords": coords_c or [], "length": len_c, "chargers": chargers_c},
		],
		"chargers_used": chargers_used,
		"battery_plan": [
			{"at": "start", "battery": battery_pct},
			{"at": "after_pickup", "battery": battery_after_a},
			{"at": "after_dropoff", "battery": battery_after_b},
			{"at": "after_escape", "battery": 100.0 if chargers_c else battery_after_b},
		],
		"pickup_waypoint_count": len(coords_a),
		"route_length": len_a + len_b + len_c,
	}


def plan_order_trip(
	drone_pos: Tuple[float, float],
	pickup: Tuple[float, float],
	dropoff: Tuple[float, float],
	drone_type: str,
	battery_pct: float,
) -> Dict[str, Any]:
	"""
	Three-phase: A (empty) drone->pickup, B (loaded) pickup->dropoff, C (empty) dropoff->charger.
	Каждая фаза может проходить через несколько станций зарядки: если прямого пути не хватает по заряду,
	строится маршрут до станции, затем от неё до цели; при необходимости добавляются следующие станции.
	Returns {ok, coords, segments, chargers_used, battery_plan, pickup_waypoint_count, route_length}
	or {ok: False, reason, details}.
	"""
	G = STATE.get("city_graph")
	city = STATE.get("city")
	if not city or G is None:
		return {"ok": False, "reason": NO_PATH_TO_PICKUP, "details": "no city graph"}
	# Всегда синхронизируем режим батареи из STATE, чтобы тест/реальность применялись при планировании
	_routing_service.set_battery_mode(STATE.get("battery_mode", "reality"))
	_routing_service.city_graphs[city] = G
	charger_nodes = STATE.get("charger_nodes") or {"base": None, "stations": []}
	# В тесте — большой запас; для грузового — после доставки оставляем 20%
	plan_reserve = PLAN_RESERVE_PCT_TEST if STATE.get("battery_mode") == "test" else PLAN_RESERVE_PCT
	if drone_type == "cargo":
		plan_reserve = max(plan_reserve, CARGO_RESERVE_AFTER_DELIVERY_PCT)
		direct_result = _plan_cargo_direct_air_trip(drone_pos, pickup, dropoff, battery_pct, plan_reserve)
		if direct_result:
			return direct_result

	pickup_node = _find_graph_node_for_point(pickup, prefer_primary=True)
	dropoff_node = _find_graph_node_for_point(dropoff, prefer_primary=True)
	start_node = _find_graph_node_for_point(drone_pos, prefer_primary=True)
	if not start_node or not pickup_node or not dropoff_node:
		logger.warning("plan_order_trip: missing node start=%s pickup=%s dropoff=%s", start_node, pickup_node, dropoff_node)
		return {"ok": False, "reason": NO_PATH_TO_PICKUP, "details": "nearest node not found"}

	# Попытка без предварительной зарядки; для грузового проверяем, что после доставки остаётся >= 20%
	path_a, coords_a, len_a, chargers_a = _routing_service.plan_with_chargers(
		G, start_node, pickup_node, battery_pct, MODE_EMPTY, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
		max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
	)
	if not path_a or not coords_a:
		if drone_type == "cargo":
			direct_result = _plan_cargo_direct_air_trip(drone_pos, pickup, dropoff, battery_pct, plan_reserve)
			if direct_result:
				logger.info("plan_order_trip: direct air fallback used for cargo pickup route")
				return direct_result
		# Не хватает заряда до точки забора — строим маршрут через станцию зарядки
		if drone_type == "cargo" and (charger_nodes.get("base") or (charger_nodes.get("stations") or [])):
			result = _plan_order_trip_via_charger_first(G, drone_pos, pickup, dropoff, drone_type, battery_pct, charger_nodes, pickup_node, dropoff_node, start_node)
			if result:
				logger.info("plan_order_trip: route via charger first (battery=%.1f%% -> pickup)", battery_pct)
				return result
		logger.info("plan_order_trip: no feasible chain to pickup battery=%.1f", battery_pct)
		return {"ok": False, "reason": NO_FEASIBLE_CHARGING_CHAIN_TO_PICKUP, "details": "no path to pickup"}

	battery_after_a = _routing_service.compute_battery_after(len_a, battery_pct, MODE_EMPTY, drone_type)
	if chargers_a:
		battery_after_a = 100.0

	# Stage B: pickup -> dropoff (loaded); для грузового после доставки должно остаться >= 20%
	path_b, coords_b, len_b, chargers_b = _routing_service.plan_with_chargers(
		G, pickup_node, dropoff_node, battery_after_a, MODE_LOADED, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
		max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
	)
	if not path_b or not coords_b:
		if drone_type == "cargo":
			direct_result = _plan_cargo_direct_air_trip(drone_pos, pickup, dropoff, battery_pct, plan_reserve)
			if direct_result:
				logger.info("plan_order_trip: direct air fallback used for cargo loaded route")
				return direct_result
		if drone_type == "cargo" and (charger_nodes.get("base") or (charger_nodes.get("stations") or [])):
			result = _plan_order_trip_via_charger_first(G, drone_pos, pickup, dropoff, drone_type, battery_pct, charger_nodes, pickup_node, dropoff_node, start_node)
			if result:
				logger.info("plan_order_trip: route via charger first (pickup->dropoff loaded infeasible)")
				return result
		logger.info("plan_order_trip: no feasible chain pickup->dropoff loaded")
		return {"ok": False, "reason": NO_FEASIBLE_CHAIN_PICKUP_TO_DROPOFF_LOADED, "details": "loaded segment infeasible"}

	battery_after_b = _routing_service.compute_battery_after(len_b, battery_after_a, MODE_LOADED, drone_type)
	if chargers_b:
		battery_after_b = 100.0
	# Грузовой: после доставки должно оставаться >= 20%; иначе маршрут через зарядку в начале
	if drone_type == "cargo" and battery_after_b < CARGO_RESERVE_AFTER_DELIVERY_PCT:
		result = _plan_order_trip_via_charger_first(G, drone_pos, pickup, dropoff, drone_type, battery_pct, charger_nodes, pickup_node, dropoff_node, start_node)
		if result:
			logger.info("plan_order_trip: route via charger first (battery after dropoff %.1f%% < 20%%)", battery_after_b)
			return result
		return {"ok": False, "reason": NO_FEASIBLE_CHAIN_PICKUP_TO_DROPOFF_LOADED, "details": "insufficient battery after delivery (need 20%% reserve)"}

	# Stage C: dropoff -> any charger (escape)
	path_c, coords_c, len_c, chargers_c = None, None, 0.0, []
	base_node = charger_nodes.get("base")
	station_nodes = charger_nodes.get("stations") or []
	best_c = None
	best_len_c = float("inf")
	for name, goal_n in [("base", base_node)] + [(f"station_{i}", sn) for i, sn in enumerate(station_nodes) if sn]:
		if goal_n is None or goal_n not in G.nodes:
			continue
		p_c, co_c, l_c, ch_c = _routing_service.plan_with_chargers(
			G, dropoff_node, goal_n, battery_after_b, MODE_EMPTY, drone_type, reserve_pct=plan_reserve, charger_nodes=charger_nodes,
			max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT, max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
		)
		if p_c and co_c and l_c < best_len_c:
			best_len_c = l_c
			best_c = (p_c, co_c, l_c, ch_c)
	if not best_c:
		if not base_node and not any(station_nodes):
			# No chargers defined — plan is still valid, no escape segment
			path_c, coords_c, len_c, chargers_c = [], [], 0.0, []
		else:
			logger.info("plan_order_trip: no escape after dropoff battery_after_b=%.1f", battery_after_b)
			return {"ok": False, "reason": NO_ESCAPE_AFTER_DROPOFF, "details": "no reachable charger after dropoff"}
	else:
		path_c, coords_c, len_c, chargers_c = best_c

	full_coords = list(coords_a)
	if coords_b:
		full_coords.extend(coords_b[1:])
	if coords_c:
		full_coords.extend(coords_c[1:])
	pickup_waypoint_count = len(coords_a)

	segments = [
		{"type": "to_pickup", "coords": coords_a, "length": len_a, "chargers": chargers_a},
		{"type": "to_dropoff", "coords": coords_b, "length": len_b, "chargers": chargers_b},
		{"type": "escape", "coords": coords_c, "length": len_c, "chargers": chargers_c},
	]
	chargers_used = list(dict.fromkeys(chargers_a + chargers_b + chargers_c))
	battery_after_escape = 100.0 if chargers_c else _routing_service.compute_battery_after(len_c, battery_after_b, MODE_EMPTY, drone_type)
	battery_plan = [
		{"at": "start", "battery": battery_pct},
		{"at": "after_pickup", "battery": battery_after_a},
		{"at": "after_dropoff", "battery": battery_after_b},
		{"at": "after_escape", "battery": battery_after_escape},
	]
	total_length = len_a + len_b + len_c

	return {
		"ok": True,
		"coords": full_coords,
		"segments": segments,
		"chargers_used": chargers_used,
		"battery_plan": battery_plan,
		"pickup_waypoint_count": pickup_waypoint_count,
		"route_length": total_length,
	}


def _plan_active_cargo_order_from_drone(
	drone: Dict[str, Any],
	order: Dict[str, Any],
	destination: Optional[Tuple[float, float]] = None,
	battery_pct: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
	"""Планирует активный грузовой дрон заказ из дрон."""
	current_pos = tuple(drone.get("pos") or ())
	if len(current_pos) != 2:
		return None
	target_dropoff = tuple(destination or order.get("end") or ())
	if len(target_dropoff) != 2:
		return None
	current_battery = float(battery_pct if battery_pct is not None else drone.get("battery", 100.0))
	plan_reserve = PLAN_RESERVE_PCT_TEST if STATE.get("battery_mode") == "test" else PLAN_RESERVE_PCT
	plan_reserve = max(plan_reserve, CARGO_RESERVE_AFTER_DELIVERY_PCT)
	if _is_drone_loaded(drone):
		coords_b, chargers_b, len_b = _plan_direct_air_route_with_stations(
			current_pos,
			target_dropoff,
			current_battery,
			MODE_LOADED,
			"cargo",
			plan_reserve,
		)
		if not coords_b:
			return None
		battery_after_b = 100.0 if chargers_b else _routing_service.compute_battery_after(len_b, current_battery, MODE_LOADED, "cargo")
		coords_c, chargers_c, len_c = _plan_air_escape_to_charger(
			target_dropoff,
			battery_after_b,
			"cargo",
			plan_reserve,
		)
		if not coords_c and _direct_charger_points():
			return None
		full_coords = list(coords_b)
		if coords_c:
			full_coords.extend(coords_c[1:])
		return {
			"ok": True,
			"coords": full_coords,
			"segments": [
				{"type": "to_dropoff", "coords": coords_b, "length": len_b, "chargers": chargers_b},
				{"type": "escape", "coords": coords_c or [], "length": len_c, "chargers": chargers_c},
			],
			"chargers_used": list(dict.fromkeys(chargers_b + chargers_c)),
			"battery_plan": [
				{"at": "start", "battery": current_battery},
				{"at": "after_dropoff", "battery": battery_after_b},
				{"at": "after_escape", "battery": 100.0 if chargers_c else battery_after_b},
			],
			"pickup_waypoint_count": 0,
			"route_length": len_b + len_c,
		}
	return _plan_cargo_direct_air_trip(
		current_pos,
		tuple(order.get("start") or current_pos),
		target_dropoff,
		current_battery,
		plan_reserve,
	)


def plan_via_base_if_needed(current: Tuple[float, float], end: Tuple[float, float], drone_type: str, battery_level: float):
	"""Планирует один сегмент с возможной зарядкой через метаграф зарядных станций."""
	if drone_type == "cargo":
		coords, _chargers, length = _plan_direct_air_route_with_stations(
			tuple(current),
			tuple(end),
			float(battery_level),
			MODE_EMPTY,
			drone_type,
			RESERVE_PCT,
		)
		return None, coords, length if coords else 0.0
	G = STATE.get("city_graph")
	city = STATE.get("city")
	if not city or G is None:
		return None, None, 0.0
	_routing_service.city_graphs[city] = G
	start_node = _find_graph_node_for_point(current, prefer_primary=True)
	end_node = _find_graph_node_for_point(end, prefer_primary=True)
	if not start_node or not end_node:
		return None, None, 0.0
	charger_nodes = STATE.get("charger_nodes") or {"base": None, "stations": []}
	path, coords, length, _ = _routing_service.plan_with_chargers(
		G, start_node, end_node, battery_level, MODE_EMPTY, drone_type, reserve_pct=RESERVE_PCT, charger_nodes=charger_nodes
	)
	return (path, coords, length) if coords else (None, None, 0.0)


def _handle_battery_depletion(drone_id: str, drone: Dict[str, Any]) -> bool:
	"""Обрабатывает полный или критический разряд батареи."""
	if drone.get("status") == "emergency_landed":
		return True
	current = tuple(drone.get("pos") or ())
	if len(current) != 2:
		return False
	if float(drone.get("battery", 0.0)) > EMERGENCY_LANDING_BATTERY_PCT:
		return False
	if _is_at_charge_location(current):
		return False
	order = _get_active_order_for_drone(drone_id)
	_remove_drone_from_charge_queues(drone_id)
	drone["battery"] = 0.0
	drone["speed_mps"] = 0.0
	drone["route"] = []
	drone["resume_route"] = []
	drone["target_idx"] = 0
	drone["status"] = "emergency_landed"
	drone["force_charge_after_route"] = False
	if order:
		order["status"] = "failed"
		order["failure_reason"] = "battery_depleted"
		_monitoring_service.fail_live_mission_for_order(
			str(order.get("id")),
			{"id": drone_id, **drone},
			reason="battery_depleted",
		)
		drone.pop("active_order_id", None)
	return True


def can_escape_after(point: Tuple[float, float], drone_type: str) -> bool:
	"""Проверяет, можно ли из точки достигнуть зарядной станции с учетом резерва."""
	G = STATE.get("city_graph")
	if G is None:
		return True
	charger_nodes = STATE.get("charger_nodes") or {"base": None, "stations": []}
	point_node = _find_graph_node_for_point(point, prefer_primary=True)
	if not point_node:
		return False
	base_node = charger_nodes.get("base")
	stations = charger_nodes.get("stations") or []
	for goal in [base_node] + [s for s in stations if s]:
		if goal is None or goal not in G.nodes:
			continue
		_, coords, _, _ = _routing_service.plan_with_chargers(
			G, point_node, goal, 80.0, MODE_EMPTY, drone_type, reserve_pct=RESERVE_PCT, charger_nodes=charger_nodes
		)
		if coords:
			return True
	return False


def simulate_step():
	# move drones along their routes, drain battery, reroute if blocked and avoid collisions
	"""Выполняет один шаг движения всех дронов и обновляет их статусы, батарею и телеметрию."""
	_routing_service.set_battery_mode(STATE.get("battery_mode", "reality"))
	city = STATE.get("city")
	G = STATE.get("city_graph")
	if not city or G is None:
		return
	for drone_id, drone in STATE["drones"].items():
		_sanitize_active_drone_state(drone_id, drone)
		if _process_service_stop(drone_id, drone):
			compute_link_quality(drone)
			_queue_live_telemetry(drone_id, {"id": drone_id, **drone})
			continue
		if _handle_battery_depletion(drone_id, drone):
			_queue_live_telemetry(drone_id, {"id": drone_id, **drone})
			continue
		if drone.get("status") == "avoidance":
			_avoidance_step(drone_id, drone)
			continue
		# Сначала: при заряде <=20% любой дрон обязан лететь на зарядку (жёсткое правило)
		if drone.get("battery", 100.0) <= FLY_TO_CHARGER_AT_PCT:
			if drone.get("status") in ("idle", "holding"):
				_save_route_for_return_if_on_order(drone_id, drone)
				drone["status"] = "low_battery"
				maybe_route_to_base_or_station(drone)
				logger.info("simulate_step: drone %s battery=%.1f%% -> routing to charger (idle/holding)", drone_id, drone.get("battery"))
			elif drone.get("status") == "enroute":
				_save_route_for_return_if_on_order(drone_id, drone)
				if (drone.get("route") or []) and drone.get("target_idx", 0) < len(drone.get("route", [])):
					# Уже едет по маршруту заказа — переключаем на зарядку
					drone["status"] = "low_battery"
					maybe_route_to_base_or_station(drone)
					logger.info("simulate_step: drone %s battery=%.1f%% -> routing to charger (was enroute), saved route for resume", drone_id, drone.get("battery"))
		# Движение по маршруту: не только enroute, но и поездка на зарядку
		if drone.get("status") not in ("enroute", "return_charge", "return_base", "low_battery", "zone_escape"):
			continue
		_mark_order_in_progress_if_started(drone_id, drone)
		route = drone.get("route") or []
		idx = drone.get("target_idx", 0)
		if not route or idx >= len(route):
			mark_order_completed_if_any(drone_id)
			if drone.get("status") != "charging":
				drone["status"] = "idle"
			# При заряде <=20% после завершения — сразу на зарядку
			if drone.get("battery", 100.0) <= FLY_TO_CHARGER_AT_PCT:
				maybe_route_to_base_or_station(drone)
			continue
		current = drone["pos"]
		target = route[idx]
		if is_point_in_any_zone(current):
			if _route_drone_out_of_no_fly_zone(drone_id, drone):
				continue
		# Для operator_area миссии не перепрокладываем через road graph:
		# там маршрут уже задан внутренними координатами зоны осмотра.
		if is_point_in_any_zone(target) and drone.get("mission_mode") != "operator_area":
			# attempt reroute
			active_order = _get_active_order_for_drone(drone_id)
			if active_order and drone.get("type") == "cargo":
				res = _plan_active_cargo_order_from_drone(drone, active_order)
				coords = res.get("coords") if res and res.get("ok") else None
			elif active_order and active_order.get("end"):
				_, coords, _ = plan_via_base_if_needed(current, tuple(active_order["end"]), drone["type"], drone["battery"])
			else:
				end = route[-1]
				_, coords, _ = plan_via_base_if_needed(current, end, drone["type"], drone["battery"])
			if coords:
				if active_order and drone.get("type") == "cargo":
					apply_midroute_charging(drone, list(coords))
				else:
					drone["route"] = coords
					drone["target_idx"] = 0
				continue
			else:
				drone["status"] = "holding"
				continue
		elif is_point_in_any_zone(target) and drone.get("mission_mode") == "operator_area":
			# Для operator_area не уводим маршрут на дороги: пропускаем заблокированную внутреннюю точку.
			drone["target_idx"] = idx + 1
			continue
		# Base speed affected by wind (simple model)
		speed_mps = _estimate_speed_mps()
		dist = haversine_m(current, target)
		if dist < speed_mps:
			# Arrive this tick
			drone["pos"] = target
			drone["speed_mps"] = round(max(0.0, dist), 2)
			active_order = _get_active_order_for_drone(drone_id)
			required_type = active_order.get("drone_type") if active_order else None
			if active_order and not required_type:
				required_type = map_order_to_drone_type(active_order.get("type", "delivery"))
			at_pickup = bool(
				active_order
				and active_order.get("start")
				and required_type == "cargo"
				and not active_order.get("pickup_serviced")
				and _is_near_point(tuple(target), tuple(active_order["start"]))
			)
			at_delivery = bool(
				active_order
				and active_order.get("end")
				and required_type == "cargo"
				and not active_order.get("delivery_serviced")
				and _is_near_point(tuple(target), tuple(active_order["end"]))
			)
			at_charger = is_at_any_station(target) or (STATE.get("base") and haversine_m(target, tuple(STATE["base"])) < STATION_NEAR_METERS)
			if at_delivery:
				active_order["delivery_serviced"] = True
				drone["post_delivery_route"] = list(route[idx + 1:])
				drone["route"] = list(route[: idx + 1])
				drone["target_idx"] = len(drone["route"])
				drone["waypoints_completed"] = drone.get("waypoints_completed", 0) + 1
				_begin_service_stop(drone, "delivery")
			elif at_pickup:
				active_order["pickup_serviced"] = True
				drone["target_idx"] = idx + 1
				drone["waypoints_completed"] = drone.get("waypoints_completed", 0) + 1
				_begin_service_stop(drone, "pickup")
			# Если прилетели на зарядку и впереди ещё точки — останавливаемся на зарядку, потом продолжаем маршрут
			elif at_charger and idx + 1 < len(route):
				drone["resume_route"] = list(route[idx + 1:])
				drone["route"] = list(route[: idx + 1])
				drone["target_idx"] = idx + 1
				drone["waypoints_completed"] = drone.get("waypoints_completed", 0) + 1
				assign_to_charger_queue(drone_id)
				drone["status"] = "charging"
			else:
				drone["target_idx"] = idx + 1
				drone["waypoints_completed"] = drone.get("waypoints_completed", 0) + 1
				if drone["target_idx"] >= len(route):
					if drone.get("force_charge_after_route"):
						drone["force_charge_after_route"] = False
						# Завершаем текущий этап area-миссии перед уходом на зарядку,
						# чтобы continuation мог подхватиться другим свободным оператором.
						mark_order_completed_if_any(drone_id)
						drone["status"] = "low_battery"
						maybe_route_to_base_or_station(drone)
						continue
					# Для маршрутов "на зарядку" не завершаем задачу в idle, пока не встанем в очередь зарядки.
					if at_charger or drone.get("status") in ("return_charge", "return_base", "low_battery"):
						if assign_to_charger_queue(drone_id):
							drone["status"] = "charging"
						else:
							# Если конечная точка получилась не рядом со станцией — перепрокладываем.
							maybe_route_to_base_or_station(drone)
					else:
						mark_order_completed_if_any(drone_id)
						if drone.get("status") != "charging":
							drone["status"] = "idle"
				else:
					# Сохраняем статус «на зарядку», если дрон едет к станции
					if drone.get("status") not in ("return_charge", "return_base", "low_battery"):
						drone["status"] = "enroute"
			battery_drain(drone, dist)
		else:
			# Move fractionally and apply basic collision avoidance
			next_pos = move_towards(current, target, speed_mps / dist)
			if will_collide(drone_id, next_pos):
				drone["status"] = "avoidance"
				drone["avoidance_ticks"] = drone.get("avoidance_ticks", 0) + 1
			else:
				drone["pos"] = next_pos
				drone["avoidance_ticks"] = 0
			drone["speed_mps"] = round(speed_mps, 2)
			battery_drain(drone, speed_mps)
		if _handle_battery_depletion(drone_id, drone):
			_queue_live_telemetry(drone_id, {"id": drone_id, **drone})
			continue
		# link quality estimation vs. nearest base or last strong point
		compute_link_quality(drone)
		_queue_live_telemetry(drone_id, {"id": drone_id, **drone})
		if drone.get("status") == "service_stop":
			continue
		# low battery: если ещё не едем на зарядку — сохраняем маршрут заказа и строим маршрут до станции
		if drone["battery"] <= FLY_TO_CHARGER_AT_PCT and drone.get("status") not in ("return_charge", "return_base", "low_battery"):
			if drone.get("status") == "enroute":
				_save_route_for_return_if_on_order(drone_id, drone)
			drone["status"] = "low_battery"
			maybe_route_to_base_or_station(drone)
			logger.info("simulate_step: drone %s battery=%.1f%% -> routing to charger (in-route check)", drone_id, drone.get("battery"))

		# Update ETA approximation (seconds) and readable remaining distance
		try:
			remaining = 0.0
			if drone.get("route") and drone.get("target_idx", 0) < len(drone["route"]):
				pos = drone.get("pos")
				tidx = drone.get("target_idx", 0)
				# distance to current target
				remaining += haversine_m(pos, drone["route"][tidx])
				for i in range(tidx, len(drone["route"]) - 1):
					remaining += haversine_m(drone["route"][i], drone["route"][i+1])
			drone["remaining_m"] = remaining
			drone["eta_s"] = int(remaining / max(0.1, speed_mps)) if remaining > 0 else 0
		except Exception:
			pass

	# charging progression & queue handling
	progress_charging()



def _avoidance_step(drone_id: str, drone: Dict[str, Any]) -> None:
	"""Выход из мёртвой блокировки «обход препятствия»: через 2 тика делаем боковой сдвиг."""
	ticks = drone.get("avoidance_ticks", 0) + 1
	drone["avoidance_ticks"] = ticks
	if ticks < 2:
		return
	route = drone.get("route") or []
	idx = drone.get("target_idx", 0)
	if not route or idx >= len(route):
		drone["status"] = "idle"
		drone["avoidance_ticks"] = 0
		return
	current = drone["pos"]
	target = route[idx]
	dist = haversine_m(current, target)
	if dist < 1.0:
		drone["avoidance_ticks"] = 0
		return
	# Не смещаем дрон в искусственную точку вне графа, чтобы не "слетал" с карты.
	# Делаем короткое удержание и продолжаем движение по исходному маршруту.
	drone["status"] = "enroute"
	drone["avoidance_ticks"] = 0


def will_collide(drone_id: str, next_pos: Tuple[float,float]) -> bool:
	"""Проверяет будущую столкновение."""
	for other_id, other in STATE["drones"].items():
		if other_id == drone_id:
			continue
		if haversine_m(next_pos, other.get("pos", next_pos)) < 8.0:  # 8 meters bubble
			return True
	return False


def _is_drone_loaded(drone: Dict[str, Any]) -> bool:
	"""Дрон загружен, если уже проехал не менее loaded_after_waypoint_count точек (едет к точке доставки)."""
	after = drone.get("loaded_after_waypoint_count", 0)
	completed = drone.get("waypoints_completed", 0)
	return completed >= after


def battery_drain(drone: Dict[str, Any], distance_m: float):
	# Single source of truth: RoutingService._get_drone_params. drain в %: distance_m / m_per_pct (без *100)
	"""Списывает заряд аккумулятора с учетом расстояния, типа дрона, груза и погодных условий."""
	p = _routing_service._get_drone_params(drone.get("type", "cargo"))
	loaded = _is_drone_loaded(drone)
	m_per_pct = p["loaded_m_per_pct"] if loaded else p["empty_m_per_pct"]
	drain = distance_m / m_per_pct if m_per_pct > 0 else 0.0
	drone["battery"] = max(0.0, drone["battery"] - drain)
	# temperature/mock telemetry drift
	drone["temp_c"] = float(drone.get("temp_c", 35.0)) + (0.02 * (distance_m/10.0))
	# record history sparsely
	try:
		hist = drone.get("history") or []
		pos = drone.get("pos")
		if pos and (not hist or haversine_m(tuple(hist[-1]), tuple(pos)) > 5.0):
			hist.append(tuple(pos))
			# cap history length
			if len(hist) > 1000:
				hist = hist[-1000:]
			drone["history"] = hist
	except Exception:
		pass


def _point_in_zone(point: Tuple[float, float], zone: Dict[str, Any]) -> bool:
	"""Обрабатывает точку в зону."""
	try:
		lat, lon = point
		zone_type = (zone.get("zone_type") or "rectangle").lower()
		if zone_type == "circle":
			center = (float(zone.get("center_lat")), float(zone.get("center_lon")))
			radius_m = float(zone.get("radius_m") or 0.0)
			return haversine_m((lat, lon), center) <= radius_m
		lat_min = min(float(zone["lat_min"]), float(zone["lat_max"]))
		lat_max = max(float(zone["lat_min"]), float(zone["lat_max"]))
		lon_min = min(float(zone["lon_min"]), float(zone["lon_max"]))
		lon_max = max(float(zone["lon_min"]), float(zone["lon_max"]))
		return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
	except Exception:
		return False


def _remaining_route_enters_zone(drone: Dict[str, Any]) -> bool:
	"""Проверяет, входит ли оставшийся маршрут в запретную зону."""
	route = drone.get("route") or []
	idx = max(0, int(drone.get("target_idx", 0)))
	for point in route[idx:]:
		if is_point_in_any_zone(tuple(point)):
			return True
	return False


def _find_shortest_escape_coords(current: Tuple[float, float], drone_type: str, battery: float) -> Tuple[Optional[List[Tuple[float, float]]], float]:
	"""Ищет кратчайший набор координат для выхода из запретной зоны."""
	if drone_type == "cargo":
		containing = [zone for zone in STATE.get("no_fly_zones", []) if _point_in_zone(current, zone)]
		if containing:
			best_route: Optional[List[Tuple[float, float]]] = None
			best_length = float("inf")
			for zone in containing:
				if (zone.get("zone_type") or "rectangle").lower() == "circle":
					center = (float(zone.get("center_lat")), float(zone.get("center_lon")))
					origin = center
					current_xy = _to_local_xy(current, origin)
					dist = math.hypot(current_xy[0], current_xy[1])
					if dist < 1.0:
						current_xy = (1.0, 0.0)
						dist = 1.0
					scale = (_zone_circle_radius(zone, AIR_ZONE_PADDING_M + 20.0)) / dist
					exit_xy = (current_xy[0] * scale, current_xy[1] * scale)
					exit_point = _from_local_xy(exit_xy, origin)
				else:
					center = _zone_center(zone)
					exit_point = move_towards(center, current, 1.4)
				route = _interpolate_air_segment(current, exit_point, step_m=AIR_ROUTE_STEP_M / 2.0)
				length = _route_length(route)
				if length < best_length:
					best_route = route
					best_length = length
			return best_route, (best_length if best_route else 0.0)
	G = STATE.get("city_graph")
	if G is None:
		return None, 0.0
	candidates: List[Tuple[float, Tuple[float, float]]] = []
	for _, attr in G.nodes(data=True):
		pos = attr.get("pos")
		if not pos or attr.get("weight") == float("inf"):
			continue
		pos = tuple(pos)
		if is_point_in_any_zone(pos):
			continue
		candidates.append((haversine_m(current, pos), pos))
	candidates.sort(key=lambda item: item[0])
	best_coords = None
	best_len = float("inf")
	for _, target in candidates[:40]:
		_, coords, length = plan_route_for(current, target, drone_type, battery, reserve_pct=0.0)
		if coords and length < best_len:
			best_coords = list(coords)
			best_len = float(length)
	return best_coords, (best_len if best_coords else 0.0)


def _route_drone_out_of_no_fly_zone(drone_id: str, drone: Dict[str, Any]) -> bool:
	"""Строит маршрут вывода дрона из запретной зоны."""
	current = tuple(drone.get("pos") or ())
	if len(current) != 2:
		return False
	escape_coords, _ = _find_shortest_escape_coords(current, drone.get("type", "cargo"), max(1.0, float(drone.get("battery", 100.0))))
	if not escape_coords:
		return False
	final_target = None
	active_order = _get_active_order_for_drone(drone_id)
	if active_order and active_order.get("end"):
		final_target = tuple(active_order.get("end"))
	elif drone.get("route"):
		final_target = tuple((drone.get("route") or [escape_coords[-1]])[-1])
	combined = list(escape_coords)
	if final_target and len(final_target) == 2 and not is_point_in_any_zone(final_target):
		if drone.get("type") == "cargo":
			if active_order:
				res = _plan_active_cargo_order_from_drone(
					{**drone, "pos": tuple(escape_coords[-1])},
					active_order,
					destination=final_target,
					battery_pct=max(1.0, float(drone.get("battery", 100.0))),
				)
				resume_coords = res.get("coords") if res and res.get("ok") else None
			else:
				resume_coords, _used, _len = _plan_direct_air_route_with_stations(
					tuple(escape_coords[-1]),
					final_target,
					max(1.0, float(drone.get("battery", 100.0))),
					MODE_EMPTY,
					drone.get("type", "cargo"),
					RESERVE_PCT,
				)
		else:
			_, resume_coords, _ = plan_via_base_if_needed(tuple(escape_coords[-1]), final_target, drone.get("type", "cargo"), max(1.0, float(drone.get("battery", 100.0))))
		if resume_coords:
			combined.extend(list(resume_coords)[1:])
	drone["route"] = combined
	drone["target_idx"] = _first_meaningful_target_idx(current, combined)
	drone["status"] = "zone_escape"
	return True


def _reroute_active_drones_after_zone_change() -> None:
	"""Перестраивает маршруты активных дронов после изменения запретных зон."""
	for drone_id, drone in STATE.get("drones", {}).items():
		status = drone.get("status")
		if status not in ("enroute", "return_charge", "return_base", "low_battery", "zone_escape"):
			continue
		current = tuple(drone.get("pos") or ())
		if len(current) != 2:
			continue
		if is_point_in_any_zone(current):
			_route_drone_out_of_no_fly_zone(drone_id, drone)
			continue
		if not _remaining_route_enters_zone(drone):
			continue
		if status in ("return_charge", "return_base", "low_battery"):
			maybe_route_to_base_or_station(drone)
			continue
		active_order = _get_active_order_for_drone(drone_id)
		if active_order and active_order.get("end"):
			if drone.get("type") == "cargo":
				res = _plan_active_cargo_order_from_drone(drone, active_order)
				coords = res.get("coords") if res and res.get("ok") else None
			else:
				_, coords, _ = plan_via_base_if_needed(current, tuple(active_order["end"]), drone["type"], drone["battery"])
			if coords:
				apply_midroute_charging(drone, coords)
		elif drone.get("route"):
			if drone.get("type") == "cargo":
				coords, _used, _len = _plan_direct_air_route_with_stations(
					current,
					tuple((drone.get("route") or [current])[-1]),
					float(drone.get("battery", 100.0)),
					MODE_EMPTY,
					drone["type"],
					RESERVE_PCT,
				)
			else:
				_, coords, _ = plan_via_base_if_needed(current, tuple((drone.get("route") or [current])[-1]), drone["type"], drone["battery"])
			if coords:
				drone["route"] = coords
				drone["target_idx"] = _first_meaningful_target_idx(current, coords)


def is_point_in_any_zone(point: Tuple[float, float]) -> bool:
	"""Проверяет точку в любой зону."""
	for z in STATE["no_fly_zones"]:
		if _point_in_zone(point, z):
			return True
	return False

def is_at_any_station(point: Tuple[float, float]) -> bool:
	"""Проверяет, находится ли точка рядом с любой зарядной станцией."""
	try:
		for s in STATE.get("stations", []):
			if haversine_m(point, tuple(s)) < STATION_NEAR_METERS:
				return True
		return False
	except Exception:
		return False


def _is_at_charge_location(point: Tuple[float, float]) -> bool:
	"""Проверяет, находится ли точка рядом с любой точкой зарядки."""
	try:
		if is_at_any_station(point):
			return True
		base = STATE.get("base")
		return bool(base) and haversine_m(tuple(point), tuple(base)) < STATION_NEAR_METERS
	except Exception:
		return False

def nearest_station_index(point: Tuple[float,float]) -> Optional[int]:
    """Находит индекс ближайшей зарядной станции."""
    try:
        best_i = None
        best_d = float('inf')
        for i, s in enumerate(STATE.get("stations", [])):
            d = haversine_m(point, tuple(s))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i
    except Exception:
        return None

def maybe_route_to_base(drone: Dict[str, Any]):
	"""При необходимости строит маршрут возвращения на базу."""
	base = STATE.get("base")
	if not base:
		return
	try:
		battery = max(0.5, float(drone.get("battery", 10.0)))
		reserve = 0.0 if battery <= 15.0 else RESERVE_PCT
		_, coords, _ = plan_route_for(
			tuple(drone.get("pos", base)), tuple(base), drone["type"], battery, reserve_pct=reserve
		)
		if coords:
			drone["route"] = coords
			drone["target_idx"] = 0
			if drone["battery"] <= 5.0:
				drone["status"] = "low_battery"
			else:
				drone["status"] = "return_base"
	except Exception:
		pass

def _save_route_for_return_if_on_order(drone_id: str, drone: Dict[str, Any]) -> None:
	"""Перед уходом на зарядку сохраняем маршрут заказа, чтобы после зарядки вернуть дрон к заданию."""
	if drone.get("saved_route_for_charge") is not None:
		return
	for o in STATE.get("orders", []):
		if o.get("drone_id") == drone_id and o.get("status") in ("assigned", "in_progress", "waiting_continuation"):
			drone["saved_order_id"] = o.get("id")
			drone["active_order_id"] = o.get("id")
			route = drone.get("route") or []
			if route:
				drone["saved_route_for_charge"] = list(route)
				drone["saved_target_idx"] = int(drone.get("target_idx", 0))
			return


def _rebuild_active_order_route(drone_id: str, drone: Dict[str, Any], order: Dict[str, Any]) -> bool:
	"""Пересобирает маршрут активного заказа, если дрон завис в holding без валидного пути."""
	try:
		required_type = order.get("drone_type") or map_order_to_drone_type(order.get("type", "delivery"))
		pos = tuple(drone.get("pos", (0.0, 0.0)))
		battery = max(1.0, float(drone.get("battery", 100.0)))
		if required_type == "cargo":
			res = _plan_active_cargo_order_from_drone(drone, order, battery_pct=battery)
			if not res:
				res = plan_order_trip(pos, order["start"], order["end"], "cargo", battery)
		else:
			res = _run_plan_for_order(pos, battery, order, required_type)
	except Exception:
		logger.exception("_rebuild_active_order_route: replanning failed for drone %s order %s", drone_id, order.get("id"))
		return False

	if not res or not res.get("ok") or not res.get("coords"):
		return False

	coords = list(res["coords"])
	if required_type == "cargo":
		apply_midroute_charging(drone, coords)
	else:
		drone["route"] = coords
		drone["resume_route"] = []
		drone["target_idx"] = _first_meaningful_target_idx(pos, coords)
		drone["status"] = "enroute" if coords else "idle"
	drone["active_order_id"] = order.get("id")
	logger.info("_rebuild_active_order_route: restored route for drone %s order %s (%s waypoints)", drone_id, order.get("id"), len(coords))
	return True


def _restore_route_after_charging(drone: Dict[str, Any]) -> bool:
	"""После зарядки восстанавливаем маршрут заказа, если он был сохранён. Возвращает True, если восстановили."""
	# Выход из charging: маршрут восстановлен — статус станет enroute в конце (исполняемое состояние, не «зарядка»).
	if drone.get("saved_route_for_charge") is not None and drone.get("type") != "cargo":
		saved = drone.get("saved_route_for_charge") or []
		drone["route"] = list(saved)
		sidx = int(drone.get("saved_target_idx", 0))
		drone["target_idx"] = max(0, min(sidx, max(0, len(saved) - 1)))
		drone["status"] = "enroute"
		order_id = drone.pop("saved_order_id", None)
		if order_id:
			drone["active_order_id"] = order_id
		drone.pop("saved_route_for_charge", None)
		drone.pop("saved_target_idx", None)
		logger.info("_restore_route_after_charging: restored route for order %s (%s waypoints from idx %s)", order_id, len(saved), drone.get("target_idx"))
		return True

	# Fallback: если маршрут не сохранился (например, был пустой), пересобираем маршрут по заказу.
	order_id = drone.get("saved_order_id")
	if not order_id:
		return False
	order = next((o for o in STATE.get("orders", []) if o.get("id") == order_id), None)
	if not order or order.get("status") not in ("assigned", "in_progress", "waiting_continuation"):
		# If order is gone or not assigned, don't replan.
		return False

	try:
		required_type = order.get("drone_type") or map_order_to_drone_type(order.get("type", "delivery"))
		pos = tuple(drone.get("pos", (0.0, 0.0)))
		if required_type == "cargo":
			res = _plan_active_cargo_order_from_drone({**drone, "pos": pos}, order, battery_pct=100.0)
		elif required_type == "operator" and (order.get("remaining_waypoints") or (order.get("handover_point") and order.get("rest_waypoints"))):
			res = plan_operator_continuation_trip({"pos": pos, "battery": 100.0, "type": "operator"}, order)
		elif required_type == "operator" and order.get("area_polygon"):
			res, _cont = plan_operator_area_trip({"pos": pos, "battery": 100.0, "type": "operator"}, order)
		elif required_type == "operator":
			res = plan_operator_point_trip(pos, order["end"], "operator", 100.0)
		elif required_type == "cleaner":
			res = plan_operator_point_trip(pos, order["end"], "cleaner", 100.0)
		else:
			res = None
	except Exception:
		logger.exception("_restore_route_after_charging: replanning failed")
		return False

	if not res or not res.get("ok") or not res.get("coords"):
		rt = order.get("drone_type") or map_order_to_drone_type(order.get("type", "delivery"))
		if rt == "operator" and order.get("area_polygon") and not (order.get("remaining_waypoints") or []):
			logger.warning(
				"_restore_route_after_charging: replan empty for operator area order %s (no remaining waypoints)",
				order.get("id"),
			)
		return False

	drone["route"] = list(res["coords"])
	drone["target_idx"] = 0
	# После восстановления миссии обязательно возвращаем дрона в исполняемый статус.
	drone["status"] = "enroute"
	drone["active_order_id"] = order_id
	drone.pop("saved_order_id", None)
	drone.pop("saved_route_for_charge", None)
	drone.pop("saved_target_idx", None)
	logger.info("_restore_route_after_charging: replanned route for order %s (%s waypoints)", order_id, len(drone["route"]))
	return True


def _has_assigned_order_for_drone(drone_id: str) -> bool:
	"""Проверяет, есть ли у дрона назначенный заказ."""
	for o in STATE.get("orders", []):
		if o.get("drone_id") == drone_id and o.get("status") in ("assigned", "in_progress", "waiting_continuation"):
			return True
	return False


def _resume_after_charge_or_hold(drone_id: str, drone: Dict[str, Any]) -> None:
	# Дрон выходит из зарядки: сначала убираем из очередей (иначе UI и логика видят «в зарядке» при battery=100).
	"""Возобновляет движение дрона после зарядки или ожидания."""
	_remove_drone_from_charge_queues(drone_id)
	# Восстановление маршрута после промежуточной зарядки:
	# 1) сначала заранее сохранённый хвост маршрута,
	# 2) затем пересборка по сохранённому order_id,
	# 3) если задание ещё назначено, не уходим в idle до явного завершения.
	resume = drone.get("resume_route") or []
	if resume:
		drone["route"] = list(resume)
		drone["resume_route"] = []
		drone["target_idx"] = 0
		# После зарядки при восстановленном маршруте переводим в активный исполняемый статус.
		drone["status"] = "enroute"
		if drone.get("saved_order_id"):
			drone["active_order_id"] = drone.get("saved_order_id")
		return
	if _restore_route_after_charging(drone):
		return
	# Нет resume_route и не удалось replan: operator area без хвоста — закрываем order после зарядки.
	if _try_close_order_if_no_flight_after_charge(drone_id, drone):
		if not _get_active_order_for_drone(drone_id):
			drone["status"] = "idle"
		return
	if _has_assigned_order_for_drone(drone_id):
		# Есть активный заказ, но route пока не восстановлен — удерживаем, не уходим в idle.
		drone["status"] = "holding"
		return
	drone.pop("active_order_id", None)
	drone["status"] = "idle"


def maybe_route_to_base_or_station(drone: Dict[str, Any]):
	"""Строит маршрут до ближайшей зарядки. При низком заряде — через другие станции (plan_with_chargers)."""
	chargers: List[Tuple[float, float]] = []
	if STATE.get("base"):
		chargers.append(tuple(STATE["base"]))
	chargers += [tuple(s) for s in STATE.get("stations", []) if isinstance(s, (list, tuple)) and len(s) == 2]
	if not chargers:
		return
	pos = tuple(drone.get("pos", chargers[0]))
	best = None
	bestd = float("inf")
	for c in chargers:
		d = haversine_m(pos, c)
		if d < bestd:
			bestd = d
			best = c
	if not best:
		return
	battery = max(0.5, float(drone.get("battery", 10.0)))
	reserve = 0.0 if battery <= FLY_TO_CHARGER_AT_PCT else RESERVE_PCT
	if drone.get("type") == "cargo":
		coords, _chargers, _length = _plan_air_escape_to_charger(pos, battery, drone["type"], reserve)
		if coords:
			drone["route"] = coords
			drone["target_idx"] = _first_meaningful_target_idx(pos, coords)
			drone["status"] = "return_charge"
			return
	_, coords, _ = plan_route_for(pos, best, drone["type"], battery, reserve_pct=reserve)
	# Если прямой путь недостижим — строим маршрут через станции зарядки (plan_with_chargers)
	if not coords:
		G = STATE.get("city_graph")
		charger_nodes = STATE.get("charger_nodes") or {"base": None, "stations": []}
		if G and (charger_nodes.get("base") is not None or (charger_nodes.get("stations") or [])):
			start_node = _find_graph_node_for_point(pos, prefer_primary=True)
			goal_node = None
			base_coords = STATE.get("base")
			if base_coords and tuple(base_coords) == best:
				goal_node = charger_nodes.get("base")
			else:
				for i, s in enumerate(STATE.get("stations") or []):
					if isinstance(s, (list, tuple)) and len(s) == 2 and tuple(s) == best:
						st = charger_nodes.get("stations") or []
						if i < len(st):
							goal_node = st[i]
						break
			if start_node and goal_node and goal_node in G.nodes:
				path_p, coords_c, _len, _ch = _routing_service.plan_with_chargers(
					G, start_node, goal_node, battery, MODE_EMPTY, drone["type"], reserve_pct=reserve,
					charger_nodes=charger_nodes, max_segment_battery_pct=MAX_SEGMENT_BATTERY_PCT,
					max_battery_pct_to_reach_charger=MAX_BATTERY_PCT_TO_REACH_CHARGER,
				)
				if path_p and coords_c:
					coords = coords_c
					logger.info("maybe_route_to_base_or_station: route to charger via plan_with_chargers (battery=%.1f%%)", battery)
		if not coords and G and best:
			max_range = _routing_service.max_reachable_distance(battery, MODE_EMPTY, drone["type"], reserve_pct=reserve)
			_, coords, _ = _routing_service.plan_direct_path(G, pos, best, max_range * 2.0)
	if coords:
		drone["route"] = coords
		drone["target_idx"] = _first_meaningful_target_idx(pos, coords)
		drone["status"] = "return_charge"

def assign_to_charger_queue(drone_id: str) -> bool:
    """Назначает дрон в очередь зарядной станции."""
    d = STATE["drones"].get(drone_id)
    if not d:
        return False
    pos = tuple(d.get("pos", (0,0)))
    # База: зарядка на месте (как раньше)
    base_close = STATE.get("base") and haversine_m(pos, tuple(STATE["base"])) < STATION_NEAR_METERS
    if base_close:
        q = STATE.get("base_queue") or {"charging": [], "queue": [], "capacity": 2}
        if drone_id not in q["charging"] and drone_id not in q["queue"]:
            if len(q["charging"]) < q.get("capacity", 2):
                q["charging"].append(drone_id)
            else:
                q["queue"].append(drone_id)
        STATE["base_queue"] = q
        return True
    # Станции зарядки: смена аккумулятора (запас 20 заряженных)
    idx = nearest_station_index(pos)
    if idx is None:
        return False
    # Ставим в очередь только если дрон действительно рядом со станцией.
    station_pos = None
    stations = STATE.get("stations") or []
    if 0 <= idx < len(stations):
        station_pos = tuple(stations[idx])
    if not station_pos or haversine_m(pos, station_pos) >= STATION_NEAR_METERS:
        return False
    key = str(idx)
    sq = (STATE.get("station_queues") or {}).get(key)
    if not sq:
        sq = {"charged_batteries": STATION_CHARGED_BATTERIES_MAX, "charging_queue": [], "queue": []}
    if "charged_batteries" in sq:
        if drone_id in sq.get("queue", []):
            return True
        if sq.get("charged_batteries", 0) > 0:
            sq["charged_batteries"] -= 1
            sq.setdefault("charging_queue", []).append(STATION_BATTERY_CHARGE_TICKS)
            d["battery"] = 100.0
            _resume_after_charge_or_hold(drone_id, d)
        else:
            sq.setdefault("queue", []).append(drone_id)
            d["status"] = "charging"
    else:
        if drone_id not in sq.get("charging", []) and drone_id not in sq.get("queue", []):
            if len(sq.get("charging", [])) < sq.get("capacity", 2):
                sq.setdefault("charging", []).append(drone_id)
            else:
                sq.setdefault("queue", []).append(drone_id)
    STATE.setdefault("station_queues", {})[key] = sq
    return True

def progress_charging():
    # base
    """Обновляет процесс зарядки дронов и состояние очередей на зарядных станциях."""
    bq = STATE.get("base_queue") or {"charging": [], "queue": [], "capacity": 2}
    done = []
    for did in list(bq.get("charging", [])):
        d = STATE["drones"].get(did)
        if not d:
            continue
        d["battery"] = min(100.0, float(d.get("battery", 0.0)) + 4.0)  # 4% per tick
        if d["battery"] >= CHARGE_COMPLETE_PCT:
            d["battery"] = 100.0
            done.append(did)
            _resume_after_charge_or_hold(did, d)
    for did in done:
        if did in bq["charging"]:
            bq["charging"].remove(did)
    # promote from queue
    while len(bq["charging"]) < bq.get("capacity", 2) and bq["queue"]:
        bq["charging"].append(bq["queue"].pop(0))
    STATE["base_queue"] = bq
    # Станции зарядки: смена аккумуляторов (тик зарядки в очереди, затем выдача ожидающим)
    sqs = STATE.get("station_queues") or {}
    for key, sq in sqs.items():
        if "charged_batteries" in sq:
            cq = sq.get("charging_queue") or []
            new_cq = []
            for t in cq:
                t -= 1
                if t <= 0:
                    sq["charged_batteries"] = min(STATION_CHARGED_BATTERIES_MAX, sq.get("charged_batteries", 0) + 1)
                else:
                    new_cq.append(t)
            sq["charging_queue"] = new_cq
            served = []
            for did in list(sq.get("queue", [])):
                if sq.get("charged_batteries", 0) <= 0:
                    break
                d = STATE["drones"].get(did)
                if not d:
                    served.append(did)
                    continue
                sq["charged_batteries"] -= 1
                sq.setdefault("charging_queue", []).append(STATION_BATTERY_CHARGE_TICKS)
                d["battery"] = 100.0
                _resume_after_charge_or_hold(did, d)
                served.append(did)
            for did in served:
                if did in sq.get("queue", []):
                    sq["queue"].remove(did)
            # Пока нет готовых аккумуляторов — дроны в очереди заряжаются постепенно (как на базе), чтобы не застревать на 19%
            for did in list(sq.get("queue", [])):
                d = STATE["drones"].get(did)
                if not d:
                    continue
                d["battery"] = min(100.0, float(d.get("battery", 0.0)) + 4.0)
                if d["battery"] >= CHARGE_COMPLETE_PCT:
                    d["battery"] = 100.0
                    if did in sq.get("queue", []):
                        sq["queue"].remove(did)
                    _resume_after_charge_or_hold(did, d)
        else:
            done = []
            for did in list(sq.get("charging", [])):
                d = STATE["drones"].get(did)
                if not d:
                    continue
                d["battery"] = min(100.0, float(d.get("battery", 0.0)) + 4.0)
                if d["battery"] >= CHARGE_COMPLETE_PCT:
                    d["battery"] = 100.0
                    done.append(did)
                    _resume_after_charge_or_hold(did, d)
            for did in done:
                if did in sq.get("charging", []):
                    sq["charging"].remove(did)
            while len(sq.get("charging", [])) < sq.get("capacity", 2) and sq.get("queue", []):
                sq["charging"].append(sq["queue"].pop(0))
        sqs[key] = sq
    STATE["station_queues"] = sqs
    # Нормализация: battery >= порога, status charging — недопустимо оставаться в charging (очередь/статус/маршрут).
    for _did, _d in list((STATE.get("drones") or {}).items()):
        _force_exit_charging_if_complete(_did, _d)

def _area_chain_has_pending(root_order_id: str, exclude_order_id: Optional[str] = None) -> bool:
	"""Проверяет, остались ли незавершенные заказы в цепочке работ по области."""
	for o in STATE.get("orders", []):
		if o.get("id") == exclude_order_id:
			continue
		if _operator_area_root_order_id(o) != root_order_id:
			continue
		if o.get("status") in ("queued", "assigned", "in_progress", "waiting_continuation"):
			return True
	return False


def _mark_area_root_completed_if_ready(root_order_id: str) -> None:
	"""Завершает корневую задачу области, если все дочерние этапы выполнены."""
	root = next((o for o in STATE.get("orders", []) if o.get("id") == root_order_id), None)
	if not root:
		return
	if _area_chain_has_pending(root_order_id, exclude_order_id=root_order_id):
		root["status"] = "in_progress"
		return
	remaining = root.get("remaining_waypoints") or []
	if remaining:
		root["status"] = "in_progress"
		return
	root["status"] = "completed"
	root.pop("drone_id", None)


def mark_order_completed_if_any(drone_id: str):
	# Финализация заказа: выполняем только при реальном окончании миссии, не при промежуточной зарядке.
	"""Проверяет достижение целевой точки и завершает заказ или этап миссии при необходимости."""
	drone = STATE.get("drones", {}).get(drone_id)
	if not drone:
		return
	route = drone.get("route") or []
	if route and int(drone.get("target_idx", 0)) < len(route):
		return
	# Во время зарядки не завершаем заказ (промежуточная остановка), кроме случая «зарядка завершена»:
	# battery >= порога и дрон снимаем с charging-очередей — иначе in_progress зависает навсегда.
	if drone.get("status") == "charging":
		if float(drone.get("battery", 0.0)) < CHARGE_COMPLETE_PCT:
			return
		_remove_drone_from_charge_queues(drone_id)
	elif drone.get("status") in ("return_charge", "return_base", "low_battery"):
		return

	order = _get_active_order_for_drone(drone_id)
	if not order:
		if drone.get("status") == "charging" and float(drone.get("battery", 0.0)) >= CHARGE_COMPLETE_PCT:
			drone["status"] = "idle"
		return

	required_type = order.get("drone_type") or map_order_to_drone_type(order.get("type", "delivery"))
	if required_type == "operator" and (order.get("area_polygon") or order.get("remaining_waypoints")):
		# area mission completed только при полном исчерпании remaining_waypoints и отсутствии continuation.
		root_id = _operator_area_root_order_id(order)
		rem = order.get("remaining_waypoints") or []
		# После зарядки (battery полная) пустой route при ненулевом remaining — не handover, а восстановление миссии.
		if rem and float(drone.get("battery", 0.0)) >= CHARGE_COMPLETE_PCT and not (drone.get("route") or []):
			if _restore_route_after_charging(drone):
				return
			drone["status"] = "holding"
			return
		if order.get("has_continuation") or rem:
			order["status"] = "waiting_continuation"
			order.pop("drone_id", None)
		else:
			order["status"] = "completed"
			order.pop("drone_id", None)
		_mark_area_root_completed_if_ready(root_id)
	else:
		# cargo/operator-point/cleaner завершаем только в конце основного маршрута миссии.
		order["status"] = "completed"
		order.pop("drone_id", None)
		_monitoring_service.complete_live_mission_for_order(order.get("id"), {"id": drone_id, **drone})

	# Чистим активный контекст дрона после завершения/передачи.
	drone.pop("active_order_id", None)
	drone.pop("mission_mode", None)
	drone["force_charge_after_route"] = False
	# После завершения/передачи заказа дрон не должен оставаться в «зарядке» или holding без заказа.
	if not _get_active_order_for_drone(drone_id) and drone.get("status") in ("charging", "holding"):
		drone["status"] = "idle"

	# Доп. защита: закрываем возможные дубликаты этого же order id в активных статусах.
	for o in STATE.get("orders", []):
		if o.get("id") != order.get("id"):
			continue
		if o is order:
			continue
		if o.get("status") in ("assigned", "in_progress", "waiting_continuation") and o.get("drone_id") == drone_id:
			o["status"] = "cancelled"
			o.pop("drone_id", None)

# Telemetry helpers
def compute_link_quality(drone: Dict[str, Any]):
    """Вычисляет связь качество."""
    base = STATE.get("base")
    if not base:
        drone["link_quality"] = 0.7  # default medium
        return
    try:
        d = haversine_m(tuple(drone.get("pos", base)), tuple(base))
        # simple path loss model: quality 1.0 within 300m, decays to 0.1 at 5km, never below 0.05
        if d <= 300:
            q = 1.0
        elif d >= 5000:
            q = 0.1
        else:
            q = max(0.05, 1.0 - (d-300)/(5000-300))
        # adjust for wind
        wind = float(STATE.get("weather",{}).get("wind_mps", 3.0))
        q *= max(0.4, 1.0 - wind*0.02)
        drone["link_quality"] = round(float(q), 2)
    except Exception:
        drone["link_quality"] = 0.5

def ensure_base_drones():
    """Проверяет и подготавливает базу дроны."""
    base = STATE.get("base")
    inv = STATE.get("inventory") or {}
    if not base or not isinstance(inv, dict):
        return
    # If there are already drones, do not auto-spawn duplicates
    if STATE.get("drones"):
        return
    total = sum(max(0, int(v)) for v in inv.values())
    if total <= 0:
        return
    for typ, cnt in inv.items():
        try:
            n = max(0, int(cnt))
        except Exception:
            n = 0
        for _ in range(n):
            spawn_drone(typ, pos=tuple(base), battery=100.0)

def _next_drone_id(drone_type: str) -> str:
    """Генерирует имя дрона по типу: грузовой1, операторский1, мойщик1 и т.д."""
    type_names = {"cargo": "грузовой", "operator": "операторский", "cleaner": "мойщик"}
    name_ru = type_names.get(drone_type, "дрон")
    numbers = []
    for k in STATE.get("drones", {}):
        if k == name_ru or k.startswith(name_ru):
            suf = k[len(name_ru):].strip()
            if suf.isdigit():
                numbers.append(int(suf))
    next_num = max(numbers, default=0) + 1
    return f"{name_ru}{next_num}"


def spawn_drone(drone_type: str, pos: Tuple[float, float], battery: float = 100.0) -> str:
    """Создает дрон дрон."""
    did = _next_drone_id(drone_type)
    STATE["drones"][did] = {
        "pos": pos,
        "type": drone_type,
        "battery": float(battery),
        "alt_m": 60.0,
        "route": [],
        "target_idx": 0,
        "status": "idle",
    }
    return did

def spawn_drone_from_inventory(
    pref_type: str,
    order_id: Optional[str] = None,
    caller: str = "unknown",
    spawn_pos: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """Создает дрон из доступного инвентаря."""
    base = STATE.get("base")
    inv = STATE.get("inventory") or {}
    if not spawn_pos and not base:
        return None
    # Запрещаем спавн без реального запаса inventory нужного типа.
    cnt_raw = inv.get(pref_type, 0)
    try:
        cnt = int(cnt_raw)
    except Exception:
        cnt = 0
    if cnt <= 0:
        logger.info(
            "spawn_drone_from_inventory: blocked type=%s order=%s caller=%s inventory_before=%s",
            pref_type, order_id, caller, cnt_raw,
        )
        return None
    before = cnt
    deploy_pos = tuple(spawn_pos) if spawn_pos else tuple(base)
    did = spawn_drone(pref_type, pos=deploy_pos, battery=100.0)
    inv[pref_type] = cnt - 1
    STATE["inventory"] = inv
    logger.info(
        "spawn_drone_from_inventory: spawned drone=%s type=%s order=%s caller=%s pos=%s inventory_before=%s inventory_after=%s",
        did, pref_type, order_id, caller, deploy_pos, before, inv.get(pref_type, 0),
    )
    return did

def _first_meaningful_target_idx(current_pos: Tuple[float, float], route: List[Tuple[float, float]]) -> int:
    """Пропускает стартовые точки, которые совпадают с текущей позицией дрона."""
    if not route:
        return 0
    idx = 0
    while idx < len(route):
        pt = tuple(route[idx])
        try:
            if haversine_m(tuple(current_pos), pt) >= 3.0:
                break
        except Exception:
            break
        idx += 1
    return min(idx, max(0, len(route) - 1))


# Split a full route into two: up to first charger (station/base), then remainder to resume after charging
def apply_midroute_charging(drone: Dict[str, Any], full_coords: List[Tuple[float, float]]):
    """Добавляет промежуточную зарядку в маршрут при необходимости."""
    try:
        if not isinstance(full_coords, list) or len(full_coords) < 2:
            drone["route"] = list(full_coords or [])
            drone["resume_route"] = []
            drone["target_idx"] = _first_meaningful_target_idx(tuple(drone.get("pos", (0.0, 0.0))), drone["route"])
            drone["status"] = "enroute" if drone["route"] else "idle"
            return
        base = STATE.get("base")
        split_idx = None
        traveled_m = 0.0
        for i in range(1, len(full_coords)):
            traveled_m += haversine_m(tuple(full_coords[i - 1]), tuple(full_coords[i]))
            pt = tuple(full_coords[i])
            at_station = is_at_any_station(pt)
            at_base = bool(base) and haversine_m(pt, tuple(base)) < STATION_NEAR_METERS
            if at_station or at_base:
                # Не считаем вылет с базы "промежуточной зарядкой", пока дрон фактически не покинул стартовую зону.
                if i < len(full_coords) - 1 and traveled_m >= max(80.0, STATION_NEAR_METERS * 2):
                    split_idx = i
                    break
        if split_idx is not None:
            drone["route"] = list(full_coords[:split_idx+1])
            drone["resume_route"] = list(full_coords[split_idx+1:])
            drone["target_idx"] = _first_meaningful_target_idx(tuple(drone.get("pos", (0.0, 0.0))), drone["route"])
            drone["status"] = "enroute"
        else:
            drone["route"] = list(full_coords)
            drone["resume_route"] = []
            drone["target_idx"] = _first_meaningful_target_idx(tuple(drone.get("pos", (0.0, 0.0))), drone["route"])
            drone["status"] = "enroute"
    except Exception:
        drone["route"] = list(full_coords or [])
        drone["resume_route"] = []
        drone["target_idx"] = _first_meaningful_target_idx(tuple(drone.get("pos", (0.0, 0.0))), drone["route"])
        drone["status"] = "enroute" if drone["route"] else "idle"

# Persistence
async def persist_state():
    """Сохраняет состояние."""
    if not _redis:
        return
    try:
        data = {
            "city": STATE.get("city"),
            "orders": STATE.get("orders", []),
            "drones": STATE.get("drones", {}),
            "no_fly_zones": STATE.get("no_fly_zones", []),
            "weather": STATE.get("weather", {}),
            "base": STATE.get("base"),
            "inventory": STATE.get("inventory", {}),
            "stations": STATE.get("stations", []),
            "station_queues": STATE.get("station_queues", {}),
            "battery_mode": STATE.get("battery_mode", "reality"),
            "drone_type": STATE.get("drone_type"),
            "ts": datetime.utcnow().isoformat(),
        }
        _redis.set("drone_planner:state", json.dumps(data))
    except Exception:
        logger.exception("persist_state failed")

async def restore_state():
    """Восстанавливает состояние."""
    if not _redis:
        return
    try:
        raw = _redis.get("drone_planner:state")
        if not raw:
            return
        data = json.loads(raw)
        STATE["city"] = data.get("city")
        STATE["orders"] = data.get("orders", [])
        STATE["drones"] = data.get("drones", {})
        STATE["no_fly_zones"] = data.get("no_fly_zones", [])
        STATE["weather"] = data.get("weather", {"wind_mps": 3.0})
        STATE["base"] = tuple(data.get("base")) if data.get("base") else None
        STATE["inventory"] = data.get("inventory", {})
        STATE["stations"] = data.get("stations", [])
        if data.get("station_queues"):
            sq = data["station_queues"]
            for k, v in list(sq.items()):
                if isinstance(v, dict) and "charged_batteries" not in v:
                    sq[k] = {"charged_batteries": STATION_CHARGED_BATTERIES_MAX, "charging_queue": [], "queue": []}
            STATE["station_queues"] = sq
        elif STATE["stations"]:
            STATE["station_queues"] = {
                str(i): {"charged_batteries": STATION_CHARGED_BATTERIES_MAX, "charging_queue": [], "queue": []}
                for i in range(len(STATE["stations"]))
            }
        if data.get("battery_mode") in ("reality", "test"):
            STATE["battery_mode"] = data["battery_mode"]
            _routing_service.set_battery_mode(data["battery_mode"])
        if data.get("drone_type") in ("cargo", "operator", "cleaner"):
            STATE["drone_type"] = data["drone_type"]
        if STATE["city"]:
            try:
                await rebuild_graph_with_zones()
            except Exception:
                logger.exception("Failed to rebuild graph on restore")
        if not STATE.get("stations") or len(STATE.get("stations") or []) < MIN_CITY_STATION_COUNT:
            STATE["stations"] = _generate_city_stations(MIN_CITY_STATION_COUNT)
            STATE["station_queues"] = {
                str(i): {"charged_batteries": STATION_CHARGED_BATTERIES_MAX, "charging_queue": [], "queue": []}
                for i in range(len(STATE["stations"]))
            }
        refresh_charger_nodes()
    except Exception:
        logger.exception("restore_state failed")


def move_towards(a: Tuple[float, float], b: Tuple[float, float], frac: float) -> Tuple[float, float]:
	"""Вычисляет промежуточную точку движения от текущей позиции к цели."""
	frac = max(0.0, min(1.0, frac))
	return (a[0] + (b[0]-a[0]) * frac, a[1] + (b[1]-a[1]) * frac)


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	"""Вычисляет расстояние между двумя географическими координатами в метрах."""
	R = 6371000.0
	lat1 = math.radians(a[0]) ; lat2 = math.radians(b[0])
	dlat = lat2 - lat1
	dlon = math.radians(b[1] - a[1])
	sa = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
	c = 2.0 * math.atan2(math.sqrt(sa), math.sqrt(1.0-sa))
	return R * c

# Run helper for uvicorn: python -m uvicorn api_server:app --reload
