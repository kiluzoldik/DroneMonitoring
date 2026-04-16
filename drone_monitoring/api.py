from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from drone_monitoring.schemas import CancelMissionRequest, LaunchMissionRequest, LiveMissionRequest, ProcessMissionRequest, SyntheticMissionRequest, UpdateMissionDestinationRequest
from drone_monitoring.service import MissionMonitoringService


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _read_static_page(name: str) -> HTMLResponse:
    path = STATIC_DIR / name
    try:
        return HTMLResponse(path.read_text(encoding="utf-8"))
    except Exception:
        return HTMLResponse(f"<h3>Page {name} not found.</h3>", status_code=404)


def register_monitoring_routes(app: FastAPI, monitor: MissionMonitoringService, runtime: Dict[str, Any]) -> None:
    @app.get("/monitor")
    async def monitoring_page():
        return _read_static_page("mission_monitor.html")

    @app.get("/mission-setup")
    async def mission_setup_page():
        return _read_static_page("mission_setup.html")

    @app.post("/api/missions/synthetic")
    async def create_synthetic_mission(body: SyntheticMissionRequest):
        if body.start_live:
            mission = await asyncio.to_thread(monitor.create_demo_live_mission, body.title, body.city, body.seed)
            await runtime["ensure_demo_environment"](mission["city"])
            order_id = await runtime["enqueue_live_order"](mission, "cargo")
            await asyncio.to_thread(monitor.register_live_order, mission["id"], order_id, {"source": "synthetic"})
            mission = await asyncio.to_thread(monitor.get_mission, mission["id"])
        else:
            mission = await asyncio.to_thread(monitor.create_synthetic_mission, body.title, body.city, body.seed)
        return {"ok": True, "mission": mission}

    @app.post("/api/missions/live")
    async def create_live_mission(body: LiveMissionRequest):
        await runtime["ensure_demo_environment"](body.city)
        exact_start = (body.start.lat, body.start.lon)
        exact_delivery = (body.delivery.lat, body.delivery.lon)
        no_fly_zones = [
            {
                "center": (zone.center.lat, zone.center.lon),
                "radius_m": zone.radius_m,
                "name": f"Запретная зона {idx + 1}",
            }
            for idx, zone in enumerate(body.no_fly_zones or [])
        ]
        if not no_fly_zones and body.no_fly_center is not None:
            no_fly_zones = [{
                "center": (body.no_fly_center.lat, body.no_fly_center.lon),
                "radius_m": body.no_fly_radius_m,
                "name": "Запретная зона 1",
            }]
        mission = await asyncio.to_thread(
            monitor.create_live_mission_draft,
            body.title,
            body.city,
            exact_start,
            exact_delivery,
            ((body.no_fly_center.lat, body.no_fly_center.lon) if body.no_fly_center is not None else None),
            body.no_fly_radius_m,
            no_fly_zones,
            body.drone_type,
        )
        for zone in no_fly_zones:
            await runtime["upsert_runtime_no_fly_zone"](
                {
                    "zone_type": "circle",
                    "center_lat": zone["center"][0],
                    "center_lon": zone["center"][1],
                    "radius_m": zone["radius_m"],
                    "mission_id": mission["id"],
                }
            )
        order_id = await runtime["enqueue_live_order"](mission, body.drone_type)
        await asyncio.to_thread(monitor.register_live_order, mission["id"], order_id, {"source": "live_setup"})
        return {"ok": True, "mission": await asyncio.to_thread(monitor.get_mission, mission["id"])}

    @app.post("/api/missions/{mission_id}/launch")
    async def launch_existing_mission(mission_id: str, body: LaunchMissionRequest):
        mission = await asyncio.to_thread(monitor.get_mission, mission_id)
        if mission.get("order_id"):
            return JSONResponse(status_code=400, content={"ok": False, "error": "Mission is already launched"})
        await runtime["ensure_demo_environment"](mission["city"])
        order_id = await runtime["enqueue_live_order"](mission, body.drone_type)
        await asyncio.to_thread(monitor.register_live_order, mission_id, order_id, {"source": "manual_launch"})
        return {"ok": True, "mission": await asyncio.to_thread(monitor.get_mission, mission_id)}

    @app.get("/api/missions")
    async def list_missions(scope: str = "active"):
        return {"missions": await asyncio.to_thread(monitor.list_missions, scope)}

    @app.get("/api/missions/{mission_id}")
    async def get_mission(mission_id: str):
        return {"mission": await asyncio.to_thread(monitor.get_mission, mission_id)}

    @app.get("/api/missions/{mission_id}/track/raw")
    async def get_raw_track(mission_id: str):
        return await asyncio.to_thread(monitor.get_raw_track, mission_id)

    @app.get("/api/missions/{mission_id}/track/processed")
    async def get_processed_track(mission_id: str):
        return await asyncio.to_thread(monitor.get_processed_track, mission_id)

    @app.get("/api/missions/{mission_id}/geozones")
    async def get_geozones(mission_id: str):
        geozones = await asyncio.to_thread(monitor.get_geozones, mission_id)
        return {"mission_id": mission_id, "geozones": geozones}

    @app.post("/api/missions/{mission_id}/process")
    async def process_mission(mission_id: str, body: ProcessMissionRequest):
        result = await asyncio.to_thread(monitor.run_processing, mission_id, body.model_dump())
        return {"ok": True, **result}

    @app.post("/api/missions/{mission_id}/destination")
    async def update_mission_destination(mission_id: str, body: UpdateMissionDestinationRequest):
        mission = await asyncio.to_thread(monitor.get_mission, mission_id)
        exact_destination = (body.destination.lat, body.destination.lon)
        updated = await asyncio.to_thread(monitor.update_mission_destination, mission_id, exact_destination)
        await runtime["update_runtime_destination"](mission, exact_destination)
        return {"ok": True, "mission": updated}

    @app.post("/api/missions/{mission_id}/cancel")
    async def cancel_mission(mission_id: str, body: CancelMissionRequest | None = None):
        await runtime["cancel_runtime_mission"](mission_id)
        mission = await asyncio.to_thread(monitor.cancel_mission, mission_id, body.reason if body else None)
        return {"ok": True, "mission": mission}

    @app.get("/api/missions/{mission_id}/events")
    async def get_events(mission_id: str):
        return await asyncio.to_thread(monitor.get_events, mission_id)

    @app.get("/api/missions/{mission_id}/metrics")
    async def get_metrics(mission_id: str):
        return await asyncio.to_thread(monitor.get_metrics, mission_id)

    @app.get("/api/drones/{drone_id}/monitoring")
    async def get_drone_monitoring(drone_id: str):
        runtime_drone = runtime["state"].get("drones", {}).get(drone_id)
        return await asyncio.to_thread(monitor.get_drone_monitoring, drone_id, runtime_drone)
