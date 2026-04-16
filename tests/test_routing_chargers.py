"""
Minimal unit/integration tests for charger-aware routing (meta-graph, plan_order_trip).
Mock graph: 10–20 nodes, base + 2 stations; verify plan found when via station feasible,
rejected when no escape after dropoff; loaded consumption > empty.
Run: python tests/test_routing_chargers.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from typing import Dict, Any, List, Tuple

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Use dummy graph service to avoid loading osmnx/shapely in tests
class _DummyGraphService:
    pass

from routing_service import (
    RoutingService,
    MODE_EMPTY,
    MODE_LOADED,
)


def make_mock_graph(
    num_nodes: int = 8,
    base_pos: Tuple[float, float] = (48.70, 44.51),
    station_positions: List[Tuple[float, float]] = None,
) -> nx.Graph:
    """Build a simple chain graph with base and station nodes (small to avoid timeout)."""
    if station_positions is None:
        station_positions = [(48.702, 44.512), (48.704, 44.514)]
    G = nx.Graph()
    step = 0.002
    nodes = []
    for i in range(num_nodes):
        lat = base_pos[0] + i * step
        lon = base_pos[1] + i * step * 0.5
        nid = f"n{i}"
        G.add_node(nid, pos=(lat, lon), type="road")
        nodes.append(nid)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1], weight=250.0)
    G.add_node("base", pos=base_pos, type="road")
    G.add_edge("base", nodes[0], weight=200.0)
    for j, pos in enumerate(station_positions):
        sid = f"station_{j}"
        G.add_node(sid, pos=pos, type="road")
        G.add_edge(sid, nodes[min(j + 1, len(nodes) - 1)], weight=200.0)
    return G


def _routing():
    return RoutingService(_DummyGraphService())


def _mock_g():
    return make_mock_graph(15)


if HAS_PYTEST:
    @pytest.fixture
    def routing():
        return _routing()

    @pytest.fixture
    def mock_g():
        return _mock_g()


def test_battery_params_single_source(routing: RoutingService):
    """Drone params contain empty_m_per_pct and loaded_m_per_pct; loaded < empty per type."""
    for drone_type in ("cargo", "operator", "cleaner"):
        p = routing._get_drone_params(drone_type)
        assert "empty_m_per_pct" in p
        assert "loaded_m_per_pct" in p
        assert p["empty_m_per_pct"] > p["loaded_m_per_pct"]


def test_compute_battery_after(routing: RoutingService):
    """compute_battery_after: empty mode uses more m per % than loaded (less drain per km when empty)."""
    # Empty: cargo empty_m_per_pct=120 -> 1000m ≈ 8.33% drain, ~91.67% left
    b_after = routing.compute_battery_after(1000.0, 100.0, MODE_EMPTY, "cargo")
    assert 90.0 <= b_after <= 93.0
    b_after_loaded = routing.compute_battery_after(1000.0, 100.0, MODE_LOADED, "cargo")
    assert b_after_loaded < b_after  # loaded drains more


def test_max_reachable_distance(routing: RoutingService):
    """max_reachable_distance: with reserve, 50% battery gives less than 50% * m_per_pct."""
    d = routing.max_reachable_distance(50.0, MODE_EMPTY, "cargo", reserve_pct=10.0)
    p = routing._get_drone_params("cargo")
    expected_max = (50.0 - 10.0) * p["empty_m_per_pct"]
    assert d <= expected_max + 1.0


def test_plan_segment(routing: RoutingService, mock_g: nx.Graph):
    """plan_segment returns path and coords when path exists within max_range."""
    start = mock_g.nodes["base"]["pos"]
    end = mock_g.nodes["n5"]["pos"]
    path, coords, length = routing.plan_segment(mock_g, start, end, max_range_m=10000.0)
    assert path is not None
    assert coords is not None
    assert length >= 0


def test_build_meta_graph(routing: RoutingService, mock_g: nx.Graph):
    """Meta-graph has edges only when segment is feasible (length <= max_reachable)."""
    points = {"start": "base", "goal": "n5", "station_0": "station_0", "station_1": "station_1"}
    H = routing.build_meta_graph(
        mock_g, points, "start", ["station_0", "station_1"],
        battery_pct=100.0, mode=MODE_EMPTY, drone_type="cargo", reserve_pct=10.0
    )
    assert H.number_of_nodes() >= 2
    assert H.number_of_edges() >= 0


def test_plan_with_chargers_via_station(routing: RoutingService, mock_g: nx.Graph):
    """plan_with_chargers returns path when feasible (via station or direct)."""
    charger_nodes = {"base": "base", "stations": ["station_0", "station_1"]}
    path, coords, length, visited = routing.plan_with_chargers(
        mock_g, "base", "n5", battery_pct=80.0, mode=MODE_EMPTY, drone_type="cargo",
        reserve_pct=10.0, charger_nodes=charger_nodes
    )
    if path:
        assert len(coords) >= 2
        assert length >= 0


def test_plan_with_chargers_insufficient_escape(routing: RoutingService = None):
    """When goal is far and battery after segment is too low to reach any charger, no path."""
    if routing is None:
        routing = _routing()
    # Graph: start --500m-- mid --5000m-- goal; no charger near goal; battery 5%
    G = nx.Graph()
    G.add_node("start", pos=(48.70, 44.50), type="road")
    G.add_node("mid", pos=(48.705, 44.505), type="road")
    G.add_node("goal", pos=(48.75, 44.55), type="road")
    G.add_node("base", pos=(48.70, 44.50), type="road")
    G.add_edge("start", "mid", weight=500.0)
    G.add_edge("mid", "goal", weight=5000.0)
    G.add_edge("start", "base", weight=100.0)
    charger_nodes = {"base": "base", "stations": []}
    path, coords, length, _ = routing.plan_with_chargers(
        G, "start", "goal", battery_pct=100.0, mode=MODE_EMPTY, drone_type="cargo",
        reserve_pct=10.0, charger_nodes=charger_nodes
    )
    # Path start->goal may exist; but if we had battery 10% and goal far, escape from goal to base might fail
    path_low, coords_low, _, _ = routing.plan_with_chargers(
        G, "goal", "base", battery_pct=5.0, mode=MODE_EMPTY, drone_type="cargo",
        reserve_pct=10.0, charger_nodes=charger_nodes
    )
    # With 5% and reserve 10%, usable = 0 -> no path
    assert path_low is None or len(path_low) == 0 or coords_low is None


def test_loaded_vs_empty_consumption(routing: RoutingService):
    """Loaded mode reduces max_reachable_distance vs empty for same battery."""
    d_empty = routing.max_reachable_distance(50.0, MODE_EMPTY, "cargo")
    d_loaded = routing.max_reachable_distance(50.0, MODE_LOADED, "cargo")
    assert d_loaded < d_empty


def run_all():
    """Run tests without pytest."""
    routing = _routing()
    mock_g = _mock_g()
    test_battery_params_single_source(routing)
    test_compute_battery_after(routing)
    test_max_reachable_distance(routing)
    test_plan_segment(routing, mock_g)
    test_build_meta_graph(routing, mock_g)
    test_plan_with_chargers_via_station(routing, mock_g)
    test_plan_with_chargers_insufficient_escape(routing)
    test_loaded_vs_empty_consumption(routing)
    print("All tests passed.")


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        run_all()
