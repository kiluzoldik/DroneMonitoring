import networkx as nx
import numpy as np
from collections import deque
import logging
from typing import Dict, List, Optional, Tuple, Any

# Modes for battery consumption: empty (no cargo) vs loaded (with cargo)
MODE_EMPTY = "empty"
MODE_LOADED = "loaded"

# Default reserve and charger arrival thresholds (80% usable / 20% reserve)
DEFAULT_RESERVE_PCT = 20.0
DEFAULT_CHARGER_ARRIVAL_MIN_PCT = 5.0


BATTERY_MODE_REALITY = "reality"
BATTERY_MODE_TEST = "test"
TEST_RANGE_DIVISOR = 8  # в тесте дальность в 8 раз меньше — маршруты почти всегда через зарядки


class RoutingService:
    def __init__(self, graph_service):
        self.graph_service = graph_service
        self.city_graphs = {}
        self.progress_callbacks = []
        self.active_routes = {}
        self.battery_mode = BATTERY_MODE_REALITY
        self.logger = logging.getLogger(__name__)

    def set_battery_mode(self, mode: str) -> None:
        """Режим расхода: 'reality' — реальные данные, 'test' — укороченная дальность для тестов маршрутов."""
        self.battery_mode = mode if mode in (BATTERY_MODE_REALITY, BATTERY_MODE_TEST) else BATTERY_MODE_REALITY

    def add_progress_callback(self, callback):
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, stage, percentage, message=""):
        for callback in self.progress_callbacks:
            callback(stage, percentage, message)
    
    def plan_routes(self, city_name, points, drone_type="cargo", battery_level=100):
        self._update_progress("route", 0, f"Планирование для {len(points)} дронов")
        
        try:
            if city_name not in self.city_graphs:
                raise ValueError(f"Граф для города '{city_name}' не загружен")
            
            G = self.city_graphs[city_name]
            if len(G.nodes) == 0:
                raise ValueError(f"Граф города '{city_name}' пуст")
            
            drone_params = self._get_drone_params(drone_type)
            max_range = drone_params['battery_range'] * (battery_level / 100)
            
            self.logger.info(f"Планирование маршрутов для {len(points)} дронов типа {drone_type}, батарея: {battery_level}%, макс. дальность: {max_range:.0f}м")
            
            routes = []
            successful_routes = 0
            
            for i, (start, end) in enumerate(points):
                try:
                    progress = int((i / len(points)) * 80)
                    self._update_progress("route", progress, f"Маршрут {i+1}/{len(points)}")
                    
                    self.logger.info(f"Поиск маршрута {i+1}: {start} → {end}")
                    
                    start_node = self._find_nearest_node(G, start)
                    end_node = self._find_nearest_node(G, end)
                    
                    if not start_node:
                        self.logger.warning(f"Не найден стартовый узел для точки {start}")
                        continue
                    
                    if not end_node:
                        self.logger.warning(f"Не найден конечный узел для точки {end}")
                        continue
                    
                    if start_node == end_node:
                        self.logger.warning(f"Стартовая и конечная точки совпадают: {start_node}")
                        # Попробуем найти ближайший соседний узел для создания реального маршрута
                        neighbors = list(G.neighbors(start_node))
                        if neighbors:
                            # Используем первого соседа как промежуточную точку
                            intermediate_node = neighbors[0]
                            path = [start_node, intermediate_node, start_node]
                            coords = [G.nodes[node]['pos'] for node in path]
                            length = self._calculate_path_length(G, path)
                            routes.append((path, length, coords))
                            successful_routes += 1
                            self.logger.info(f"✓ Создан циклический маршрут: {length:.1f}м")
                        else:
                            # Если нет соседей, создаем минимальный маршрут
                            coords = [G.nodes[start_node]['pos']]
                            routes.append(([start_node], 0.0, coords))
                            successful_routes += 1
                            self.logger.info("✓ Создан минимальный маршрут (точка)")
                        continue
                    
                    path = self._find_safe_path(G, start_node, end_node, max_range)
                    if path and len(path) > 1:
                        coords = [G.nodes[node]['pos'] for node in path]
                        length = self._calculate_path_length(G, path)
                        
                        if length <= max_range:
                            routes.append((path, length, coords))
                            successful_routes += 1
                            self.logger.info(f"✓ Маршрут {i+1} найден: {length:.1f}м, {len(path)} точек")
                        else:
                            self.logger.warning(f"Маршрут {i+1} слишком длинный: {length:.1f}м > {max_range:.1f}м")
                    else:
                        self.logger.warning(f"Не удалось найти маршрут {i+1}")
                        
                except Exception as e:
                    self.logger.error(f"Ошибка планирования маршрута {i+1}: {e}")
                    continue
            
            self._update_progress("route", 100, f"Построено {successful_routes}/{len(points)} маршрутов")
            self.logger.info(f"Планирование завершено: {successful_routes}/{len(points)} успешных маршрутов")
            return routes
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка планирования маршрутов: {e}")
            self._update_progress("error", 0, f"Ошибка планирования: {str(e)}")
            raise

    def plan_direct_path(self, G, start_coords, end_coords, max_range, waypoints=None):
        """Планирование маршрута от двери до двери: добавляем временные узлы у точек."""
        try:
            if G is None or len(G.nodes) == 0:
                return None, None, 0.0

            # Find nearest graph nodes to connect temporary points
            start_nearest = self._find_nearest_node(G, start_coords)
            end_nearest = self._find_nearest_node(G, end_coords)
            if start_nearest is None or end_nearest is None:
                return None, None, 0.0

            # Create temp nodes
            tmp_start = ('tmp_start', id(start_coords))
            tmp_end = ('tmp_end', id(end_coords))

            # Ensure unique keys not colliding with existing nodes
            while tmp_start in G.nodes:
                tmp_start = (tmp_start[0], tmp_start[1] + 1)
            while tmp_end in G.nodes:
                tmp_end = (tmp_end[0], tmp_end[1] + 1)

            # Add temp nodes and connect to nearest nodes with realistic weights
            G.add_node(tmp_start, pos=start_coords, type='temp')
            G.add_node(tmp_end, pos=end_coords, type='temp')

            def _approx_dist(a, b):
                # approximate meters between lat/lon pairs
                lat_diff = (a[0] - b[0]) * 111000.0
                avg_lat = (a[0] + b[0]) / 2.0
                lon_diff = (a[1] - b[1]) * 111000.0 * np.cos(np.radians(avg_lat))
                return float(np.sqrt(lat_diff**2 + lon_diff**2))

            # Connect to several nearest neighbors to avoid being blocked by zones
            candidates_start = []
            candidates_end = []
            for node, attr in G.nodes(data=True):
                pos = attr.get('pos')
                if not pos:
                    continue
                if attr.get('weight', 0) == float('inf'):
                    continue
                candidates_start.append((node, _approx_dist(start_coords, pos)))
                candidates_end.append((node, _approx_dist(end_coords, pos)))

            candidates_start.sort(key=lambda x: x[1])
            candidates_end.sort(key=lambda x: x[1])
            k = 5
            for node, dist in candidates_start[:k]:
                G.add_edge(tmp_start, node, weight=max(1.0, dist), type='temp_connection')
            for node, dist in candidates_end[:k]:
                G.add_edge(tmp_end, node, weight=max(1.0, dist), type='temp_connection')

            # Build waypoint chain
            sequence = [tmp_start]
            waypoints = waypoints or []
            temp_nodes = []
            for wp in waypoints:
                node = ('tmp_wp', id(wp))
                while node in G.nodes:
                    node = (node[0], node[1] + 1)
                G.add_node(node, pos=wp, type='temp')
                for n, dist in sorted(candidates_start, key=lambda x: x[1])[:k]:
                    G.add_edge(node, n, weight=max(1.0, _approx_dist(wp, G.nodes[n]['pos'])), type='temp_connection')
                sequence.append(node)
                temp_nodes.append(node)
            sequence.append(tmp_end)

            # Find path through sequence
            full_path = []
            for i in range(len(sequence)-1):
                seg = self._find_safe_path(G, sequence[i], sequence[i+1], max_range)
                if not seg:
                    full_path = None
                    break
                if i > 0:
                    seg = seg[1:]
                full_path.extend(seg)

            path = full_path

            if not path:
                # Cleanup before return
                try:
                    if tmp_start in G:
                        G.remove_node(tmp_start)
                    if tmp_end in G:
                        G.remove_node(tmp_end)
                    for n in temp_nodes:
                        if n in G:
                            G.remove_node(n)
                except Exception:
                    pass
                return None, None, 0.0

            # Build coords and length while temp nodes are still in G
            coords = [G.nodes[n]['pos'] for n in path]
            length = self._calculate_path_length(G, path)

            # Cleanup temp nodes/edges
            try:
                if tmp_start in G:
                    G.remove_node(tmp_start)
                if tmp_end in G:
                    G.remove_node(tmp_end)
                for n in temp_nodes:
                    if n in G:
                        G.remove_node(n)
            except Exception:
                pass

            return path, coords, length
        except Exception as e:
            self.logger.error(f"Ошибка door-to-door планирования: {e}")
            return None, None, 0.0
    
    def emergency_landing(self, drone_id, current_pos, obstacle_pos):
        emergency_path = self._calculate_emergency_path(current_pos, obstacle_pos)
        return emergency_path
    
    def _find_safe_path(self, G, start, end, max_range):
        """Поиск безопасного пути с использованием различных алгоритмов"""
        try:
            # Проверяем существование узлов
            if start not in G.nodes or end not in G.nodes:
                self.logger.warning(f"Узлы {start} или {end} не найдены в графе")
                return None
            
            # Если старт и финиш совпадают
            if start == end:
                self.logger.debug("Старт и финиш совпадают")
                return [start]
            
            # Проверяем связность
            if not nx.has_path(G, start, end):
                self.logger.debug(f"Нет пути между узлами {start} и {end}")
                # Попробуем найти ближайший доступный узел к конечной точке
                alternative_end = self._find_nearest_reachable_node(G, start, end)
                if alternative_end and alternative_end != start:
                    self.logger.info(f"Используем альтернативную конечную точку: {alternative_end}")
                    end = alternative_end
                else:
                    return None
            
            # Пробуем разные алгоритмы поиска пути
            algorithms = [
                ('astar', lambda: nx.astar_path(G, start, end, weight='weight')),
                ('dijkstra', lambda: nx.dijkstra_path(G, start, end, weight='weight')),
                ('shortest', lambda: nx.shortest_path(G, start, end, weight='weight'))
            ]
            
            best_path = None
            best_length = float('inf')
            
            feasible_candidates = []
            for algo_name, algo_func in algorithms:
                try:
                    path = algo_func()
                    if path and len(path) > 1:
                        path_length = self._calculate_path_length(G, path)
                        self.logger.debug(f"Алгоритм {algo_name}: {len(path)} точек, {path_length:.1f}м")

                        if path_length <= max_range:
                            feasible_candidates.append((path_length, path, algo_name))
                        elif path_length < best_length:
                            best_path = path
                            best_length = path_length
                            
                except Exception as e:
                    self.logger.debug(f"Алгоритм {algo_name} не сработал: {e}")
                    continue

            if feasible_candidates:
                feasible_candidates.sort(key=lambda item: item[0])
                self.logger.debug(
                    "Выбран кратчайший безопасный путь алгоритмом %s: %.1fм",
                    feasible_candidates[0][2], feasible_candidates[0][0]
                )
                return feasible_candidates[0][1]

            # Если лучший найденный путь длиннее max_range — считаем, что на одном заряде сегмент недостижим
            # (пусть вышестоящий код попробует строить маршрут через станции зарядки).
            if best_path and best_length > max_range:
                self.logger.debug(
                    "Лучший путь %.1fм превышает лимит %.1fм — считаем сегмент недостижимым на одном заряде",
                    best_length, max_range,
                )
                best_path = None

            if not best_path:
                self.logger.debug(f"Не удалось найти подходящий путь между {start} и {end}")
                return None

            # best_path есть и укладывается в max_range
            return best_path

        except Exception as e:
            self.logger.error(f"Ошибка поиска пути: {e}")
            return None
    
    def _find_nearest_reachable_node(self, G, start, target):
        """Поиск ближайшего достижимого узла к целевой точке"""
        try:
            target_pos = G.nodes[target]['pos']
            reachable_nodes = nx.node_connected_component(G, start)
            
            min_distance = float('inf')
            nearest_node = None
            
            for node in reachable_nodes:
                if node != start:
                    node_pos = G.nodes[node]['pos']
                    distance = np.sqrt((target_pos[0] - node_pos[0])**2 + (target_pos[1] - node_pos[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_node = node
            
            return nearest_node
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска ближайшего достижимого узла: {e}")
            return None
    
    def _calculate_emergency_path(self, current_pos, obstacle_pos):
        return [current_pos, (current_pos[0] + 0.001, current_pos[1] + 0.001)]
    
    def _find_nearest_node(self, G, point):
        """Поиск ближайшего узла к заданной точке"""
        try:
            if not point or len(point) != 2:
                self.logger.warning(f"Некорректная точка: {point}")
                return None
            
            min_dist = float('inf')
            nearest_node = None
            
            # Используем пространственный индекс для более точного поиска
            # Сначала найдем узлы в радиусе ~1км от точки
            search_radius = 0.01  # примерно 1км в градусах
            
            lat, lon = point
            
            # Создаем список узлов для проверки
            nodes_to_check = []
            for node, attr in G.nodes(data=True):
                try:
                    if 'pos' not in attr:
                        continue
                    
                    node_pos = attr['pos']
                    if len(node_pos) != 2:
                        continue
                    
                    node_lat, node_lon = node_pos
                    
                    # Проверяем, находится ли узел в радиусе поиска
                    if (abs(node_lat - lat) <= search_radius and 
                        abs(node_lon - lon) <= search_radius):
                        nodes_to_check.append((node, node_pos))
                        
                except Exception as e:
                    self.logger.debug(f"Ошибка обработки узла {node}: {e}")
                    continue
            
            # Если в радиусе нет узлов, расширяем поиск
            if not nodes_to_check:
                self.logger.warning(f"Нет узлов в радиусе {search_radius}° от точки {point}, расширяем поиск")
                search_radius = 0.1  # примерно 10км
                
                for node, attr in G.nodes(data=True):
                    try:
                        if 'pos' not in attr:
                            continue
                        
                        node_pos = attr['pos']
                        if len(node_pos) != 2:
                            continue
                        
                        node_lat, node_lon = node_pos
                        
                        if (abs(node_lat - lat) <= search_radius and 
                            abs(node_lon - lon) <= search_radius):
                            nodes_to_check.append((node, node_pos))
                            
                    except Exception as e:
                        continue
            
            # Если все еще нет узлов, ищем среди всех узлов (но ограничиваем количество)
            if not nodes_to_check:
                self.logger.warning(f"Нет узлов в расширенном радиусе, ищем среди всех узлов")
                max_nodes = min(5000, len(G.nodes))
                nodes_checked = 0
                
                for node, attr in G.nodes(data=True):
                    try:
                        if 'pos' not in attr:
                            continue
                        
                        node_pos = attr['pos']
                        if len(node_pos) != 2:
                            continue
                        
                        nodes_to_check.append((node, node_pos))
                        nodes_checked += 1
                        
                        if nodes_checked >= max_nodes:
                            break
                            
                    except Exception as e:
                        continue
            
            # Находим ближайший узел среди отобранных
            for node, node_pos in nodes_to_check:
                try:
                    # Вычисляем расстояние в градусах
                    dist = np.sqrt((lat - node_pos[0])**2 + (lon - node_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
                        
                except Exception as e:
                    self.logger.debug(f"Ошибка вычисления расстояния для узла {node}: {e}")
                    continue
            
            if nearest_node:
                # Преобразуем расстояние в метры для логирования (debug — при многих станциях не спамим)
                distance_meters = min_dist * 111000  # приблизительно
                self.logger.debug(f"Найден ближайший узел {nearest_node} на расстоянии {distance_meters:.1f}м от {point}")
            else:
                self.logger.error(f"Не удалось найти ближайший узел для точки {point}")
            
            return nearest_node
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска ближайшего узла: {e}")
            return None
    
    def _calculate_path_length(self, G, path):
        """Вычисление длины пути в метрах"""
        try:
            if not path or len(path) < 2:
                return 0.0
            
            length = 0.0
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                
                try:
                    if G.has_edge(node1, node2):
                        # Используем вес ребра если он есть
                        edge_weight = G[node1][node2].get('weight', 0)
                        if edge_weight > 0:
                            length += edge_weight
                        else:
                            # Вычисляем расстояние между узлами
                            length += self._calculate_node_distance(G, node1, node2)
                    else:
                        # Вычисляем расстояние между несвязанными узлами
                        length += self._calculate_node_distance(G, node1, node2)
                        
                except Exception as e:
                    self.logger.debug(f"Ошибка вычисления расстояния между {node1} и {node2}: {e}")
                    continue
            
            return length
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления длины пути: {e}")
            return 0.0
    
    def _calculate_node_distance(self, G, node1, node2):
        """Вычисление расстояния между двумя узлами в метрах"""
        try:
            pos1 = G.nodes[node1]['pos']
            pos2 = G.nodes[node2]['pos']
            
            # Вычисляем расстояние в градусах
            lat_diff = pos1[0] - pos2[0]
            lon_diff = pos1[1] - pos2[1]
            
            # Приблизительное преобразование в метры
            # 1 градус широты ≈ 111 км, 1 градус долготы зависит от широты
            lat_meters = lat_diff * 111000  # широта
            avg_lat = (pos1[0] + pos2[0]) / 2
            lon_meters = lon_diff * 111000 * np.cos(np.radians(avg_lat))  # долгота
            
            distance = np.sqrt(lat_meters**2 + lon_meters**2)
            return distance
            
        except Exception as e:
            self.logger.debug(f"Ошибка вычисления расстояния между узлами: {e}")
            return 1000.0  # Возвращаем большое значение по умолчанию

    def _approx_point_distance_m(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        try:
            lat_meters = (float(a[0]) - float(b[0])) * 111000.0
            avg_lat = (float(a[0]) + float(b[0])) / 2.0
            lon_meters = (float(a[1]) - float(b[1])) * 111000.0 * max(0.01, np.cos(np.radians(avg_lat)))
            return float(np.sqrt(lat_meters ** 2 + lon_meters ** 2))
        except Exception:
            return float("inf")
    
    def _get_drone_params(self, drone_type: str) -> Dict[str, Any]:
        """Параметры по реальным данным расхода (Вт·ч/км, время полёта).
        empty_m_per_pct: метров пути на 1% заряда (пустой). Меньше = выше расход.
        Грузовой: 150–400 Wh/km, 15–40 мин → ~12 км пустой, ~8 км с грузом.
        Операторский: 40–120 Wh/km, 20–40 мин → ~20 км / ~15 км.
        Сервисный: 30–80 Wh/km, 25–60 мин → ~28 км / ~20 км.
        """
        params = {
            "cargo": {
                "battery_range": 20000,
                "empty_m_per_pct": 200.0,   # demo-профиль: до ~20 км на 100%
                "loaded_m_per_pct": 140.0,  # demo-профиль: рабочий сегмент с резервом ~11 км
                "reserve_pct": DEFAULT_RESERVE_PCT,
                "charger_arrival_min_pct": DEFAULT_CHARGER_ARRIVAL_MIN_PCT,
            },
            "operator": {
                "battery_range": 20000,
                "empty_m_per_pct": 200.0,
                "loaded_m_per_pct": 150.0,
                "reserve_pct": DEFAULT_RESERVE_PCT,
                "charger_arrival_min_pct": DEFAULT_CHARGER_ARRIVAL_MIN_PCT,
            },
            "cleaner": {
                "battery_range": 28000,
                "empty_m_per_pct": 280.0,
                "loaded_m_per_pct": 200.0,
                "reserve_pct": DEFAULT_RESERVE_PCT,
                "charger_arrival_min_pct": DEFAULT_CHARGER_ARRIVAL_MIN_PCT,
            },
        }
        base = params.get(drone_type, params["cargo"]).copy()
        for k, v in params["cargo"].items():
            if k not in base:
                base[k] = v
        if self.battery_mode == BATTERY_MODE_TEST:
            base["empty_m_per_pct"] = base["empty_m_per_pct"] / TEST_RANGE_DIVISOR
            base["loaded_m_per_pct"] = base["loaded_m_per_pct"] / TEST_RANGE_DIVISOR
            base["battery_range"] = base.get("battery_range", 20000) / TEST_RANGE_DIVISOR
        return base

    def compute_battery_after(
        self,
        distance_m: float,
        battery_before_pct: float,
        mode: str,
        drone_type: str,
    ) -> float:
        """Compute battery level after flying distance_m with given mode. Returns 0..100."""
        p = self._get_drone_params(drone_type)
        m_per_pct = p["loaded_m_per_pct"] if mode == MODE_LOADED else p["empty_m_per_pct"]
        if m_per_pct <= 0:
            return 0.0
        drain_pct = (distance_m / m_per_pct)
        return max(0.0, min(100.0, battery_before_pct - drain_pct))

    def max_reachable_distance(
        self,
        battery_pct: float,
        mode: str,
        drone_type: str,
        reserve_pct: Optional[float] = None,
    ) -> float:
        """Max distance in meters reachable with given battery, respecting reserve."""
        p = self._get_drone_params(drone_type)
        reserve = reserve_pct if reserve_pct is not None else p.get("reserve_pct", DEFAULT_RESERVE_PCT)
        usable = max(0.0, battery_pct - reserve)
        m_per_pct = p["loaded_m_per_pct"] if mode == MODE_LOADED else p["empty_m_per_pct"]
        return usable * m_per_pct

    def plan_segment(
        self,
        G: nx.Graph,
        a_point_or_node: Any,
        b_point_or_node: Any,
        max_range_m: float,
        algorithm: str = "dijkstra",
    ) -> Tuple[Optional[List], Optional[List], float]:
        """
        Plan a single segment on city graph. a/b can be (lat, lon) or node_id.
        Returns (path_nodes, coords, length_m) or (None, None, 0.0).
        """
        if G is None or len(G.nodes) == 0:
            return None, None, 0.0
        # Resolve to nodes
        if isinstance(a_point_or_node, (list, tuple)) and len(a_point_or_node) == 2:
            a_node = self._find_nearest_node(G, tuple(a_point_or_node))
        else:
            a_node = a_point_or_node if a_point_or_node in G.nodes else None
        if isinstance(b_point_or_node, (list, tuple)) and len(b_point_or_node) == 2:
            b_node = self._find_nearest_node(G, tuple(b_point_or_node))
        else:
            b_node = b_point_or_node if b_point_or_node in G.nodes else None
        if not a_node or not b_node:
            return None, None, 0.0
        if a_node == b_node:
            pos = G.nodes[a_node]["pos"]
            return [a_node], [pos], 0.0
        path = self._find_safe_path(G, a_node, b_node, max_range_m)
        if not path:
            return None, None, 0.0
        coords = [G.nodes[n]["pos"] for n in path]
        length = self._calculate_path_length(G, path)
        return path, coords, length

    def build_meta_graph(
        self,
        G: nx.Graph,
        points: Dict[str, Any],
        start_name: str,
        charger_names: List[str],
        battery_pct: float,
        mode: str,
        drone_type: str,
        reserve_pct: Optional[float] = None,
        max_segment_battery_pct: Optional[float] = None,
        max_battery_pct_to_reach_charger: Optional[float] = None,
    ) -> nx.DiGraph:
        """
        Build meta-graph H. points: name -> node_id.
        Edge (a, b) exists if segment a->b is feasible: path exists and length <= max_range.
        max_range for edges FROM start = max_reachable(battery_pct); FROM charger = max_reachable(100).
        If max_segment_battery_pct is set (e.g. 50): рёбра из не-зарядки в goal ограничиваем —
        один сегмент без зарядки не должен «съедать» больше этой доли батареи, иначе путь через станции.
        If max_battery_pct_to_reach_charger is set (e.g. 85): до зарядки разрешаем лететь с меньшим запасом
        (оставляем 15%), чтобы дотянуть до станции и там зарядиться — иначе станции «на пути» не используются.
        """
        H = nx.DiGraph()
        p = self._get_drone_params(drone_type)
        reserve = reserve_pct if reserve_pct is not None else p.get("reserve_pct", DEFAULT_RESERVE_PCT)
        max_from_start = self.max_reachable_distance(battery_pct, mode, drone_type, reserve_pct=reserve)
        max_from_charger = self.max_reachable_distance(100.0, mode, drone_type, reserve_pct=reserve)
        # Макс. дальность одного сегмента без зарядки до цели (чтобы вставлять станции при длинных перегонах)
        if max_segment_battery_pct is not None and max_segment_battery_pct < 100:
            reserve_for_direct_goal = 100.0 - max_segment_battery_pct
            max_direct_to_goal = self.max_reachable_distance(
                100.0, mode, drone_type, reserve_pct=reserve_for_direct_goal
            )
        else:
            max_direct_to_goal = None
        # Допустимый запас при полёте до зарядки: можно дотянуть до станции с меньшим резервом
        if max_battery_pct_to_reach_charger is not None and max_battery_pct_to_reach_charger > 0:
            reserve_to_charger = 100.0 - max_battery_pct_to_reach_charger
            max_to_charger_from_start = self.max_reachable_distance(
                battery_pct, mode, drone_type, reserve_pct=reserve_to_charger
            )
            max_to_charger_from_charger = self.max_reachable_distance(
                100.0, mode, drone_type, reserve_pct=reserve_to_charger
            )
        else:
            max_to_charger_from_start = max_from_start
            max_to_charger_from_charger = max_from_charger
        charger_set = set(charger_names)
        goal_name = "goal"

        names = list(points.keys())
        for name_a in names:
            max_dist = max_from_charger if name_a in charger_set else max_from_start
            for name_b in names:
                if name_a == name_b:
                    continue
                # Перегон в зарядку: разрешаем длиннее сегмент (дотянуть до станции)
                if name_b in charger_set and max_battery_pct_to_reach_charger is not None:
                    max_dist_this = max_to_charger_from_charger if name_a in charger_set else max_to_charger_from_start
                # Прямой перегон в цель без зарядки — не длиннее чем max_segment_battery_pct батареи
                elif max_direct_to_goal is not None and name_b == goal_name and name_a not in charger_set:
                    max_dist_this = min(max_dist, max_direct_to_goal)
                else:
                    max_dist_this = max_dist
                pos_a = G.nodes[points[name_a]].get("pos")
                pos_b = G.nodes[points[name_b]].get("pos")
                if pos_a and pos_b:
                    direct_m = self._approx_point_distance_m(tuple(pos_a), tuple(pos_b))
                    if direct_m > max_dist_this * 1.1:
                        continue
                path, coords, length = self.plan_segment(
                    G, points[name_a], points[name_b], max_range_m=max_dist_this, algorithm="dijkstra"
                )
                if path is not None and length <= max_dist_this:
                    H.add_edge(name_a, name_b, weight=length, path=path, coords=coords, length=length)
        return H

    def plan_with_chargers(
        self,
        G: nx.Graph,
        start_node: Any,
        goal_node: Any,
        battery_pct: float,
        mode: str,
        drone_type: str,
        reserve_pct: Optional[float] = None,
        charger_nodes: Optional[Dict[str, Any]] = None,
        max_segment_battery_pct: Optional[float] = None,
        max_battery_pct_to_reach_charger: Optional[float] = None,
    ) -> Tuple[Optional[List], Optional[List], float, List[str]]:
        """
        Plan path from start to goal with optional charging stops (одна или несколько станций).
        charger_nodes: {"base": node_id or None, "stations": [node_id, ...]}.
        max_segment_battery_pct: макс. доля батареи на один перегон без зарядки (иначе путь через станции).
        max_battery_pct_to_reach_charger: допустимая доля батареи на перелёт до зарядки (дотянуть до станции).
        Returns (full_path_nodes, full_coords, total_length_m, visited_charger_names).
        """
        if G is None or not start_node or not goal_node:
            return None, None, 0.0, []
        # Resolve to node ids if coords
        if isinstance(start_node, (list, tuple)) and len(start_node) == 2:
            start_node = self._find_nearest_node(G, tuple(start_node))
        if isinstance(goal_node, (list, tuple)) and len(goal_node) == 2:
            goal_node = self._find_nearest_node(G, tuple(goal_node))
        if not start_node or start_node not in G.nodes or not goal_node or goal_node not in G.nodes:
            return None, None, 0.0, []
        p = self._get_drone_params(drone_type)
        reserve = reserve_pct if reserve_pct is not None else p.get("reserve_pct", DEFAULT_RESERVE_PCT)
        direct_max_range = self.max_reachable_distance(battery_pct, mode, drone_type, reserve_pct=reserve)
        if max_segment_battery_pct is not None and max_segment_battery_pct < 100:
            reserve_for_direct_goal = 100.0 - max_segment_battery_pct
            direct_max_range = min(
                direct_max_range,
                self.max_reachable_distance(100.0, mode, drone_type, reserve_pct=reserve_for_direct_goal),
            )
        direct_path, direct_coords, direct_length = self.plan_segment(
            G, start_node, goal_node, max_range_m=direct_max_range, algorithm="dijkstra"
        )
        if direct_path is not None and direct_coords is not None:
            return direct_path, direct_coords, direct_length, []

        points = {"start": start_node, "goal": goal_node}
        ch = charger_nodes or {}
        base_node = ch.get("base")
        station_nodes = ch.get("stations") or []
        if len(station_nodes) > 12:
            start_pos = G.nodes[start_node].get("pos")
            goal_pos = G.nodes[goal_node].get("pos")
            if start_pos and goal_pos:
                scored_stations = []
                for sn in station_nodes:
                    if sn is None or sn not in G.nodes:
                        continue
                    pos = G.nodes[sn].get("pos")
                    if not pos:
                        continue
                    score = self._approx_point_distance_m(tuple(start_pos), tuple(pos)) + self._approx_point_distance_m(tuple(pos), tuple(goal_pos))
                    scored_stations.append((score, sn))
                scored_stations.sort(key=lambda item: item[0])
                station_nodes = [sn for _, sn in scored_stations[:12]]
        charger_names_list = []
        if base_node is not None and base_node in G.nodes:
            points["base"] = base_node
            charger_names_list.append("base")
        for i, sn in enumerate(station_nodes):
            if sn is not None and sn in G.nodes:
                points[f"station_{i}"] = sn
                charger_names_list.append(f"station_{i}")

        H = self.build_meta_graph(
            G, points, "start", charger_names_list, battery_pct, mode, drone_type, reserve_pct,
            max_segment_battery_pct=max_segment_battery_pct,
            max_battery_pct_to_reach_charger=max_battery_pct_to_reach_charger,
        )
        if "start" not in H or "goal" not in H:
            return None, None, 0.0, []

        try:
            path_names = nx.shortest_path(H, "start", "goal", weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.logger.debug("plan_with_chargers: no path start->goal in meta-graph (battery=%.1f%%, mode=%s)", battery_pct, mode)
            return None, None, 0.0, []

        if len(path_names) > 2:
            self.logger.info(
                "plan_with_chargers: route via chargers: %s (battery=%.1f%%)",
                " -> ".join(path_names), battery_pct,
            )
        full_path = []
        full_coords = []
        total_length = 0.0
        visited_chargers = []
        battery = battery_pct
        m_per_pct = (
            self._get_drone_params(drone_type)["loaded_m_per_pct"]
            if mode == MODE_LOADED
            else self._get_drone_params(drone_type)["empty_m_per_pct"]
        )
        charger_names = {"base"} | {f"station_{i}" for i in range(len(station_nodes))}

        for i in range(len(path_names) - 1):
            a, b = path_names[i], path_names[i + 1]
            edge_data = H.get_edge_data(a, b)
            if not edge_data:
                return None, None, 0.0, []
            seg_path = edge_data.get("path") or []
            seg_coords = edge_data.get("coords") or []
            seg_len = edge_data.get("length", 0.0)
            if not seg_path:
                continue
            # When leaving a charger, battery is 100%
            if a in charger_names:
                battery = 100.0
            # Единая формула: m_per_pct = метров на 1% заряда → расход в % = distance_m / m_per_pct (без *100)
            drain = seg_len / m_per_pct
            battery_after = max(0.0, battery - drain)
            if battery_after < 0:
                return None, None, 0.0, []
            if b in charger_names:
                visited_chargers.append(b)
            battery = 100.0 if b in charger_names else battery_after

            if full_path and full_path[-1] == seg_path[0]:
                full_path.extend(seg_path[1:])
                full_coords.extend(seg_coords[1:] if seg_coords else [])
            else:
                full_path.extend(seg_path)
                full_coords.extend(seg_coords)
            total_length += seg_len

        return full_path, full_coords, total_length, visited_chargers
