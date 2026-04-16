import networkx as nx
import numpy as np
from shapely.geometry import Point
import osmnx as ox
import logging

class GraphService:
    def __init__(self):
        self.graphs = {}
        self.progress_callbacks = []
        self.logger = logging.getLogger(__name__)
    
    def add_progress_callback(self, callback):
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, stage, percentage, message=""):
        for callback in self.progress_callbacks:
            callback(stage, percentage, message)
    
    def build_city_graph(self, city_data, drone_type="cargo"):
        self._update_progress("graph", 0, "Построение графа для " + drone_type)
        
        try:
            drone_params = self._get_drone_params(drone_type)
            road_graph = city_data['road_graph']
            buildings = city_data.get('buildings', None)
            no_fly_zones = city_data.get('no_fly_zones', [])
            
            self._update_progress("graph", 20, "Конвертация дорожного графа")
            G = self._convert_ox_graph_to_nx(road_graph)
            
            if len(G.nodes) == 0:
                raise Exception("Получен пустой граф дорог")
            
            self._update_progress("graph", 50, f"Добавление узлов для {drone_type}")
            if buildings is not None and not buildings.empty:
                G = self._add_drone_specific_nodes(G, buildings, drone_params, drone_type)
            else:
                self.logger.warning("Данные о зданиях отсутствуют")
            
            self._update_progress("graph", 80, "Добавление запретных зон")
            if no_fly_zones:
                G = self._add_no_fly_zones(G, no_fly_zones)
            
            # Добавляем информацию о графе
            G.graph['city_name'] = city_data.get('city_name', 'Unknown')
            G.graph['drone_type'] = drone_type
            G.graph['drone_params'] = drone_params
            
            self._update_progress("graph", 100, f"Граф построен: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
            self.logger.info(f"Граф построен для {drone_type}: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
            return G
            
        except Exception as e:
            self.logger.error(f"Ошибка построения графа: {e}")
            self._update_progress("error", 0, f"Ошибка построения графа: {str(e)}")
            raise
    
    def _get_drone_params(self, drone_type):
        params = {
            "cargo": {"weight": 2.0, "max_altitude": 200, "battery_range": 20000},
            "operator": {"weight": 1.5, "max_altitude": 150, "battery_range": 15000},
            "cleaner": {"weight": 1.0, "max_altitude": 100, "battery_range": 10000}
        }
        return params.get(drone_type, params["cargo"])
    
    def _convert_ox_graph_to_nx(self, ox_graph):
        G = nx.Graph()
        
        # Добавляем узлы
        for node, data in ox_graph.nodes(data=True):
            try:
                # Проверяем наличие координат
                if 'x' in data and 'y' in data:
                    G.add_node(node, 
                             pos=(data['y'], data['x']), 
                             type='road', 
                             weight=1.0,
                             original_data=data)
                else:
                    self.logger.warning(f"Узел {node} без координат пропущен")
            except Exception as e:
                self.logger.warning(f"Ошибка добавления узла {node}: {e}")
        
        # Добавляем рёбра
        for u, v, data in ox_graph.edges(data=True):
            try:
                if u in G.nodes and v in G.nodes:
                    # Вычисляем длину ребра
                    length = data.get('length', 0)
                    if length == 0:
                        # Вычисляем расстояние между узлами если длина не указана (метры: широта и долгота с cos(lat))
                        pos1 = G.nodes[u]['pos']
                        pos2 = G.nodes[v]['pos']
                        lat1, lon1 = pos1[0], pos1[1]
                        lat2, lon2 = pos2[0], pos2[1]
                        lat_m = (lat2 - lat1) * 111000.0
                        avg_lat = (lat1 + lat2) / 2.0
                        lon_m = (lon2 - lon1) * 111000.0 * max(0.01, np.cos(np.radians(avg_lat)))
                        length = float(np.sqrt(lat_m**2 + lon_m**2))
                    
                    G.add_edge(u, v, 
                             weight=length, 
                             type='road',
                             original_data=data)
            except Exception as e:
                self.logger.warning(f"Ошибка добавления ребра {u}-{v}: {e}")
        
        self.logger.info(f"Конвертирован граф: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
        return G
    
    def _add_drone_specific_nodes(self, G, buildings, drone_params, drone_type):
        """Добавление специфичных узлов для разных типов дронов"""
        try:
            if drone_type == "cleaner" and buildings is not None and not buildings.empty:
                added_nodes = 0
                max_buildings = min(50, len(buildings))  # Ограничиваем количество зданий
                
                for idx, building in buildings.head(max_buildings).iterrows():
                    try:
                        if hasattr(building, 'geometry') and building.geometry and not building.geometry.is_empty:
                            centroid = building.geometry.centroid
                            if not centroid.is_empty:
                                node_id = f"{drone_type}_{idx}"
                                G.add_node(node_id, 
                                         pos=(centroid.y, centroid.x), 
                                         type=drone_type, 
                                         weight=drone_params['weight'],
                                         building_id=idx)
                                self._connect_to_nearest_road(G, node_id, (centroid.y, centroid.x))
                                added_nodes += 1
                    except Exception as e:
                        self.logger.warning(f"Ошибка обработки здания {idx}: {e}")
                        continue
                
                self.logger.info(f"Добавлено {added_nodes} узлов для дронов типа {drone_type}")
            
            elif drone_type in ["cargo", "operator"]:
                # Для грузовых и операторских дронов добавляем дополнительные узлы в ключевых точках
                self._add_critical_points(G, drone_params, drone_type)
                
        except Exception as e:
            self.logger.error(f"Ошибка добавления узлов для {drone_type}: {e}")
        
        return G
    
    def _add_critical_points(self, G, drone_params, drone_type):
        """Добавление критических точек для грузовых и операторских дронов"""
        try:
            # Добавляем узлы в центрах крупных дорог
            road_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'road']
            if len(road_nodes) > 0:
                # Выбираем случайные узлы для добавления критических точек
                import random
                critical_nodes = random.sample(road_nodes, min(10, len(road_nodes)))
                
                for i, node_id in enumerate(critical_nodes):
                    critical_id = f"critical_{drone_type}_{i}"
                    pos = G.nodes[node_id]['pos']
                    G.add_node(critical_id, 
                             pos=pos, 
                             type='critical', 
                             weight=drone_params['weight'] * 0.5)
                    G.add_edge(critical_id, node_id, weight=1.0, type='critical_connection')
                    
        except Exception as e:
            self.logger.warning(f"Ошибка добавления критических точек: {e}")
    
    def _add_no_fly_zones(self, G, no_fly_zones):
        for zone in no_fly_zones:
            # Mark nodes inside zone
            for node, attr in list(G.nodes(data=True)):
                if 'pos' not in attr:
                    continue
                if self._point_in_no_fly_zone(attr['pos'], zone):
                    # Mark node as blocked by setting very high weight
                    G.nodes[node]['weight'] = float('inf')
            # Penalize/block edges if either endpoint is inside a zone OR the segment intersects the rectangle
            for u, v, data in list(G.edges(data=True)):
                try:
                    upos = G.nodes[u].get('pos')
                    vpos = G.nodes[v].get('pos')
                    if (upos and self._point_in_no_fly_zone(upos, zone)) or (vpos and self._point_in_no_fly_zone(vpos, zone)):
                        data['weight'] = float('inf')
                    elif upos and vpos and self._segment_intersects_zone(upos, vpos, zone):
                        data['weight'] = float('inf')
                except Exception:
                    continue
        return G
    
    def _point_in_no_fly_zone(self, point, zone):
        # Support rectangular and circular runtime zones.
        try:
            lat, lon = point
            zone_type = (zone.get('zone_type') or 'rectangle').lower()
            if zone_type == 'circle':
                center_lat = zone.get('center_lat')
                center_lon = zone.get('center_lon')
                radius_m = float(zone.get('radius_m') or 0.0)
                if None in (center_lat, center_lon) or radius_m <= 0:
                    return False
                lat_m = (lat - float(center_lat)) * 111000.0
                avg_lat = (lat + float(center_lat)) / 2.0
                lon_m = (lon - float(center_lon)) * 111000.0 * max(0.01, np.cos(np.radians(avg_lat)))
                return float(np.sqrt(lat_m**2 + lon_m**2)) <= radius_m
            lat_min = zone.get('lat_min')
            lat_max = zone.get('lat_max')
            lon_min = zone.get('lon_min')
            lon_max = zone.get('lon_max')
            if None in (lat_min, lat_max, lon_min, lon_max):
                return False
            if lat_min > lat_max:
                lat_min, lat_max = lat_max, lat_min
            if lon_min > lon_max:
                lon_min, lon_max = lon_max, lon_min
            return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)
        except Exception:
            return False

    def _segment_intersects_zone(self, a, b, zone):
        """Check if line segment a->b intersects rectangle or circle no-fly zone."""
        try:
            zone_type = (zone.get('zone_type') or 'rectangle').lower()
            if zone_type == 'circle':
                for frac in np.linspace(0.0, 1.0, 24):
                    sample = (
                        a[0] + (b[0] - a[0]) * float(frac),
                        a[1] + (b[1] - a[1]) * float(frac),
                    )
                    if self._point_in_no_fly_zone(sample, zone):
                        return True
                return False
            lat_min = zone.get('lat_min')
            lat_max = zone.get('lat_max')
            lon_min = zone.get('lon_min')
            lon_max = zone.get('lon_max')
            if None in (lat_min, lat_max, lon_min, lon_max):
                return False
            if lat_min > lat_max:
                lat_min, lat_max = lat_max, lat_min
            if lon_min > lon_max:
                lon_min, lon_max = lon_max, lon_min

            (ax, ay) = a  # lat, lon
            (bx, by) = b

            # Quick reject via bbox
            seg_lat_min = min(ax, bx)
            seg_lat_max = max(ax, bx)
            seg_lon_min = min(ay, by)
            seg_lon_max = max(ay, by)
            if seg_lat_max < lat_min or seg_lat_min > lat_max or seg_lon_max < lon_min or seg_lon_min > lon_max:
                return False

            # If either endpoint inside -> treat as intersecting (already handled above, but keep safe)
            if (lat_min <= ax <= lat_max and lon_min <= ay <= lon_max) or (lat_min <= bx <= lat_max and lon_min <= by <= lon_max):
                return True

            # Line intersection with each of 4 rectangle edges
            rect_edges = [
                ((lat_min, lon_min), (lat_min, lon_max)),
                ((lat_max, lon_min), (lat_max, lon_max)),
                ((lat_min, lon_min), (lat_max, lon_min)),
                ((lat_min, lon_max), (lat_max, lon_max)),
            ]

            def ccw(p1, p2, p3):
                return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

            def segments_intersect(p1, p2, p3, p4):
                return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

            p1 = (ax, ay)
            p2 = (bx, by)
            for e1, e2 in rect_edges:
                if segments_intersect(p1, p2, e1, e2):
                    return True
            return False
        except Exception:
            return False
    
    def _connect_to_nearest_road(self, G, building_node, building_pos):
        """Соединение узла здания с ближайшей дорогой"""
        try:
            min_dist = float('inf')
            nearest_road = None
            road_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'road']
            
            # Ограничиваем поиск для производительности
            max_search = min(200, len(road_nodes))
            
            for road_node in road_nodes[:max_search]:
                try:
                    road_pos = G.nodes[road_node]['pos']
                    # Вычисляем расстояние в градусах (приблизительно)
                    dist = np.sqrt((building_pos[0]-road_pos[0])**2 + (building_pos[1]-road_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_road = road_node
                except Exception as e:
                    self.logger.warning(f"Ошибка вычисления расстояния для узла {road_node}: {e}")
                    continue
            
            # Соединяем если расстояние разумное (менее 0.01 градуса ≈ 1 км)
            if nearest_road and min_dist < 0.01:
                # Преобразуем расстояние в метры (приблизительно)
                distance_meters = min_dist * 111000  # 1 градус ≈ 111 км
                G.add_edge(building_node, nearest_road, 
                         weight=distance_meters, 
                         type='connection',
                         distance=min_dist)
                self.logger.debug(f"Соединён узел {building_node} с дорогой на расстоянии {distance_meters:.1f}м")
            else:
                self.logger.warning(f"Не удалось найти ближайшую дорогу для узла {building_node} (мин. расстояние: {min_dist:.6f})")
                
        except Exception as e:
            self.logger.error(f"Ошибка соединения узла {building_node} с дорогой: {e}")
