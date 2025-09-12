#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from rclpy.action import ActionClient
import numpy as np
import cv2
from rclpy.parameter import Parameter
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math
import heapq
from enum import Enum
import time

class ExecutionState(Enum):
    WAITING_FOR_ODOM = 0
    PLANNING = 1
    NAVIGATING_TO_CELL = 2      # NOWY - A* do komórki
    EXECUTING_BOUSTROPHEDON = 3 # NOWY - boustrophedon w komórce
    FINISHED = 4

class ChosetPignonNav2Planner(Node):
    def __init__(self):
        super().__init__('choset_pignon_nav2_planner')
        
        # Map name
        self.map_name = "map_test"
        
        # Ustaw parametr use_sim_time na True
        param = rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        self.set_parameters([param])
        
        # Parametry robota
        self.ROBOT_WIDTH_METERS = 0.2
        self.SAFETY_MARGIN_METERS = 0.4  # 0.45 JEDEN MARGINES DLA WSZYSTKIEGO
        
        # Pliki mapy
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.MAP_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.pgm")
        self.YAML_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.yaml")
        
        # Stan wykonania - NOWE STANY
        self.state = ExecutionState.WAITING_FOR_ODOM
        self.robot_position = None
        self.robot_pixel_position = None
        self.map_loaded = False
        
        # DODANE: Rzeczywiste śledzenie czasu
        self.real_start_time = None
        self.cell_start_times = {}
        self.cell_completion_times = {}
        self.total_real_time = 0.0
        self.complete_real_path = []  # Rzeczywista ścieżka robota
        
        # Choset-Pignon dane
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.free_space = None
        self.safe_space = None  # JEDNA BEZPIECZNA PRZESTRZEŃ DLA WSZYSTKIEGO
        self.cells = []
        self.cell_paths = {}
        self.cell_sequence = []
        self.critical_events = []
        
        # Wykonanie - NOWE ZMIENNE
        self.current_cell_index = 0
        self.current_path = None
        self.execution_start_time = None
        self.current_navigation_phase = "none"  # "to_cell" lub "boustrophedon"
        
        # QoS
        costmap_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        
        # Publikatory
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.global_costmap_pub = self.create_publisher(
            OccupancyGrid, '/global_costmap/costmap', costmap_qos)
        self.local_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap/costmap', costmap_qos)
        
        # Klient akcji Nav2
        self.nav2_client = ActionClient(self, FollowPath, '/follow_path')
        
        # Subskrypcje
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Inicjalizacja
        self.load_map()
        self.run_choset_pignon_decomposition()

    def load_map(self):
        """Wczytaj mapę"""
        try:
            with open(self.YAML_FILE) as f:
                map_meta = yaml.safe_load(f)
                self.map_resolution = map_meta['resolution']
                self.map_origin = map_meta['origin'][:2]
            
            img = cv2.imread(self.MAP_FILE, cv2.IMREAD_GRAYSCALE)
            self.map_data = np.zeros_like(img, dtype=np.int8)
            self.map_data[img == 254] = 0    # wolna przestrzeń
            self.map_data[img == 0] = 100    # przeszkody
            self.map_data[img == 205] = -1   # nieznane
            
            self.publish_costmap()
            self.map_loaded = True
            
        except Exception as e:
            self.get_logger().error(f"Błąd ładowania mapy: {str(e)}")
            raise

    def publish_costmap(self):
        """Publikuje mapę do costmapa"""
        costmap_msg = OccupancyGrid()
        costmap_msg.header.frame_id = 'map'
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.info.resolution = self.map_resolution
        costmap_msg.info.width = self.map_data.shape[1]
        costmap_msg.info.height = self.map_data.shape[0]
        costmap_msg.info.origin.position.x = self.map_origin[0]
        costmap_msg.info.origin.position.y = self.map_origin[1]
        costmap_msg.info.origin.orientation.w = 1.0
        costmap_msg.data = self.map_data.flatten().tolist()
        
        self.global_costmap_pub.publish(costmap_msg)
        self.local_costmap_pub.publish(costmap_msg)

    def run_choset_pignon_decomposition(self):
        """Uruchom dekompozycję Choset-Pignon z wizualizacją"""
        if not self.map_loaded:
            return
            
        try:
            self.create_clean_free_space()
            self.find_critical_events()
            self.create_cells_from_events()
            self.generate_boustrophedon_paths()
            self.plan_cell_sequence()
            
            # WIZUALIZACJE OSOBNO - BEZ ZAPISYWANIA PNG
            self.visualize_decomposition_window()
            self.visualize_paths_window() 
            self.visualize_boustrophedon_window()
            self.visualize_coverage_window()
            
        except Exception as e:
            self.get_logger().error(f"Błąd dekompozycji: {str(e)}")


    def create_clean_free_space(self):
        """POPRAWIONE: Jedna bezpieczna przestrzeń dla wszystkiego"""
        # Podstawowa wolna przestrzeń
        self.free_space = (self.map_data == 0).astype(np.uint8)
        
        # Lekkie wygładzenie
        kernel = np.ones((3, 3), np.uint8)
        self.free_space = cv2.morphologyEx(self.free_space, cv2.MORPH_OPEN, kernel)
        
        # JEDEN SAFETY MARGIN dla wszystkiego - A* i boustrophedon
        safety_margin_pixels = int(self.SAFETY_MARGIN_METERS / self.map_resolution)
        
        self.get_logger().info(f"🛡️ Using safety margin: {self.SAFETY_MARGIN_METERS}m = {safety_margin_pixels} pixels")
        
        if safety_margin_pixels > 0:
            # Oblicz distance transform
            dist_transform = cv2.distanceTransform(self.free_space, cv2.DIST_L2, 5)
            
            # JEDNA BEZPIECZNA PRZESTRZEŃ dla A* i boustrophedon
            self.safe_space = (dist_transform > safety_margin_pixels).astype(bool)
        else:
            self.safe_space = self.free_space.astype(bool)
        
        # Zachowaj connectivity
        self.safe_space = self.ensure_connectivity(self.safe_space)
        
        # Info o marginesie
        safe_area = np.sum(self.safe_space)
        free_area = np.sum(self.free_space)
        self.get_logger().info(f"📊 Safe area: {safe_area} pixels ({safe_area/free_area*100:.1f}% of free space)")

    def ensure_connectivity(self, space):
        """Zachowaj największy połączony komponent"""
        num_labels, labels = cv2.connectedComponents(space.astype(np.uint8))
        
        if num_labels <= 1:
            return space
        
        # Znajdź największy komponent
        largest_component = 0
        largest_size = 0
        for i in range(1, num_labels):
            size = np.sum(labels == i)
            if size > largest_size:
                largest_size = size
                largest_component = i
        
        # Zwróć tylko największy komponent
        result = np.zeros_like(space)
        result[labels == largest_component] = True
        
        return result

    def find_critical_events(self):
        """Znajdź critical events - używa podstawowej free_space do dekompozycji"""
        h, w = self.free_space.shape
        y_coords, x_coords = np.where(self.free_space)
        if len(x_coords) == 0:
            return
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        self.critical_events = []
        self.critical_events.append({'x': x_min, 'type': 'START'})
        
        scan_step = 2
        prev_segments = None
        
        for x in range(x_min + scan_step, x_max, scan_step):
            curr_segments = self.get_segments_at_x(x, y_min, y_max)
            
            if prev_segments is not None:
                event_type = self.analyze_topology_change(prev_segments, curr_segments)
                if event_type != 'NONE':
                    self.critical_events.append({
                        'x': x, 'type': event_type,
                        'prev_segments': prev_segments,
                        'curr_segments': curr_segments
                    })
            
            prev_segments = curr_segments
        
        self.critical_events.append({'x': x_max, 'type': 'END'})

    def get_segments_at_x(self, x, y_min, y_max):
        """Znajdź segmenty wolnej przestrzeni w danym x"""
        segments = []
        current_start = None
        
        for y in range(y_min, y_max + 1):
            if (0 <= y < self.free_space.shape[0] and 
                0 <= x < self.free_space.shape[1] and 
                self.free_space[y, x]):
                
                if current_start is None:
                    current_start = y
            else:
                if current_start is not None:
                    segment_length = y - current_start
                    if segment_length >= 10:
                        segments.append((current_start, y - 1))
                    current_start = None
        
        if current_start is not None:
            segment_length = y_max - current_start + 1
            if segment_length >= 10:
                segments.append((current_start, y_max))
        
        return segments

    def analyze_topology_change(self, prev_segments, curr_segments):
        """Analizuj zmiany topologii"""
        prev_count = len(prev_segments)
        curr_count = len(curr_segments)
        
        if curr_count > prev_count:
            return 'SPLIT'
        elif curr_count < prev_count:
            return 'MERGE'
        elif prev_count == curr_count and prev_count > 0:
            for i in range(min(len(prev_segments), len(curr_segments))):
                prev_start, prev_end = prev_segments[i]
                curr_start, curr_end = curr_segments[i]
                
                start_shift = abs(prev_start - curr_start)
                end_shift = abs(prev_end - curr_end)
                
                if start_shift > 8 or end_shift > 8:
                    return 'SHIFT'
        
        return 'NONE'

    def create_cells_from_events(self):
        """Utwórz komórki z critical events"""
        if len(self.critical_events) < 2:
            return
        
        cell_id = 1
        
        for i in range(len(self.critical_events) - 1):
            start_event = self.critical_events[i]
            end_event = self.critical_events[i + 1]
            
            x_start = start_event['x']
            x_end = end_event['x']
            
            x_middle = (x_start + x_end) // 2
            y_coords, x_coords = np.where(self.free_space)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            segments = self.get_segments_at_x(x_middle, y_min, y_max)
            
            for seg_start, seg_end in segments:
                if (x_end - x_start) > 12 and (seg_end - seg_start) > 10:
                    cell = {
                        'id': cell_id,
                        'contour': [[x_start, seg_start], [x_end, seg_start], 
                                  [x_end, seg_end], [x_start, seg_end]],
                        'x_start': x_start, 'x_end': x_end,
                        'y_start': seg_start, 'y_end': seg_end,
                        'centroid': ((x_start + x_end) // 2, (seg_start + seg_end) // 2),
                        'area': (x_end - x_start) * (seg_end - seg_start),
                        'area_m2': (x_end - x_start) * (seg_end - seg_start) * (self.map_resolution ** 2)
                    }
                    
                    cell['has_internal_obstacles'] = self.check_internal_obstacles(cell)
                    self.cells.append(cell)
                    cell_id += 1

    def check_internal_obstacles(self, cell):
        """Sprawdź czy komórka ma przeszkody wewnętrzne"""
        x_start, x_end = cell['x_start'], cell['x_end']
        y_start, y_end = cell['y_start'], cell['y_end']
        
        if (x_end - x_start) <= 0 or (y_end - y_start) <= 0:
            return False
        
        region = self.free_space[y_start:y_end+1, x_start:x_end+1]
        if region.size == 0:
            return False
        
        free_ratio = np.sum(region) / region.size
        return free_ratio < 0.5

    def generate_boustrophedon_paths(self):
        """Generuj ścieżki boustrophedon - UŻYWA safe_space"""
        robot_width_pixels = int(self.ROBOT_WIDTH_METERS / self.map_resolution * 1.7)
        
        for cell in self.cells:
            if cell.get('has_internal_obstacles', False):
                continue
            
            cell_name = f"C{cell['id']}"
            waypoints = self.create_vertical_boustrophedon(cell, robot_width_pixels)
            
            if waypoints:
                self.cell_paths[cell_name] = {
                    'cell_id': cell['id'],
                    'waypoints': waypoints,
                    'start_point': waypoints[0],
                    'end_point': waypoints[-1],
                    'waypoint_count': len(waypoints)
                }

    def create_vertical_boustrophedon(self, cell, robot_width_pixels):
        """Utwórz pionową ścieżkę boustrophedon - DOSTOSOWANĄ do RPP lookahead"""
        x_start, x_end = cell['x_start'], cell['x_end']
        y_start, y_end = cell['y_start'], cell['y_end']
        
        # AUTOMATYCZNY spacing dla RPP
        lookahead_meters = 0.2  # Twój lookahead distance  
        optimal_spacing_meters = lookahead_meters * 0.6  # 50% lookahead dla pewności
        point_spacing = max(2, int(optimal_spacing_meters / self.map_resolution))
        
        self.get_logger().info(f"🎯 Boustrophedon spacing: {point_spacing} px ({point_spacing * self.map_resolution:.3f}m) dla lookahead {lookahead_meters}m")
        
        waypoints = []
        x_current = x_start + robot_width_pixels
        going_down = True
        
        while x_current < x_end - robot_width_pixels:
            safe_y_start, safe_y_end = self.find_safe_y_range(x_current, y_start, y_end)
            
            if safe_y_start is not None and safe_y_end is not None:
                adjusted_y_start = safe_y_start
                adjusted_y_end = safe_y_end 
                
                if adjusted_y_end <= adjusted_y_start:
                    x_current += robot_width_pixels
                    continue
                
                # Generuj punkty z spacing dostosowanym do RPP
                if going_down:
                    dense_points = self.generate_dense_line_points(
                        (x_current, adjusted_y_start), 
                        (x_current, adjusted_y_end), 
                        point_spacing
                    )
                else:
                    dense_points = self.generate_dense_line_points(
                        (x_current, adjusted_y_end), 
                        (x_current, adjusted_y_start), 
                        point_spacing
                    )
                
                waypoints.extend(dense_points)
                going_down = not going_down
            
            x_current += robot_width_pixels
        
        return waypoints

    def generate_dense_line_points(self, start_point, end_point, spacing=None):
        """Generuj punkty dostosowane do RPP lookahead distance"""
        if spacing is None:
            # AUTOMATYCZNY spacing na podstawie lookahead distance
            lookahead_meters = 0.2  # Twój lookahead distance
            optimal_spacing_meters = lookahead_meters * 0.7  # 60% lookahead = bezpieczny odstęp
            spacing = max(2, int(optimal_spacing_meters / self.map_resolution))  # Minimum 2 piksele
        
        x1, y1 = start_point
        x2, y2 = end_point
        
        points = []
        
        # Oblicz długość linii
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length < spacing:
            return [start_point, end_point]
        
        # Oblicz liczbę punktów na podstawie spacing
        num_points = max(2, int(line_length / spacing))
        
        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            points.append((x, y))
        
        return points

    def find_safe_y_range(self, x, y_start, y_end):
        """Znajdź bezpieczny zakres Y dla danego X - UŻYWA safe_space"""
        safe_start = None
        safe_end = None
        
        for y in range(y_start, y_end + 1):
            if self.is_point_safe((x, y)):
                if safe_start is None:
                    safe_start = y
                safe_end = y
        
        return safe_start, safe_end

    def is_point_safe(self, point):
        """POPRAWIONE: Sprawdź czy punkt jest bezpieczny - UŻYWA safe_space"""
        x, y = int(point[0]), int(point[1])
        if (0 <= x < self.safe_space.shape[1] and 
            0 <= y < self.safe_space.shape[0]):
            return self.safe_space[y, x]
        return False

    def plan_cell_sequence(self):
        """Zaplanuj kolejność odwiedzania komórek"""
        if not self.cell_paths:
            return
        
        cell_names = list(self.cell_paths.keys())
        
        if len(cell_names) <= 1:
            self.cell_sequence = cell_names
            return
        
        # Algorytm najbliższego sąsiada
        visited = set()
        current_cell = cell_names[0]
        self.cell_sequence = [current_cell]
        visited.add(current_cell)
        
        while len(visited) < len(cell_names):
            min_distance = float('inf')
            next_cell = None
            
            current_end = self.cell_paths[current_cell]['end_point']
            
            for cell_name in cell_names:
                if cell_name in visited:
                    continue
                
                cell_start = self.cell_paths[cell_name]['start_point']
                distance = math.sqrt((current_end[0] - cell_start[0])**2 + 
                                   (current_end[1] - cell_start[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    next_cell = cell_name
            
            if next_cell:
                self.cell_sequence.append(next_cell)
                visited.add(next_cell)
                current_cell = next_cell

    def advanced_astar_navigation(self, start, goal):
        """POPRAWIONY A* - UŻYWA safe_space z pełnym safety margin"""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        h, w = self.safe_space.shape  # ZMIENIONE: używa safe_space zamiast navigation_space
        
        self.get_logger().info(f"🔍 A* navigation with {self.SAFETY_MARGIN_METERS}m safety margin")
        
        closed_set = set()
        open_set = {start}
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda point: f_score.get(point, float('inf')))
            
            # PRECYZYJNA tolerancja
            if current == goal or self.heuristic(current, goal) <= 1:
                path = self.reconstruct_path(came_from, current)
                
                # WYMUSZAJ dokładny koniec w goal
                if path and path[-1] != goal:
                    if self.is_point_safe(goal):
                        path.append(goal)
                        
                # WIĘKSZY spacing dla densify
                return self.densify_path(path, spacing=8)
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 8-kierunkowa nawigacja dla lepszej płynności
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                
                if not (0 <= nx < w and 0 <= ny < h) or neighbor in closed_set:
                    continue
                
                # UŻYWA safe_space zamiast navigation_space
                if not self.is_point_safe((nx, ny)):
                    continue
                
                if not self.is_safe_path(current, neighbor):
                    continue
                
                # Koszt diagonal vs prosty
                step_cost = math.sqrt(dx*dx + dy*dy)
                tentative_g_score = g_score[current] + step_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
        
        self.get_logger().warn(f"⚠️ A* nie znalazł ścieżki - fallback")
        # Fallback z auto spacing dla RPP
        fallback = self.generate_dense_line_points(start, goal, spacing=None)  # Auto spacing
        if fallback and fallback[-1] != goal:
            fallback[-1] = goal  # Wymuszaj dokładny koniec
        return fallback

    def densify_path(self, path, spacing=None):
        """Dodaj punkty do ścieżki A* dostosowane do RPP lookahead"""
        if len(path) <= 1:
            return path
        
        if spacing is None:
            # Auto spacing dla RPP
            lookahead_meters = 0.2
            optimal_spacing_meters = lookahead_meters * 0.6  # 60% lookahead
            spacing = max(2, int(optimal_spacing_meters / self.map_resolution))
        
        dense_path = []
        
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            segment_points = self.generate_dense_line_points(start_point, end_point, spacing)
            
            if i == 0:
                dense_path.extend(segment_points)
            else:
                dense_path.extend(segment_points[1:])  # Usuń duplikat
        
        return dense_path

    def heuristic(self, p1, p2):
        """Heurystyka Manhattan"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def reconstruct_path(self, came_from, current):
        """Rekonstruuj ścieżkę"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def is_safe_path(self, p1, p2):
        """POPRAWIONE: Sprawdź bezpieczeństwo ścieżki - UŻYWA safe_space"""
        x1, y1 = p1
        x2, y2 = p2
        
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            
            if not self.is_point_safe((x, y)):
                return False
                
            # Sprawdź margines bezpieczeństwa - już uwzględniony w safe_space
            # Dodatkowe sprawdzenie nie jest potrzebne
        
        return True

    # === WIZUALIZACJA ===
    
    def visualize_decomposition_window(self):
        """OKNO 1: Dekompozycja z safety margin"""
        if not self.cells:
            self.get_logger().warn("Brak komórek do wizualizacji")
            return
        
        # Przygotuj mapę
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200    # wolna przestrzeń - jasna
        display_map[self.map_data == 100] = 0    # przeszkody - czarne
        display_map[self.map_data == -1] = 100   # nieznane - szare
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        # POKAŻ BEZPIECZNĄ PRZESTRZEŃ
        safe_overlay = np.zeros_like(self.safe_space, dtype=np.uint8)
        safe_overlay[self.safe_space] = 1
        ax.contour(safe_overlay, levels=[0.5], colors='blue', linewidths=2, alpha=0.7)
        
        # Kolory dla komórek
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(self.cells), 12)))
        if len(self.cells) > 12:
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(self.cells), 20)))
        
        for i, cell in enumerate(self.cells):
            color = colors[i % len(colors)]
            contour = np.array(cell['contour'])
            
            # Różny styl dla komórek z przeszkodami
            if cell.get('has_internal_obstacles', False):
                polygon = Polygon(contour, fill=True, alpha=0.4, 
                                color='gray', edgecolor='red', linewidth=3)
                obstacle_marker = " ⚠️"
            else:
                polygon = Polygon(contour, fill=True, alpha=0.7, 
                                color=color, edgecolor='black', linewidth=2)
                obstacle_marker = ""
            
            ax.add_patch(polygon)
            
            cell_text = str(cell['id']) + obstacle_marker
            ax.text(cell['centroid'][0], cell['centroid'][1], cell_text, 
                color='black', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))
        
        ax.set_title(f'Choset-Pignon Decomposition + Safety Margin ({self.SAFETY_MARGIN_METERS}m)\n{len(self.cells)} komórek, Blue = Safe Space')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić dekompozycji (headless environment)")

    def visualize_paths_window(self):
        """OKNO 2: Rzeczywiste ścieżki A* między komórkami"""
        if not self.cell_sequence:
            self.get_logger().warn("Brak sekwencji komórek")
            return
        
        # Przygotuj mapę
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200    # wolna przestrzeń - jasna
        display_map[self.map_data == 100] = 0    # przeszkody - czarne
        display_map[self.map_data == -1] = 100   # nieznane - szare
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        # Kolory dla komórek
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(self.cells), 12)))
        if len(self.cells) > 12:
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(self.cells), 20)))
        
        # Komórki z numeracją kolejności (półprzeźroczyste)
        for i, cell in enumerate(self.cells):
            color = colors[i % len(colors)]
            contour = np.array(cell['contour'])
            cell_name = f"C{cell['id']}"
            
            if cell.get('has_internal_obstacles', False):
                polygon = Polygon(contour, fill=True, alpha=0.2, 
                                color='gray', edgecolor='red', linewidth=1)
            else:
                polygon = Polygon(contour, fill=True, alpha=0.3, 
                                color=color, edgecolor='black', linewidth=1)
            
            ax.add_patch(polygon)
            
            if cell_name in self.cell_sequence:
                seq_num = self.cell_sequence.index(cell_name) + 1
                ax.text(cell['centroid'][0], cell['centroid'][1], str(seq_num), 
                    color='white', fontsize=16, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.9, boxstyle='circle,pad=0.4'))
        
        # RZECZYWISTE ŚCIEŻKI A* - z safety margin
        for i in range(len(self.cell_sequence) - 1):
            current_cell = self.cell_sequence[i]
            next_cell = self.cell_sequence[i + 1]
            
            if current_cell in self.cell_paths and next_cell in self.cell_paths:
                start_point = self.cell_paths[current_cell]['end_point']
                end_point = self.cell_paths[next_cell]['start_point']
                
                # PRAWDZIWA ŚCIEŻKA A* z safety margin
                real_path = self.advanced_astar_navigation(start_point, end_point)
                
                if real_path and len(real_path) > 1:
                    path_x = [p[0] for p in real_path]
                    path_y = [p[1] for p in real_path]
                    ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, 
                            label=f'A* Safe Navigation ({self.SAFETY_MARGIN_METERS}m)' if i == 0 else "")
                    
                    # Strzałka na końcu ścieżki
                    if len(real_path) >= 2:
                        end_x, end_y = real_path[-1]
                        prev_x, prev_y = real_path[-2]
                        dx = end_x - prev_x
                        dy = end_y - prev_y
                        length = np.sqrt(dx**2 + dy**2)
                        if length > 0:
                            dx = dx / length * 15
                            dy = dy / length * 15
                            ax.arrow(end_x - dx, end_y - dy, dx, dy, 
                                    head_width=8, head_length=8, 
                                    fc='blue', ec='blue', alpha=0.8)
        
        sequence_text = " → ".join(self.cell_sequence) if len(self.cell_sequence) < 8 else f"{len(self.cell_sequence)} cells"
        ax.set_title(f'Execution Sequence + Safe A* Paths\n{sequence_text}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        if len(self.cell_sequence) > 1:
            ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić ścieżek A* (headless environment)")

    def visualize_boustrophedon_window(self):
        """OKNO 3: Ścieżki boustrophedon"""
        if not self.cell_paths:
            self.get_logger().warn("Brak ścieżek boustrophedon")
            return
        
        # Przygotuj mapę
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200    # wolna przestrzeń - jasna
        display_map[self.map_data == 100] = 0    # przeszkody - czarne
        display_map[self.map_data == -1] = 100   # nieznane - szare
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        # Kolory dla komórek
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(self.cells), 12)))
        if len(self.cells) > 12:
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(self.cells), 20)))
        
        # Komórki (bardzo półprzeźroczyste)
        for i, cell in enumerate(self.cells):
            color = colors[i % len(colors)]
            contour = np.array(cell['contour'])
            
            if not cell.get('has_internal_obstacles', False):
                polygon = Polygon(contour, fill=True, alpha=0.1, 
                                color=color, edgecolor='gray', linewidth=1)
                ax.add_patch(polygon)
        
        # ŚCIEŻKI BOUSTROPHEDON
        total_coverage = 0
        path_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        
        for i, (cell_name, cell_data) in enumerate(self.cell_paths.items()):
            waypoints = cell_data['waypoints']
            color = path_colors[i % len(path_colors)]
            
            if len(waypoints) > 1:
                # Ścieżka boustrophedon
                x_coords = [wp[0] for wp in waypoints]
                y_coords = [wp[1] for wp in waypoints]
                ax.plot(x_coords, y_coords, color=color, linewidth=2.5, alpha=0.9, 
                        label=f'{cell_name}' if i < 6 else "")
                
                # Punkty start/end
                start = cell_data['start_point']
                end = cell_data['end_point']
                
                ax.plot(start[0], start[1], 'o', color='lime', markersize=10, 
                        markeredgecolor='black', markeredgewidth=2)
                ax.plot(end[0], end[1], 's', color='red', markersize=10, 
                        markeredgecolor='black', markeredgewidth=2)
                
                total_coverage += len(waypoints)
        
        # DODAJ RZECZYWISTY CZAS JEŚLI DOSTĘPNY
        time_text = ""
        if self.total_real_time > 0:
            time_text = f" | Real time: {self.total_real_time:.1f}s"
        elif hasattr(self, 'real_start_time') and self.real_start_time:
            current_time = time.time() - self.real_start_time
            time_text = f" | Current time: {current_time:.1f}s"
        
        ax.set_title(f'Safe Boustrophedon Paths ({self.SAFETY_MARGIN_METERS}m margin)\n{total_coverage} waypoints{time_text}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić boustrophedon (headless environment)")

    def visualize_coverage_window(self):
        """OKNO 4: Analiza pokrycia z kółkami skanera"""
        if not self.cell_paths:
            self.get_logger().warn("Brak ścieżek do analizy pokrycia")
            return
        
        # Przygotuj mapę
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200    # wolna przestrzeń - jasna
        display_map[self.map_data == 100] = 0    # przeszkody - czarne
        display_map[self.map_data == -1] = 100   # nieznane - szare
        
        # Połącz wszystkie ścieżki w jedną kompletną trasę
        complete_path = []
        for cell_name in self.cell_sequence:
            if cell_name in self.cell_paths:
                cell_waypoints = self.cell_paths[cell_name]['waypoints']
                complete_path.extend(cell_waypoints)
                
                # Dodaj A* roadmap między komórkami
                current_index = self.cell_sequence.index(cell_name)
                if current_index < len(self.cell_sequence) - 1:
                    next_cell = self.cell_sequence[current_index + 1]
                    if next_cell in self.cell_paths:
                        start_point = self.cell_paths[cell_name]['end_point']
                        end_point = self.cell_paths[next_cell]['start_point']
                        roadmap_path = self.advanced_astar_navigation(start_point, end_point)
                        if roadmap_path:
                            complete_path.extend(roadmap_path)
        
        if not complete_path:
            self.get_logger().warn("Brak ścieżki do analizy pokrycia")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        # PARAMETRY SKANERA
        robot_diameter_px = int(self.ROBOT_WIDTH_METERS / self.map_resolution)
        scanner_diameter_px = robot_diameter_px * 2  # 2x robot width
        scanner_radius = scanner_diameter_px / 2
        
        # Wybierz punkty w regularnych odstępach
        circle_spacing = scanner_radius * 1.0  # Gęściej dla lepszego pokrycia
        selected_points = []
        
        if complete_path:
            path_distance = 0
            selected_points = [complete_path[0]]
            
            for i in range(1, len(complete_path)):
                prev_x, prev_y = complete_path[i-1]
                curr_x, curr_y = complete_path[i]
                
                segment_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                path_distance += segment_distance
                
                if path_distance >= circle_spacing:
                    selected_points.append(complete_path[i])
                    path_distance = 0
            
            if selected_points[-1] != complete_path[-1]:
                selected_points.append(complete_path[-1])
            
            # CIEMNE kółka pokrycia
            for x, y in selected_points:
                # Ciemne wypełnienie
                scanner_circle = plt.Circle((x, y), radius=scanner_radius, 
                                        color='darkblue', alpha=0.4, fill=True)
                ax.add_patch(scanner_circle)
                
                # Ciemna obwódka
                scanner_outline = plt.Circle((x, y), radius=scanner_radius, 
                                        color='navy', alpha=0.8, fill=False, linewidth=2)
                ax.add_patch(scanner_outline)
                
                # Białe centrum
                ax.scatter(x, y, c='white', s=8, alpha=0.9, edgecolors='black', linewidths=0.5)
            
            # Pokaż trasę
            px, py = zip(*complete_path)
            ax.plot(px, py, 'yellow', linewidth=1.5, alpha=0.8, label='Complete Choset Path')
            
            # Dodaj rzeczywistą ścieżkę robota jeśli dostępna
            if self.complete_real_path and len(self.complete_real_path) > 1:
                real_x = [p[0] for p in self.complete_real_path]
                real_y = [p[1] for p in self.complete_real_path]
                ax.plot(real_x, real_y, 'lime', linewidth=2, alpha=0.7, label='Robot Real Path')
        
        # OBLICZ POKRYCIE
        coverage_map = np.zeros_like(self.map_data, dtype=np.uint8)
        h, w = self.map_data.shape
        
        for x, y in selected_points:
            scanner_r_int = int(scanner_radius)
            for dy in range(-scanner_r_int - 1, scanner_r_int + 2):
                for dx in range(-scanner_r_int - 1, scanner_r_int + 2):
                    nx, ny = x + dx, y + dy
                    if dx*dx + dy*dy <= scanner_radius**2:
                        if 0 <= nx < w and 0 <= ny < h:
                            coverage_map[ny, nx] = 1
        
        # Dylatacja dla wypełnienia dziur
        kernel = np.ones((2, 2), np.uint8)
        coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)
        
        # OBLICZ POKRYCIE - tylko wolne obszary
        free_mask = (self.map_data == 0)  # Tylko wolne obszary
        coverage_mask = (coverage_map == 1)
        
        # Przecięcie: pokryte wolne obszary
        covered_free_mask = free_mask & coverage_mask
        
        free_cells = np.sum(free_mask)
        covered_free_cells = np.sum(covered_free_mask)
        coverage_percent = covered_free_cells / free_cells * 100 if free_cells > 0 else 0
        
        # ACCESSIBLE COVERAGE
        robot_accessible_mask = self.safe_space if hasattr(self, 'safe_space') else free_mask
        accessible_cells = np.sum(robot_accessible_mask)
        covered_accessible_cells = np.sum(robot_accessible_mask & coverage_mask)
        accessible_coverage_percent = covered_accessible_cells / accessible_cells * 100 if accessible_cells > 0 else 0
        
        # Dodaj pozycję robota jeśli dostępna
        if hasattr(self, 'robot_pixel_position') and self.robot_pixel_position:
            ax.plot(self.robot_pixel_position[0], self.robot_pixel_position[1], 
                    'ro', markersize=15, markeredgecolor='yellow', markeredgewidth=3,
                    label='Robot Position')
        
        # RZECZYWISTY CZAS
        time_text = ""
        if self.total_real_time > 0:
            time_text = f"Real time: {self.total_real_time:.1f}s"
        elif hasattr(self, 'real_start_time') and self.real_start_time:
            current_time = time.time() - self.real_start_time
            time_text = f"Current: {current_time:.1f}s"
        else:
            # Estymacja
            if complete_path:
                total_distance_px = 0
                for i in range(1, len(complete_path)):
                    prev_x, prev_y = complete_path[i-1]
                    curr_x, curr_y = complete_path[i]
                    total_distance_px += np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                
                total_distance_m = total_distance_px * self.map_resolution
                estimated_speed = 0.3  # m/s
                estimated_time_s = total_distance_m / estimated_speed
                time_text = f"Est: {estimated_time_s:.1f}s"
            else:
                time_text = "No path"
        
        # TYTUŁ I OSIE z accessible coverage
        ax.set_title(f'Choset Coverage Analysis - Scanner Range Visualization\n'
                    f'Coverage: {coverage_percent:.1f}% (all free) | {accessible_coverage_percent:.1f}% (accessible) | {len(selected_points)} scan points | {time_text}')
        ax.set_xlabel(f'Total Coverage: {coverage_percent:.1f}% | Accessible Coverage: {accessible_coverage_percent:.1f}% | Scan Points: {len(selected_points)}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        self.get_logger().info(f"📊 Choset Coverage Analysis: {coverage_percent:.1f}% total | {accessible_coverage_percent:.1f}% accessible")
        
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić wizualizacji pokrycia (headless environment)")
        
        # STATYSTYKI POKRYCIA z accessible coverage
        self.get_logger().info(f"🎯 CHOSET COVERAGE DETAILS:")
        self.get_logger().info(f"  • Scanner diameter: {scanner_diameter_px} px ({scanner_diameter_px * self.map_resolution:.3f}m)")
        self.get_logger().info(f"  • Total scan points: {len(selected_points)}")
        self.get_logger().info(f"  • Free cells: {free_cells}")
        self.get_logger().info(f"  • Covered cells: {covered_free_cells}")
        self.get_logger().info(f"  • Total coverage: {coverage_percent:.1f}%")
        self.get_logger().info(f"  • Accessible cells: {accessible_cells}")
        self.get_logger().info(f"  • Accessible coverage: {accessible_coverage_percent:.1f}%")
        
        return accessible_coverage_percent

    def print_complete_statistics(self):
        """Rozszerzone statystyki z jednolitym safety margin"""
        self.get_logger().info("=== CHOSET-PIGNON SAFE ANALYSIS ===")
        
        # DODAJ INFO O SAFETY MARGIN
        safety_margin_px = int(self.SAFETY_MARGIN_METERS / self.map_resolution)
        self.get_logger().info(f"🛡️ SAFETY MARGIN: {self.SAFETY_MARGIN_METERS}m ({safety_margin_px} pixels)")
        
        if self.cells:
            areas = [cell['area_m2'] for cell in self.cells]
            total_area = sum(areas)
            
            self.get_logger().info(f"📊 DECOMPOSITION:")
            self.get_logger().info(f"  • Total cells: {len(self.cells)}")
            self.get_logger().info(f"  • Cells with paths: {len(self.cell_paths)}")
            self.get_logger().info(f"  • Total area: {total_area:.2f} m²")
            
            # Info o bezpiecznej przestrzeni
            safe_area = np.sum(self.safe_space)
            free_area = np.sum(self.free_space)
            self.get_logger().info(f"  • Safe navigation area: {safe_area/free_area*100:.1f}% of free space")
            
            if self.cell_paths:
                total_waypoints = sum(cell['waypoint_count'] for cell in self.cell_paths.values())
                self.get_logger().info(f"🌀 SAFE BOUSTROPHEDON:")
                self.get_logger().info(f"  • Total waypoints: {total_waypoints}")
                self.get_logger().info(f"  • Average per cell: {total_waypoints/len(self.cell_paths):.1f}")
                
                # Oblicz całkowitą długość ścieżki
                complete_path = []
                for cell_name in self.cell_sequence:
                    if cell_name in self.cell_paths:
                        complete_path.extend(self.cell_paths[cell_name]['waypoints'])
                
                if complete_path:
                    total_distance_px = 0
                    for i in range(1, len(complete_path)):
                        prev_x, prev_y = complete_path[i-1]
                        curr_x, curr_y = complete_path[i]
                        total_distance_px += np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    
                    total_distance_m = total_distance_px * self.map_resolution
                    
                    self.get_logger().info(f"📏 SAFE PATH ANALYSIS:")
                    self.get_logger().info(f"  • Total path length: {total_distance_m:.1f} m")
                    self.get_logger().info(f"  • Safety margin applied: {self.SAFETY_MARGIN_METERS}m")
                    
                    # RZECZYWISTY CZAS vs ESTYMACJA
                    if self.total_real_time > 0:
                        self.get_logger().info(f"⏱️ REAL TIME: {self.total_real_time:.1f}s ({self.total_real_time/60:.1f} min)")
                        real_speed = total_distance_m / self.total_real_time if self.total_real_time > 0 else 0
                        self.get_logger().info(f"  • Real average speed: {real_speed:.2f} m/s")
                    elif hasattr(self, 'real_start_time') and self.real_start_time:
                        current_time = time.time() - self.real_start_time
                        self.get_logger().info(f"⏱️ CURRENT TIME: {current_time:.1f}s (running...)")
                    else:
                        self.get_logger().info(f"  • Estimated time (0.3 m/s): {total_distance_m/0.3:.1f}s")
            
            self.get_logger().info(f"🗺️ EXECUTION:")
            sequence_text = ' → '.join(self.cell_sequence) if len(self.cell_sequence) < 8 else f"{len(self.cell_sequence)} cells"
            self.get_logger().info(f"  • Sequence: {sequence_text}")
            
            # Informacje o czasach komórek
            if self.cell_completion_times:
                self.get_logger().info(f"📋 CELL TIMES:")
                for cell_name, completion_time in self.cell_completion_times.items():
                    if cell_name in self.cell_start_times:
                        cell_duration = completion_time - self.cell_start_times[cell_name]
                        self.get_logger().info(f"  • {cell_name}: {cell_duration:.1f}s")
            
            obstacles_count = sum(1 for cell in self.cells if cell.get('has_internal_obstacles', False))
            if obstacles_count > 0:
                self.get_logger().info(f"⚠️ Skipped cells (obstacles): {obstacles_count}")
            
            self.get_logger().info("✅ Safe analysis complete!")

            self.get_logger().info("🎯 Obliczam końcowe metryki całej trasy...")
            self.calculate_final_path_metrics()

    # === NAV2 INTEGRATION - ROZŁĄCZONE ŚCIEŻKI ===
    
    def odom_callback(self, msg):
        """Główny callback - ROZŁĄCZONE stany z safety margin"""
        if not self.map_loaded or not self.cell_sequence:
            return
            
        # Aktualizuj pozycję robota
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        robot_x = int((self.robot_position[0] - self.map_origin[0]) / self.map_resolution)
        robot_y = self.map_data.shape[0] - 1 - int((self.robot_position[1] - self.map_origin[1]) / self.map_resolution)
        self.robot_pixel_position = (robot_x, robot_y)
        
        # Dodaj do rzeczywistej ścieżki
        if self.robot_pixel_position:
            self.complete_real_path.append(self.robot_pixel_position)
        
        # ROZŁĄCZONA maszyna stanów
        if self.state == ExecutionState.WAITING_FOR_ODOM:
            self.state = ExecutionState.PLANNING
            self.execution_start_time = self.get_clock().now()
            self.real_start_time = time.time()
            self.get_logger().info(f"🚀 CHOSET STARTED with {self.SAFETY_MARGIN_METERS}m safety margin!")
            
        elif self.state == ExecutionState.PLANNING:
            if self.current_cell_index < len(self.cell_sequence):
                self.execute_navigation_to_cell()
            else:
                if self.real_start_time:
                    self.total_real_time = time.time() - self.real_start_time
                    self.get_logger().info(f"🏁 CHOSET FINISHED - Total real time: {self.total_real_time:.1f}s")
                self.state = ExecutionState.FINISHED

    def execute_navigation_to_cell(self):
        """FAZA 1: A* do komórki z safety margin"""
        cell_name = self.cell_sequence[self.current_cell_index]
        cell_data = self.cell_paths[cell_name]
        
        if self.current_cell_index == 0:
            self.first_astar_start = time.time()  # ← DODAJ TĘ LINIJKĘ

        if self.real_start_time:
            self.cell_start_times[cell_name] = time.time() - self.real_start_time
        
        self.state = ExecutionState.NAVIGATING_TO_CELL
        self.current_navigation_phase = "to_cell"
        
        current_robot_pos = self.robot_pixel_position
        cell_start = cell_data['start_point']
        
        # Sprawdź odległość
        distance_to_start = math.sqrt((current_robot_pos[0] - cell_start[0])**2 + 
                                     (current_robot_pos[1] - cell_start[1])**2)
        
        if distance_to_start < 10:
            self.get_logger().info(f"🎯 Robot blisko startu {cell_name} - przechodzę do boustrophedon")
            self.execute_boustrophedon_in_cell()
            return
        
        # A* z safety margin
        self.get_logger().info(f"🛤️ A* navigation to {cell_name} with {self.SAFETY_MARGIN_METERS}m safety")
        navigation_path = self.advanced_astar_navigation(current_robot_pos, cell_start)
        
        self.send_path_to_nav2(navigation_path, f"Safe A* Navigation to {cell_name}")

    def execute_boustrophedon_in_cell(self):
        """FAZA 2: Boustrophedon w komórce z GĘSTSZYMI punktami dla RPP"""
        cell_name = self.cell_sequence[self.current_cell_index]
        cell_data = self.cell_paths[cell_name]
        
        if self.current_cell_index == 0:
            self.coverage_with_inter_astar_start = time.time() 

        self.state = ExecutionState.EXECUTING_BOUSTROPHEDON
        self.current_navigation_phase = "boustrophedon"
        
        current_robot_pos = self.robot_pixel_position
        original_path = cell_data['waypoints']
        
        # POPRAWKA: Dodaj gęste punkty przejścia + smooth start
        if len(original_path) > 0:
            first_boustro_point = original_path[0]
            
            # 1. Gęsta ścieżka od robota do pierwszego punktu boustrophedon
            transition_path = self.generate_dense_line_points(
                current_robot_pos, first_boustro_point, spacing=None  # Auto spacing dla RPP
            )
            
            # 2. Zagęszczenie całej ścieżki boustrophedon
            dense_boustro_path = self.densify_boustrophedon_path(original_path, spacing=None)
            
            # 3. Połącz: przejście + gęsty boustrophedon (bez duplikowania pierwszego punktu)
            if len(transition_path) > 1 and len(dense_boustro_path) > 0:
                if transition_path[-1] == dense_boustro_path[0]:
                    modified_path = transition_path + dense_boustro_path[1:]
                else:
                    modified_path = transition_path + dense_boustro_path
            else:
                modified_path = transition_path + dense_boustro_path
        else:
            modified_path = [current_robot_pos]
        
        self.get_logger().info(f"🎯 Dense boustrophedon w {cell_name}: {len(modified_path)} punktów (orig: {len(original_path)})")
        
        self.send_path_to_nav2(modified_path, f"Dense Boustrophedon in {cell_name}")

    def densify_boustrophedon_path(self, path, spacing=None):
        """Zagęść całą ścieżkę boustrophedon dla RPP lookahead"""
        if len(path) <= 1:
            return path
        
        if spacing is None:
            # Auto spacing dla RPP
            lookahead_meters = 0.2
            optimal_spacing_meters = lookahead_meters * 0.4  # 40% lookahead dla extra gęstości
            spacing = max(2, int(optimal_spacing_meters / self.map_resolution))
        
        dense_path = []
        
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # Gęste punkty między każdymi dwoma punktami
            segment_points = self.generate_dense_line_points(start_point, end_point, spacing)
            
            if i == 0:
                dense_path.extend(segment_points)
            else:
                # Usuń duplikat punktu (pierwszy punkt segmentu = ostatni punkt poprzedniego)
                dense_path.extend(segment_points[1:])
        
        return dense_path

    def send_path_to_nav2(self, path_px, description):
        """Wyślij ścieżkę do Nav2 (wersja z safety margin)"""
        if not path_px or len(path_px) == 0:
            self.get_logger().error(f"Pusta ścieżka: {description}")
            self.handle_navigation_failure()
            return
            
        path_msg = self.create_ros_path(path_px)
        
        self.path_pub.publish(path_msg)
        self.current_path = path_msg
        
        goal_msg = FollowPath.Goal()
        goal_msg.path = path_msg
        
        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 serwer niedostępny!")
            self.handle_navigation_failure()
            return
        
        self.get_logger().info(f"🛤️ Wysyłam: {description} ({len(path_px)} punktów)")
        
        future = self.nav2_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.nav2_goal_response(f, description))

    def create_ros_path(self, path_px):
        """Konwertuj ścieżkę pikselową na ROS Path"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for i, (x, y) in enumerate(path_px):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            
            # Konwersja współrzędnych
            pose.pose.position.x = self.map_origin[0] + x * self.map_resolution
            pose.pose.position.y = self.map_origin[1] + (self.map_data.shape[0] - 1 - y) * self.map_resolution
            pose.pose.position.z = 0.0
            
            # Orientacja
            if i < len(path_px) - 1:
                next_x, next_y = path_px[i+1]
                dx = (next_x - x) * self.map_resolution
                dy = -(next_y - y) * self.map_resolution
                yaw = np.arctan2(dy, dx)
            else:
                yaw = 0.0
            
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = np.sin(yaw / 2)
            pose.pose.orientation.w = np.cos(yaw / 2)
            
            path_msg.poses.append(pose)
        
        return path_msg

    def nav2_goal_response(self, future, description):
        """Obsługa odpowiedzi Nav2"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f"❌ Nav2 odrzucił: {description}")
                self.handle_navigation_failure()
                return
                
            self.get_logger().info(f"✅ Nav2 zaakceptował: {description}")
            goal_handle.get_result_async().add_done_callback(
                lambda f: self.nav2_result_callback(f, description))
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd Nav2: {str(e)}")
            self.handle_navigation_failure()

    def nav2_result_callback(self, future, description):
        """Obsługa wyniku - ROZŁĄCZONE fazy z safety margin"""
        try:
            result = future.result().result
            self.get_logger().info(f"✅ Zakończono: {description}")
            
            if self.state == ExecutionState.NAVIGATING_TO_CELL:
                # A* zakończony - przejdź do boustrophedon
                self.get_logger().info(f"🎯 Safe A* zakończony - rozpoczynam safe boustrophedon")
                self.execute_boustrophedon_in_cell()
                
            elif self.state == ExecutionState.EXECUTING_BOUSTROPHEDON:
                # Boustrophedon zakończony
                cell_name = self.cell_sequence[self.current_cell_index]
                
                if self.real_start_time:
                    self.cell_completion_times[cell_name] = time.time() - self.real_start_time
                    if cell_name in self.cell_start_times:
                        duration = self.cell_completion_times[cell_name] - self.cell_start_times[cell_name]
                        self.get_logger().info(f"⏱️ {cell_name} completed safely in {duration:.1f}s")
                
                # Następna komórka
                self.current_cell_index += 1
                if self.current_cell_index >= len(self.cell_sequence):  # ostatnia komórka zakończona
                    if hasattr(self, 'coverage_with_inter_astar_start'):
                        time_without_first_astar = time.time() - self.coverage_with_inter_astar_start
                        self.get_logger().info(f"🎯 TIME WITHOUT FIRST A*: {time_without_first_astar:.1f}s (includes inter-zone A*)")
                    if hasattr(self, 'first_astar_start') and hasattr(self, 'coverage_with_inter_astar_start'):
                        first_astar_duration = self.coverage_with_inter_astar_start - self.first_astar_start
                        self.get_logger().info(f"📍 FIRST A* DURATION: {first_astar_duration:.1f}s (excluded from test)")
                self.state = ExecutionState.PLANNING
                self.get_logger().info(f"🔄 {cell_name} zakończona bezpiecznie - planowanie następnej")
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd wyniku: {str(e)}")
            self.handle_navigation_failure()
    
    def handle_navigation_failure(self):
        """Obsługa błędów nawigacji z safety margin"""
        if self.state == ExecutionState.NAVIGATING_TO_CELL:
            self.get_logger().warn("⚠️ Błąd safe A* - próbuję safe boustrophedon")
            self.execute_boustrophedon_in_cell()
            
        elif self.state == ExecutionState.EXECUTING_BOUSTROPHEDON:
            self.get_logger().warn("⚠️ Błąd safe boustrophedon - następna komórka")
            
            if self.current_cell_index < len(self.cell_sequence):
                cell_name = self.cell_sequence[self.current_cell_index]
                if self.real_start_time:
                    self.cell_completion_times[cell_name] = time.time() - self.real_start_time
            
            self.current_cell_index += 1
            self.state = ExecutionState.PLANNING
        else:
            self.get_logger().warn("⚠️ Błąd nawigacji - kontynuuję bezpiecznie")
            self.current_cell_index += 1
            self.state = ExecutionState.PLANNING
    

    def print_path_metrics(self, path_px):
        """
        Oblicz i wydrukuj podstawowe metryki ścieżki - SKOPIOWANE Z T1.PY
        """
        if not path_px or len(path_px) < 2:
            return
        
        # DYSTANS
        total_distance = 0.0
        for i in range(len(path_px) - 1):
            x1, y1 = path_px[i]
            x2, y2 = path_px[i + 1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.map_resolution
            total_distance += distance
        
        # ZAKRĘTY
        turn_count = 0
        for i in range(1, len(path_px) - 1):
            p1, p2, p3 = path_px[i-1], path_px[i], path_px[i+1]
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 > 0 and len2 > 0:
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180.0 / np.pi
                
                if angle > 45.0:
                    turn_count += 1
        
        # REDUNDANCJA - po prostu sprawdź duplikaty!
        unique_points = len(set(path_px))  # unikalne punkty
        total_points = len(path_px)        # wszystkie punkty
        duplicates = total_points - unique_points
        redundancy_percent = (duplicates / total_points) * 100
        
        # PRINT - ZMIENIONE NA CHOSET
        print(f"CHOSET DYSTANS: {total_distance:.2f}m | ZAKRĘTY: {turn_count} | REDUNDANCJA: {redundancy_percent:.1f}% ({duplicates}/{total_points})")

    def calculate_final_path_metrics(self):
        """
        Prosta wersja - połącz wszystkie ścieżki i policz metryki
        """
        if not self.cell_sequence or not self.cell_paths:
            self.get_logger().warn("❌ Brak danych do metryk")
            return
        
        # Połącz CAŁĄ trasę: boustrophedon + A* roadmaps
        complete_path = []
        
        for i, cell_name in enumerate(self.cell_sequence):
            if cell_name in self.cell_paths:
                # 1. Dodaj ścieżkę boustrophedon w komórce
                cell_waypoints = self.cell_paths[cell_name]['waypoints']
                complete_path.extend(cell_waypoints)
                
                # 2. Dodaj A* roadmap do następnej komórki (jeśli istnieje)
                if i < len(self.cell_sequence) - 1:
                    next_cell = self.cell_sequence[i + 1]
                    if next_cell in self.cell_paths:
                        start_point = self.cell_paths[cell_name]['end_point']
                        end_point = self.cell_paths[next_cell]['start_point']
                        
                        # Generuj A* roadmap
                        try:
                            roadmap_path = self.advanced_astar_navigation(start_point, end_point)
                            if roadmap_path:
                                complete_path.extend(roadmap_path)
                        except Exception as e:
                            self.get_logger().warn(f"⚠️ Błąd A* roadmap: {e}")
                            # Bez roadmap - kontynuuj
        
        if not complete_path:
            self.get_logger().warn("❌ Pusta ścieżka końcowa")
            return
        
        self.get_logger().info(f"🎯 Obliczam metryki dla {len(complete_path)} punktów (boustrophedon + A* roadmaps)")
        
        # Wywołaj prostą funkcję metryk (DOKŁADNIE jak w t1.py)
        self.print_path_metrics(complete_path)

def main():
    rclpy.init()
    planner = ChosetPignonNav2Planner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Zatrzymywanie plannera...")
    finally:
        # Zapisz końcowy czas
        if hasattr(planner, 'real_start_time') and planner.real_start_time:
            planner.total_real_time = time.time() - planner.real_start_time
            planner.get_logger().info(f"🏁 FINAL SAFE CHOSET TIME: {planner.total_real_time:.1f}s")
        
        planner.destroy_node()

if __name__ == '__main__':
    main()