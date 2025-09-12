#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import yaml
from matplotlib.patches import Polygon, Rectangle
from scipy import ndimage
from skimage.segmentation import watershed
import math
import heapq
import time
from enum import Enum

class ExecutionState(Enum):
    WAITING_FOR_ODOM = 0
    PLANNING = 1
    NAVIGATING_TO_CELL = 2  # NOWY STAN - A* do komórki
    EXECUTING_BOUSTROPHEDON = 3  # NOWY STAN - boustrophedon w komórce
    FINISHED = 4

class SimpleBoustrophedonNav2Planner(Node):
    # Podstawowa konfiguracja
    MIN_CELL_AREA = 50
    PEAK_MIN_DISTANCE = 45
    SAFETY_MARGIN_METERS = 0.32
    ROBOT_WIDTH_METERS = 0.2
    MIN_RECT_COVERAGE = 0.2
    
    def __init__(self):
        super().__init__('simple_boustrophedon_nav2_planner')
        
        # Map name
        self.map_name = "map_test"
        
        # Ustaw parametr use_sim_time na True
        param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([param])
        
        # Pliki mapy
        self.MAP_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.pgm")
        self.YAML_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.yaml")
        
        # Stan wykonania
        self.state = ExecutionState.WAITING_FOR_ODOM
        self.robot_position = None
        self.robot_pixel_position = None
        self.map_loaded = False
        
        # DODANE: Rzeczywiste śledzenie czasu (jak w choset)
        self.real_start_time = None
        self.cell_start_times = {}
        self.cell_completion_times = {}
        self.total_real_time = 0.0
        self.complete_real_path = []  # Rzeczywista ścieżka robota
        
        # Dane z oryginalnego b2.py
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.safe_free_mask = None
        self.watershed_cells = []
        self.coverage_rects = []
        self.cell_zones = {}
        self.cell_sequence = []
        self.roadmap_paths = {}
        
        # NOWE: Wykonanie Nav2 z rozłączonymi ścieżkami
        self.current_cell_index = 0
        self.current_path = None
        self.execution_start_time = None
        self.current_navigation_phase = "none"  # "to_cell" lub "boustrophedon"
        
        # QoS dla costmap
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
        
        # Inicjalizacja - uruchom oryginalną logikę b2.py
        self.run_b2_decomposition()
    
    def find_map_file(self, filename):
        for path in ['.', os.path.dirname(__file__)]:
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                return filepath
        return filename

    def run_b2_decomposition(self):
        """Uruchom oryginalną dekompozycję z b2.py"""
        try:
            self.load_map()
            self.create_watershed_cells_with_gentle_cleaning()
            self.create_safe_rectangles()
            self.generate_safe_boustrophedon()
            self.plan_cell_sequence()
            self.create_inter_cell_roadmap()
            
            # WIZUALIZACJE OSOBNO - BEZ ZAPISYWANIA PNG
            self.visualize_watershed_window()
            self.visualize_rectangles_window()
            self.visualize_sequence_window()
            self.visualize_paths_window()
            self.visualize_coverage_window()
            
            self.get_logger().info(f"🎯 B2 Decomposition complete: {len(self.cell_zones)} cells ready")
            
        except Exception as e:
            self.get_logger().error(f"Błąd dekompozycji B2: {str(e)}")


    def load_map(self):
        """Wczytaj mapę (z b2.py + publikacja costmap)"""
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
            
            # Publikuj costmap
            self.publish_costmap()
            self.map_loaded = True
            
        except Exception as e:
            self.get_logger().error(f"Błąd ładowania mapy: {str(e)}")
            raise

    def publish_costmap(self):
        """Publikuje mapę do costmapa (z choset.py)"""
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

    # ============== ORYGINALNA LOGIKA B2.PY ==============
    
    def create_watershed_cells_with_gentle_cleaning(self):
        """Watershed z DELIKATNYM czyszczeniem (oryginalna logika b2.py)"""
        free_space = (self.map_data == 0).astype(np.uint8)
        
        # Safety margin
        safety_pixels = int(self.SAFETY_MARGIN_METERS / self.map_resolution)
        if safety_pixels > 0:
            obstacle_mask = (self.map_data == 100).astype(np.uint8)
            kernel = np.ones((safety_pixels * 2 + 1, safety_pixels * 2 + 1), np.uint8)
            dilated_obstacles = cv2.dilate(obstacle_mask, kernel, iterations=1)
            safe_free = (free_space == 1) & (dilated_obstacles == 0)
        else:
            safe_free = free_space
        
        # Minimalne czyszczenie (z b2.py)
        safe_free_cleaned = self.remove_small_islands_minimal(safe_free)
        self.safe_free_mask = safe_free_cleaned
        
        # Watershed
        distance = ndimage.distance_transform_edt(safe_free_cleaned)
        peaks = self.find_local_maxima(distance)
        
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (y, x) in enumerate(peaks):
            markers[y, x] = i + 1
        
        labels = watershed(-distance, markers, mask=safe_free_cleaned)
        self.watershed_labels = labels
        
        # Utwórz komórki
        for label in np.unique(labels):
            if label == 0:
                continue
                
            mask = (labels == label).astype(np.uint8)
            area = np.sum(mask)
            
            if area < self.MIN_CELL_AREA:
                continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            self.watershed_cells.append({
                'id': len(self.watershed_cells) + 1,
                'label': label,
                'contour': [p[0] for p in approx_contour],
                'mask': mask,
                'centroid': (cX, cY),
                'bbox': cv2.boundingRect(largest_contour),
                'area': area
            })

    def remove_small_islands_minimal(self, binary_mask):
        """Minimalne czyszczenie (z b2.py)"""
        labeled_mask, num_labels = ndimage.label(binary_mask)
        component_areas = ndimage.sum(binary_mask, labeled_mask, range(1, num_labels + 1))
        
        cleaned_mask = binary_mask.copy()
        
        for i, area in enumerate(component_areas):
            if area <= 4:
                cleaned_mask[labeled_mask == (i + 1)] = False
        
        # Wypełnij małe dziury
        inverted = ~cleaned_mask
        labeled_holes, num_holes = ndimage.label(inverted)
        hole_areas = ndimage.sum(inverted, labeled_holes, range(1, num_holes + 1))
        
        for i, area in enumerate(hole_areas):
            if area <= 8:
                cleaned_mask[labeled_holes == (i + 1)] = True
        
        return cleaned_mask

    def create_safe_rectangles(self):
        """Utwórz TYLKO 100% bezpieczne prostokąty (z b2.py)"""
        h, w = self.map_data.shape
        robot_width_pixels = int(self.ROBOT_WIDTH_METERS / self.map_resolution *1.7)
        
        for cell in self.watershed_cells:
            cell_label = cell['label']
            cell_bbox = cell['bbox']
            x_start, y_start, bbox_w, bbox_h = cell_bbox
            
            cell_rects = []
            
            for y in range(y_start, y_start + bbox_h, robot_width_pixels):
                for x in range(x_start, x_start + bbox_w, robot_width_pixels):
                    
                    x_end = min(x + robot_width_pixels, w)
                    y_end = min(y + robot_width_pixels, h)
                    
                    if x >= x_end or y >= y_end:
                        continue
                    
                    # Sprawdzenie bezpieczeństwa
                    rect_region = self.safe_free_mask[y:y_end, x:x_end]
                    safe_pixels = np.sum(rect_region)
                    total_pixels = rect_region.size
                    
                    if total_pixels == 0:
                        continue
                    
                    safe_coverage = safe_pixels / total_pixels
                    
                    # Sprawdź przynależność do komórki
                    cell_region = self.watershed_labels[y:y_end, x:x_end]
                    cell_pixels = np.sum(cell_region == cell_label)
                    cell_coverage = cell_pixels / total_pixels
                    
                    # Środek i rogi
                    center_x, center_y = x + (x_end - x) // 2, y + (y_end - y) // 2
                    
                    # Sprawdź czy środek i wszystkie rogi są bezpieczne
                    corners_safe = all([
                        self.is_point_ultra_safe((x, y)),
                        self.is_point_ultra_safe((x_end-1, y)),
                        self.is_point_ultra_safe((x, y_end-1)),
                        self.is_point_ultra_safe((x_end-1, y_end-1)),
                        self.is_point_ultra_safe((center_x, center_y))
                    ])
                    
                    # TYLKO jeśli prostokąt jest w 100% bezpieczny
                    if safe_coverage == 1.0 and cell_coverage > 0.8 and corners_safe:
                        rect = {
                            'id': len(self.coverage_rects) + 1,
                            'cell_id': cell['id'],
                            'bbox': (x, y, x_end - x, y_end - y),
                            'center': (center_x, center_y),
                            'safe': True,
                            'used_in_path': False
                        }
                        
                        cell_rects.append(rect)
                        self.coverage_rects.append(rect)

    def generate_safe_boustrophedon(self):
        """ULEPSZONE generowanie ścieżek z auto spacing dla Pure Pursuit (z choset.py)"""
        for cell in self.watershed_cells:
            cell_id = cell['id']
            cell_name = f"C{cell_id}"
            
            # Znajdź prostokąty w tej komórce
            cell_rects = [r for r in self.coverage_rects if r['cell_id'] == cell_id]
            
            if not cell_rects:
                continue
            
            # Utwórz bezpieczną ścieżkę zygzak z walidacją linii
            waypoints = self.create_safe_zigzag_with_validation(cell_rects)
            
            # ULEPSZONE GĘSTE PUNKTY z auto spacing dla Pure Pursuit
            if waypoints:
                dense_waypoints = self.densify_waypoints_for_pure_pursuit(waypoints, spacing=None)  # Auto spacing
                
                self.cell_zones[cell_name] = {
                    'cell_id': cell_id,
                    'waypoints': dense_waypoints,
                    'start_point': dense_waypoints[0],
                    'end_point': dense_waypoints[-1],
                    'waypoint_count': len(dense_waypoints)
                }
                
                self.get_logger().info(f"Enhanced {cell_name}: {len(dense_waypoints)} auto-spaced waypoints (orig: {len(waypoints)})")

    def densify_waypoints_for_pure_pursuit(self, waypoints, spacing=None):
        """ULEPSZONE zagęszczenie waypoints z auto spacing (z choset.py)"""
        if len(waypoints) <= 1:
            return waypoints
        
        if spacing is None:
            # AUTO SPACING dla Pure Pursuit
            lookahead_meters = 0.2
            optimal_spacing_meters = lookahead_meters * 0.9  # 50% lookahead dla pewności
            spacing = max(2, int(optimal_spacing_meters / self.map_resolution))
            
            self.get_logger().info(f"🎯 Auto boustrophedon spacing: {spacing} px ({spacing * self.map_resolution:.3f}m)")
        
        dense_waypoints = []
        
        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            
            # Dodaj gęste punkty między start a end
            segment_points = self.generate_dense_line_points(start_point, end_point, spacing)
            
            if i == 0:
                # Pierwszy segment - dodaj wszystkie punkty
                dense_waypoints.extend(segment_points)
            else:
                # Kolejne segmenty - pomiń pierwszy punkt (duplikat)
                dense_waypoints.extend(segment_points[1:])
        
        return dense_waypoints

    def generate_dense_line_points(self, start_point, end_point, spacing=None):
        """ULEPSZONE generowanie punktów dostosowane do Pure Pursuit (z choset.py)"""
        if spacing is None:
            # AUTO SPACING na podstawie lookahead distance
            lookahead_meters = 0.2  # Twój lookahead distance
            optimal_spacing_meters = lookahead_meters * 0.6  # 60% lookahead = bezpieczny odstęp
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

    def create_safe_zigzag_with_validation(self, safe_rects):
        """Utwórz bezpieczny zygzak z walidacją (z b2.py)"""
        if not safe_rects:
            return []
        
        waypoints = []
        
        # Grupuj według wierszy Y
        y_groups = {}
        for rect in safe_rects:
            y_key = rect['bbox'][1]
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(rect)
        
        # Sortuj wiersze
        sorted_y_keys = sorted(y_groups.keys())
        
        for row_idx, y_key in enumerate(sorted_y_keys):
            row_rects = sorted(y_groups[y_key], key=lambda r: r['bbox'][0])
            
            # Znajdź ciągłe segmenty w wierszu
            segments = self.find_continuous_segments_in_row(row_rects)
            
            # Zygzak: co drugi wiersz w przeciwnym kierunku
            if row_idx % 2 == 0:  # Lewo -> prawo
                for segment in segments:
                    for rect in segment:
                        center = rect['center']
                        if self.is_point_ultra_safe(center):
                            if (not waypoints or 
                                len(waypoints) == 0 or
                                self.is_connection_reasonable(waypoints[-1], center) or
                                self.is_line_completely_safe(waypoints[-1], center)):
                                waypoints.append(center)
                                rect['used_in_path'] = True
            else:  # Prawo -> lewo
                for segment in reversed(segments):
                    for rect in reversed(segment):
                        center = rect['center']
                        if self.is_point_ultra_safe(center):
                            if (not waypoints or 
                                len(waypoints) == 0 or
                                self.is_connection_reasonable(waypoints[-1], center) or
                                self.is_line_completely_safe(waypoints[-1], center)):
                                waypoints.append(center)
                                rect['used_in_path'] = True
        
        return waypoints

    def find_continuous_segments_in_row(self, row_rects):
        """Znajdź ciągłe segmenty prostokątów w wierszu (z b2.py)"""
        if not row_rects:
            return []
        
        segments = []
        current_segment = [row_rects[0]]
        
        robot_width_pixels = int(self.ROBOT_WIDTH_METERS / self.map_resolution)
        
        for i in range(1, len(row_rects)):
            prev_rect = row_rects[i-1]
            curr_rect = row_rects[i]
            
            prev_x_end = prev_rect['bbox'][0] + prev_rect['bbox'][2]
            curr_x_start = curr_rect['bbox'][0]
            
            # Jeśli prostokąty są obok siebie
            if curr_x_start - prev_x_end <= robot_width_pixels:
                current_segment.append(curr_rect)
            else:
                if len(current_segment) > 0:
                    segments.append(current_segment)
                current_segment = [curr_rect]
        
        if len(current_segment) > 0:
            segments.append(current_segment)
        
        return segments

    def is_connection_reasonable(self, point1, point2):
        """Sprawdź czy odległość między punktami jest rozsądna (z b2.py)"""
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        max_reasonable_distance = self.ROBOT_WIDTH_METERS / self.map_resolution * 3
        return distance <= max_reasonable_distance

    def is_line_completely_safe(self, point1, point2):
        """Sprawdzenie czy linia jest bezpieczna (z b2.py)"""
        x1, y1 = int(point1[0]), int(point1[1])
        x2, y2 = int(point2[0]), int(point2[1])
        
        points_on_line = self.get_line_points(x1, y1, x2, y2)
        
        for i, (x, y) in enumerate(points_on_line):
            if i % 2 != 0:
                continue
                
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = x + dx, y + dy
                    
                    if (0 <= check_x < self.safe_free_mask.shape[1] and 
                        0 <= check_y < self.safe_free_mask.shape[0]):
                        if not self.safe_free_mask[check_y, check_x]:
                            return False
                    else:
                        return False
        
        return True

    def get_line_points(self, x1, y1, x2, y2):
        """Algorytm Bresenhama (z b2.py)"""
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points

    def is_point_safe(self, point):
        """Sprawdź czy punkt jest bezpieczny (z b2.py)"""
        x, y = int(point[0]), int(point[1])
        if (0 <= x < self.safe_free_mask.shape[1] and 
            0 <= y < self.safe_free_mask.shape[0]):
            return self.safe_free_mask[y, x]
        return False

    def is_point_ultra_safe(self, point):
        """ULTRA-sprawdzenie czy punkt i jego okolice są bezpieczne (z b2.py)"""
        x, y = int(point[0]), int(point[1])
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.safe_free_mask.shape[1] and 
                    0 <= check_y < self.safe_free_mask.shape[0]):
                    if not self.safe_free_mask[check_y, check_x]:
                        return False
                else:
                    return False
        return True

    def find_local_maxima(self, distance):
        """Znajdź lokalne maksima dla watershed (z b2.py)"""
        h, w = distance.shape
        peaks = []
        
        for y in range(self.PEAK_MIN_DISTANCE, h - self.PEAK_MIN_DISTANCE):
            for x in range(self.PEAK_MIN_DISTANCE, w - self.PEAK_MIN_DISTANCE):
                if distance[y, x] < 5:
                    continue
                
                neighborhood = distance[y-20:y+21, x-20:x+21]
                
                if distance[y, x] == np.max(neighborhood):
                    too_close = False
                    for py, px in peaks:
                        if np.sqrt((y-py)**2 + (x-px)**2) < self.PEAK_MIN_DISTANCE:
                            too_close = True
                            break
                    
                    if not too_close:
                        peaks.append((y, x))
        
        return peaks

    def plan_cell_sequence(self):
        """TSP - kolejność komórek (z b2.py)"""
        if not self.cell_zones:
            return
        
        cell_names = list(self.cell_zones.keys())
        
        if len(cell_names) <= 1:
            self.cell_sequence = cell_names
            return
        
        # Najbliższy sąsiad
        visited = set()
        current_cell = cell_names[0]
        self.cell_sequence = [current_cell]
        visited.add(current_cell)
        
        while len(visited) < len(cell_names):
            min_distance = float('inf')
            next_cell = None
            
            current_end = self.cell_zones[current_cell]['end_point']
            
            for cell_name in cell_names:
                if cell_name in visited:
                    continue
                
                cell_start = self.cell_zones[cell_name]['start_point']
                distance = math.sqrt((current_end[0] - cell_start[0])**2 + 
                                   (current_end[1] - cell_start[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    next_cell = cell_name
            
            if next_cell:
                self.cell_sequence.append(next_cell)
                visited.add(next_cell)
                current_cell = next_cell

    def create_inter_cell_roadmap(self):
        """ULEPSZONA roadmap z enhanced A* między komórkami"""
        for i in range(len(self.cell_sequence) - 1):
            current_cell = self.cell_sequence[i]
            next_cell = self.cell_sequence[i + 1]
            
            start = self.cell_zones[current_cell]['end_point']
            goal = self.cell_zones[next_cell]['start_point']
            
            # UŻYJ ULEPSZONEGO A*
            path = self.advanced_astar_navigation(start, goal)
            
            self.roadmap_paths[f"{current_cell}->{next_cell}"] = {
                'from_cell': current_cell,
                'to_cell': next_cell,
                'path': path
            }
            
            self.get_logger().info(f"Enhanced roadmap {current_cell}->{next_cell}: {len(path)} auto-spaced points")

    # ============== ENHANCED A* Z CHOSET.PY ==============
    
    def advanced_astar_navigation(self, start, goal):
        """ULEPSZONA A* nawigacja z choset.py - 8-kierunkowa, auto spacing dla Pure Pursuit"""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        h, w = self.safe_free_mask.shape
        
        self.get_logger().info(f"🔍 Enhanced A* navigation with {self.SAFETY_MARGIN_METERS}m safety margin")
        
        closed_set = set()
        open_set = {start}
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda point: f_score.get(point, float('inf')))
            
            # PRECYZYJNA tolerancja zakończenia
            if current == goal or self.heuristic(current, goal) <= 1:
                path = self.reconstruct_path(came_from, current)
                
                # WYMUSZAJ dokładny koniec w goal
                if path and path[-1] != goal:
                    if self.is_point_ultra_safe(goal):
                        path.append(goal)
                        
                # AUTO SPACING dla Pure Pursuit - jak w choset.py
                return self.densify_path(path, spacing=None)  # Auto spacing
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 8-KIERUNKOWA nawigacja dla lepszej płynności (jak w choset.py)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                
                if not (0 <= nx < w and 0 <= ny < h) or neighbor in closed_set:
                    continue
                
                # ULTRA SAFE sprawdzenie
                if not self.is_point_ultra_safe((nx, ny)):
                    continue
                
                if not self.is_safe_path_enhanced(current, neighbor):
                    continue
                
                # Koszt diagonal vs prosty (jak w choset.py)
                step_cost = math.sqrt(dx*dx + dy*dy)
                tentative_g_score = g_score[current] + step_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
        
        self.get_logger().warn(f"⚠️ Enhanced A* nie znalazł ścieżki - fallback")
        # Fallback z auto spacing dla Pure Pursuit
        fallback = self.generate_dense_line_points(start, goal, spacing=None)  # Auto spacing
        if fallback and fallback[-1] != goal:
            fallback[-1] = goal  # Wymuszaj dokładny koniec
        return fallback

    def densify_path(self, path, spacing=None):
        """ULEPSZONE zagęszczenie ścieżki A* dostosowane do Pure Pursuit (z choset.py)"""
        if len(path) <= 1:
            return path
        
        if spacing is None:
            # AUTO SPACING dla Pure Pursuit - jak w choset.py
            lookahead_meters = 0.2  # Standardowy lookahead distance dla Pure Pursuit
            optimal_spacing_meters = lookahead_meters * 0.6  # 60% lookahead dla bezpieczeństwa
            spacing = max(2, int(optimal_spacing_meters / self.map_resolution))
            
            self.get_logger().info(f"🎯 Auto A* spacing: {spacing} px ({spacing * self.map_resolution:.3f}m) dla lookahead {lookahead_meters}m")
        
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
        """Ulepszona heurystyka - Euclidean dla lepszych diagonal moves"""
        # Euclidean distance dla 8-kierunkowej nawigacji
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def reconstruct_path(self, came_from, current):
        """Rekonstruuj ścieżkę z path smoothing"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        
        # OPTIONAL: Basic path smoothing
        return self.smooth_path(total_path)

    def smooth_path(self, path):
        """Podstawowe wygładzenie ścieżki - usuwa niepotrzebne punkty"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            # Sprawdź jak daleko możemy "skoczyć" w linii prostej
            j = len(path) - 1
            
            while j > i + 1:
                if self.is_line_completely_safe(path[i], path[j]):
                    # Możemy przeskoczyć bezpośrednio do punktu j
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # Nie można przeskoczyć - dodaj następny punkt
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed

    def is_safe_path_enhanced(self, p1, p2):
        """ULEPSZONE sprawdzenie bezpieczeństwa ścieżki (z choset.py)"""
        x1, y1 = p1
        x2, y2 = p2
        
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            
            # ULTRA SAFE sprawdzenie każdego punktu na linii
            if not self.is_point_ultra_safe((x, y)):
                return False
        
        return True

    # ============== WIZUALIZACJA (ORYGINALNA B2.PY) ==============
    
    def visualize_watershed_window(self):
        """OKNO 1: SAMO WATERSHED"""
        if not self.watershed_cells:
            self.get_logger().warn("Brak komórek watershed")
            return
        
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200
        display_map[self.map_data == 100] = 0
        display_map[self.map_data == -1] = 100
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.watershed_cells)))
        
        for i, cell in enumerate(self.watershed_cells):
            contour = np.array(cell['contour'])
            polygon = Polygon(contour, fill=True, alpha=0.4, 
                            color=colors[i], edgecolor='red', linewidth=2)
            ax.add_patch(polygon)
            ax.text(cell['centroid'][0], cell['centroid'][1], f"C{cell['id']}", 
                color='black', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))
        
        ax.set_title(f'Watershed Decomposition (minimalne czyszczenie)\n{len(self.watershed_cells)} komórek')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić watershed (headless environment)")

    def visualize_rectangles_window(self):
        """OKNO 2: Watershed + Prostokąty pomocnicze"""
        if not self.coverage_rects:
            self.get_logger().warn("Brak prostokątów pomocniczych")
            return
        
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200
        display_map[self.map_data == 100] = 0
        display_map[self.map_data == -1] = 100
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        rect_colors = plt.cm.viridis(np.linspace(0, 1, len(self.watershed_cells)))
        
        # Prostokąty pomocnicze z kolorami komórek
        for rect in self.coverage_rects:
            x, y, w, h = rect['bbox']
            cell_id = rect['cell_id']
            color = rect_colors[(cell_id - 1) % len(rect_colors)]
            
            rectangle = Rectangle((x, y), w, h, 
                                fill=True, alpha=0.6, 
                                color=color,
                                edgecolor='black', linewidth=1)
            rectangle.set_edgecolor('black')  # WYMUŚ czarny
            ax.add_patch(rectangle)
        
        # Komórki watershed
        for i, cell in enumerate(self.watershed_cells):
            contour = np.array(cell['contour'])
            polygon = Polygon(contour, fill=False, 
                            edgecolor='red', linewidth=3, alpha=0.8)
            ax.add_patch(polygon)
            ax.text(cell['centroid'][0], cell['centroid'][1], f"C{cell['id']}", 
                color='red', fontsize=16, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        
        ax.set_title(f'Safe Rectangles {self.ROBOT_WIDTH_METERS}m (tylko bezpieczne)\n{len(self.watershed_cells)} komórek, {len(self.coverage_rects)} kwadratów')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić prostokątów (headless environment)")

    def visualize_sequence_window(self):
        """OKNO 3: Kolejność odwiedzania"""
        if not self.cell_sequence:
            self.get_logger().warn("Brak sekwencji komórek")
            return
        
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200
        display_map[self.map_data == 100] = 0
        display_map[self.map_data == -1] = 100
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        rect_colors = plt.cm.viridis(np.linspace(0, 1, len(self.watershed_cells)))
        
        # Prostokąty z kolorami
        for rect in self.coverage_rects:
            x, y, w, h = rect['bbox']
            cell_id = rect['cell_id']
            color = rect_colors[(cell_id - 1) % len(rect_colors)]
            
            rectangle = Rectangle((x, y), w, h, 
                                fill=True, alpha=0.5, 
                                color=color,
                                edgecolor='black', linewidth=1)
            rectangle.set_edgecolor('black')  # WYMUŚ czarny
            ax.add_patch(rectangle)
        
        # Komórki z numeracją kolejności
        for i, cell in enumerate(self.watershed_cells):
            contour = np.array(cell['contour'])
            cell_name = f"C{cell['id']}"
            
            polygon = Polygon(contour, fill=False, 
                            edgecolor='black', linewidth=2)
            ax.add_patch(polygon)
            
            if cell_name in self.cell_sequence:
                seq_num = self.cell_sequence.index(cell_name) + 1
                ax.text(cell['centroid'][0], cell['centroid'][1], str(seq_num), 
                    color='white', fontsize=18, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.9, boxstyle='circle,pad=0.4'))
        
        ax.set_title(f'Execution Sequence\nSekwencja: {" → ".join(self.cell_sequence)}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić sekwencji (headless environment)")

    def visualize_paths_window(self):
        """OKNO 4: Enhanced Boustrophedon + Enhanced Roadmap + KWADRATY NA ŚCIEŻCE"""
        if not self.cell_zones:
            self.get_logger().warn("Brak ścieżek do wizualizacji")
            return
        
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200
        display_map[self.map_data == 100] = 0
        display_map[self.map_data == -1] = 100
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.watershed_cells)))
        rect_colors = plt.cm.viridis(np.linspace(0, 1, len(self.watershed_cells)))
        
        # Komórki (półprzeźroczyste)
        for i, cell in enumerate(self.watershed_cells):
            contour = np.array(cell['contour'])
            polygon = Polygon(contour, fill=True, alpha=0.15, 
                            color=colors[i], edgecolor='red', linewidth=1)
            ax.add_patch(polygon)
        
        # WSZYSTKIE prostokąty (szare, półprzeźroczyste)
        for rect in self.coverage_rects:
            x, y, w, h = rect['bbox']
            rectangle = Rectangle((x, y), w, h, 
                                fill=True, alpha=0.2, 
                                color='gray',
                                edgecolor='black', linewidth=1)
            rectangle.set_edgecolor('black')  # WYMUŚ czarny
            ax.add_patch(rectangle)
        
        # KWADRATY UŻYWANE W ŚCIEŻCE (kolorowe, wyraziste)
        for rect in self.coverage_rects:
            if rect.get('used_in_path', False):
                x, y, w, h = rect['bbox']
                cell_id = rect['cell_id']
                color = rect_colors[(cell_id - 1) % len(rect_colors)]
                
                rectangle = Rectangle((x, y), w, h, 
                                    fill=True, alpha=0.8, 
                                    color=color,
                                    edgecolor='black', linewidth=1)
                rectangle.set_edgecolor('black')  # WYMUŚ czarny
                ax.add_patch(rectangle)
        
        # ENHANCED ŚCIEŻKI - TYLKO BEZPIECZNE SEGMENTY
        total_coverage = 0
        cell_colors = ['green', 'red', 'yellow', 'white', 'purple', 'brown', 'magenta', 'cyan', 'red', 'purple']
        
        for i, (cell_name, cell_data) in enumerate(self.cell_zones.items()):
            waypoints = cell_data['waypoints']
            color = cell_colors[i % len(cell_colors)]
            
            if len(waypoints) > 1:
                # Użyj funkcji rysowania bezpiecznych ścieżek
                self.draw_path_through_rectangles(ax, waypoints, color, cell_name)
                
                # PUNKTY WEJŚCIA/WYJŚCIA STREFY
                start = cell_data['start_point']
                end = cell_data['end_point']
                
                # Duże, wyraźne punkty
                ax.plot(start[0], start[1], 'o', color='lime', markersize=12, 
                        markeredgecolor='black', markeredgewidth=3, label='Start' if i == 0 else "")
                ax.plot(end[0], end[1], 's', color='red', markersize=12, 
                        markeredgecolor='black', markeredgewidth=3, label='End' if i == 0 else "")
                
                # Etykiety START/END
                ax.text(start[0], start[1] - 12, f"START\n{cell_name}", 
                    color='lime', fontsize=8, fontweight='bold', ha='center',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                ax.text(end[0], end[1] + 12, f"END\n{cell_name}", 
                    color='red', fontsize=8, fontweight='bold', ha='center',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                
                total_coverage += len(waypoints)
        
        # ENHANCED A* roadmap między komórkami - GRUBE LINIE
        total_roadmap = 0
        for path_data in self.roadmap_paths.values():
            path = path_data['path']
            if len(path) > 1:
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                ax.plot(x_coords, y_coords, 'b--', linewidth=4, alpha=0.9, 
                        label='Enhanced A* Roadmap' if total_roadmap == 0 else "")
                ax.plot(path[0][0], path[0][1], 'bs', markersize=10, alpha=0.8)
                ax.plot(path[-1][0], path[-1][1], 'bs', markersize=10, alpha=0.8)
                total_roadmap += len(path)
        
        # DODAJ RZECZYWISTY CZAS JEŚLI DOSTĘPNY
        time_text = ""
        if self.total_real_time > 0:
            time_text = f" | Real time: {self.total_real_time:.1f}s"
        elif hasattr(self, 'real_start_time') and self.real_start_time:
            current_time = time.time() - self.real_start_time
            time_text = f" | Current time: {current_time:.1f}s"
        
        ax.set_title(f'Enhanced Boustrophedon + Smooth A*{time_text}\n{total_coverage} waypoints, {total_roadmap} roadmap')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        
        plt.tight_layout()
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić ścieżek (headless environment)")

    def visualize_coverage_window(self):
        """OKNO 5: ENHANCED coverage analysis with auto spacing info"""
        if not self.cell_zones:
            self.get_logger().warn("Brak komórek do analizy pokrycia")
            return
        
        # Przygotuj mapę
        display_map = np.zeros_like(self.map_data, dtype=np.uint8)
        display_map[self.map_data == 0] = 200    # wolna przestrzeń - jasna
        display_map[self.map_data == 100] = 0    # przeszkody - czarne
        display_map[self.map_data == -1] = 100   # nieznane - szare
        
        # Połącz wszystkie ścieżki w jedną kompletną trasę
        complete_path = []
        for cell_name in self.cell_sequence:
            if cell_name in self.cell_zones:
                cell_waypoints = self.cell_zones[cell_name]['waypoints']
                complete_path.extend(cell_waypoints)
                
                # Dodaj też Enhanced A* roadmap między komórkami
                roadmap_key = None
                current_index = self.cell_sequence.index(cell_name)
                if current_index < len(self.cell_sequence) - 1:
                    next_cell = self.cell_sequence[current_index + 1]
                    roadmap_key = f"{cell_name}->{next_cell}"
                    
                    if roadmap_key in self.roadmap_paths:
                        roadmap_waypoints = self.roadmap_paths[roadmap_key]['path']
                        complete_path.extend(roadmap_waypoints)
        
        if not complete_path:
            self.get_logger().warn("Brak ścieżki do analizy pokrycia")
            return
        
        # Utwórz osobne okno
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(display_map, cmap='gray', origin='upper')
        
        # PARAMETRY SKANERA
        robot_diameter_px = int(self.ROBOT_WIDTH_METERS / self.map_resolution)
        scanner_diameter_px = robot_diameter_px * 2  # 2x robot width
        scanner_radius = scanner_diameter_px / 2
        
        # Wybierz punkty w regularnych odstępach
        circle_spacing = scanner_radius * 1.2
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
        
        # RYSUJ OKRĘGI SKANERA
        for x, y in selected_points:
            # Okrąg reprezentujący zasięg skanera - CIEMNONIEBIESKI z przezroczystością
            scanner_circle = plt.Circle((x, y), radius=scanner_radius, 
                                    color='darkblue', alpha=0.3, fill=True)
            ax.add_patch(scanner_circle)
            
            # Kontur okręgu - CIEMNA OBWÓDKA
            scanner_outline = plt.Circle((x, y), radius=scanner_radius, 
                                    color='navy', alpha=0.8, fill=False, linewidth=2)
            ax.add_patch(scanner_outline)
            
            # Punkt środkowy - CZERWONY żeby był widoczny
            ax.scatter(x, y, c='red', s=12, alpha=0.9, edgecolors='black', linewidths=1)
        
        # Pokaż trasę
        if complete_path:
            px, py = zip(*complete_path)
            ax.plot(px, py, 'yellow', linewidth=1.2, alpha=0.8, label='Enhanced Path')
        
        # OBLICZ RZECZYWISTE POKRYCIE
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
        
        # LEPSZA dylatacja dla wypełnienia dziur
        kernel = np.ones((3, 3), np.uint8)
        coverage_map = cv2.dilate(coverage_map, kernel, iterations=2)
        
        # OBLICZ POKRYCIE z lepszą metodą
        free_mask = (self.map_data == 0)  # Tylko wolne obszary
        coverage_mask = (coverage_map == 1)
        
        # Przecięcie: pokryte wolne obszary
        covered_free_mask = free_mask & coverage_mask
        
        free_cells = np.sum(free_mask)
        covered_free_cells = np.sum(covered_free_mask)
        coverage_percent = covered_free_cells / free_cells * 100 if free_cells > 0 else 0
        
        # DODATKOWO: sprawdź pokrycie w obszarze dostępnym dla robota
        robot_accessible_mask = self.safe_free_mask if hasattr(self, 'safe_free_mask') else free_mask
        accessible_cells = np.sum(robot_accessible_mask)
        covered_accessible_cells = np.sum(robot_accessible_mask & coverage_mask)
        accessible_coverage_percent = covered_accessible_cells / accessible_cells * 100 if accessible_cells > 0 else 0
        
        # DODAJ POZYCJĘ ROBOTA
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
        
        # TYTUŁ I OSIE z lepszymi informacjami
        ax.set_title(f'Enhanced B2 Coverage Analysis - Auto Spacing\n'
                    f'Coverage: {coverage_percent:.1f}% (all free) | {accessible_coverage_percent:.1f}% (accessible) | {len(selected_points)} scan points | {time_text}')
        ax.set_xlabel(f'Enhanced Features: 8-dir A*, Auto Spacing, Path Smoothing | Coverage: {coverage_percent:.1f}%')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        self.get_logger().info(f"📊 Enhanced Coverage Analysis: {coverage_percent:.1f}% total | {accessible_coverage_percent:.1f}% accessible")
        
        try:
            plt.show()
        except:
            self.get_logger().info("Nie można wyświetlić wizualizacji pokrycia (headless environment)")
        
        # ENHANCED STATYSTYKI POKRYCIA
        self.get_logger().info(f"🎯 ENHANCED COVERAGE DETAILS:")
        self.get_logger().info(f"  • Scanner diameter: {scanner_diameter_px} px ({scanner_diameter_px * self.map_resolution:.3f}m)")
        self.get_logger().info(f"  • Total scan points: {len(selected_points)}")
        self.get_logger().info(f"  • Auto spacing used: YES (Pure Pursuit optimized)")
        self.get_logger().info(f"  • 8-directional A*: YES")
        self.get_logger().info(f"  • Path smoothing: YES")
        self.get_logger().info(f"  • Free cells: {free_cells}")
        self.get_logger().info(f"  • Covered cells: {covered_free_cells}")
        self.get_logger().info(f"  • Total coverage: {coverage_percent:.1f}%")
        self.get_logger().info(f"  • Accessible coverage: {accessible_coverage_percent:.1f}%")
        
        return accessible_coverage_percent

    def draw_path_through_rectangles(self, ax, waypoints, color, cell_name):
        """Rysuj ścieżki TYLKO bezpieczne - pomiń niebezpieczne połączenia"""
        if len(waypoints) < 2:
            return
        
        safe_segments = []
        current_segment = [waypoints[0]]
        
        # Podziel ścieżkę na bezpieczne segmenty
        for i in range(1, len(waypoints)):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]
            
            # KLUCZOWE: Sprawdź czy linia jest bezpieczna
            if self.is_line_completely_safe(prev_wp, curr_wp):
                current_segment.append(curr_wp)
            else:
                # Linia niebezpieczna - zakończ segment
                if len(current_segment) > 1:
                    safe_segments.append(current_segment)
                current_segment = [curr_wp]  # Nowy segment
        
        # Dodaj ostatni segment
        if len(current_segment) > 1:
            safe_segments.append(current_segment)
        
        # Rysuj tylko bezpieczne segmenty
        for segment in safe_segments:
            if len(segment) > 1:
                x_coords = [wp[0] for wp in segment]
                y_coords = [wp[1] for wp in segment]
                ax.plot(x_coords, y_coords, color=color, linewidth=3, alpha=0.8, zorder=5)
        
        # Zaznacz wszystkie waypoints jako punkty
        for wp in waypoints:
            ax.plot(wp[0], wp[1], 'o', color=color, markersize=4, alpha=0.8)
        
        # Label tylko raz
        ax.plot([], [], color=color, linewidth=3, alpha=0.8, 
            label=f'Enhanced Boustrophedon {cell_name}')
    
    def print_b2_statistics(self):
        """Rozszerzone statystyki z enhanced features"""
        self.get_logger().info("=== ENHANCED B2 BOUSTROPHEDON ANALYSIS ===")
        
        if self.watershed_cells:
            total_waypoints = sum(cell['waypoint_count'] for cell in self.cell_zones.values())
            total_roadmap = sum(len(path['path']) for path in self.roadmap_paths.values())
            used_rects = sum(1 for rect in self.coverage_rects if rect.get('used_in_path', False))
            
            self.get_logger().info(f"📊 ENHANCED DECOMPOSITION:")
            self.get_logger().info(f"  • Total cells: {len(self.watershed_cells)}")
            self.get_logger().info(f"  • Cells with paths: {len(self.cell_zones)}")
            self.get_logger().info(f"  • Total rectangles: {len(self.coverage_rects)}")
            self.get_logger().info(f"  • Used in paths: {used_rects}")
            
            self.get_logger().info(f"🌀 ENHANCED BOUSTROPHEDON:")
            self.get_logger().info(f"  • Auto-spaced waypoints: {total_waypoints}")
            self.get_logger().info(f"  • Average per cell: {total_waypoints/len(self.cell_zones):.1f}")
            self.get_logger().info(f"  • Enhanced A* roadmap: {total_roadmap}")
            
            # RZECZYWISTY CZAS
            if self.total_real_time > 0:
                self.get_logger().info(f"⏱️ REAL TIME: {self.total_real_time:.1f}s ({self.total_real_time/60:.1f} min)")
            elif hasattr(self, 'real_start_time') and self.real_start_time:
                current_time = time.time() - self.real_start_time
                self.get_logger().info(f"⏱️ CURRENT TIME: {current_time:.1f}s (running...)")
            
            self.get_logger().info(f"🗺️ ENHANCED EXECUTION:")
            sequence_text = ' → '.join(self.cell_sequence) if len(self.cell_sequence) < 8 else f"{len(self.cell_sequence)} cells"
            self.get_logger().info(f"  • Sequence: {sequence_text}")
            self.get_logger().info(f"  • Features: 8-dir A*, auto spacing, path smoothing")
            
            # Informacje o czasach komórek
            if self.cell_completion_times:
                self.get_logger().info(f"📋 CELL TIMES:")
                for cell_name, completion_time in self.cell_completion_times.items():
                    if cell_name in self.cell_start_times:
                        cell_duration = completion_time - self.cell_start_times[cell_name]
                        self.get_logger().info(f"  • {cell_name}: {cell_duration:.1f}s")
            
            self.get_logger().info("✅ Enhanced B2 Analysis complete!")
            
            # *** DODAJ WYWOŁANIE METRYK TUTAJ ***
            self.get_logger().info("🎯 Obliczam końcowe metryki całej trasy...")
            self.calculate_final_path_metrics()

    # ============== NAV2 INTEGRATION - ROZŁĄCZONE ŚCIEŻKI ==============
    
    def odom_callback(self, msg):
        """Główny callback - sterowanie wykonaniem z rozłączonymi ścieżkami"""
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
        
        # Maszyna stanów - ROZŁĄCZONA
        if self.state == ExecutionState.WAITING_FOR_ODOM:
            self.state = ExecutionState.PLANNING
            self.execution_start_time = self.get_clock().now()
            # ROZPOCZNIJ RZECZYWISTY POMIAR CZASU
            self.real_start_time = time.time()
            self.get_logger().info("🚀 ENHANCED B2 STARTED - Real time measurement began!")
            
        elif self.state == ExecutionState.PLANNING:
            if self.current_cell_index < len(self.cell_sequence):
                self.execute_navigation_to_cell()
            else:
                # ZAKOŃCZENIE - zapisz całkowity czas
                if self.real_start_time:
                    self.total_real_time = time.time() - self.real_start_time
                    self.get_logger().info(f"🏁 ENHANCED B2 FINISHED - Total real time: {self.total_real_time:.1f}s")
                self.state = ExecutionState.FINISHED

    def execute_navigation_to_cell(self):
        """FAZA 1: Enhanced A* do komórki"""
        cell_name = self.cell_sequence[self.current_cell_index]
        cell_data = self.cell_zones[cell_name]
        
        if self.current_cell_index == 0:
            self.first_astar_start = time.time() 

        # ROZPOCZNIJ POMIAR CZASU KOMÓRKI
        if self.real_start_time:
            self.cell_start_times[cell_name] = time.time() - self.real_start_time
        
        self.state = ExecutionState.NAVIGATING_TO_CELL
        self.current_navigation_phase = "to_cell"
        
        # ZAWSZE zacznij od aktualnej pozycji robota
        current_robot_pos = self.robot_pixel_position
        cell_start = cell_data['start_point']
        
        # SPRAWDŹ CZY ROBOT JEST JUŻ BLISKO STARTU KOMÓRKI
        distance_to_start = math.sqrt((current_robot_pos[0] - cell_start[0])**2 + 
                                     (current_robot_pos[1] - cell_start[1])**2)
        
        # Jeśli robot jest bardzo blisko (mniej niż 10 pikseli), pomiń A* i idź do boustrophedon
        if distance_to_start < 10:
            self.get_logger().info(f"🎯 Robot już blisko startu {cell_name} - przechodzę do enhanced boustrophedon")
            self.execute_boustrophedon_in_cell()
            return
        
        # ENHANCED A* do startu komórki
        self.get_logger().info(f"🛤️ Enhanced A* navigation to {cell_name}")
        navigation_path = self.advanced_astar_navigation(current_robot_pos, cell_start)
        
        # Wyślij TYLKO Enhanced A* ścieżkę
        self.send_path_to_nav2(navigation_path, f"Enhanced A* Navigation to {cell_name}")

    def execute_boustrophedon_in_cell(self):
        """FAZA 2: Enhanced boustrophedon wewnątrz komórki"""
        cell_name = self.cell_sequence[self.current_cell_index]
        cell_data = self.cell_zones[cell_name]
        
        if self.current_cell_index == 0:
            self.coverage_with_inter_astar_start = time.time()

        self.state = ExecutionState.EXECUTING_BOUSTROPHEDON
        self.current_navigation_phase = "boustrophedon"
        
        # Pobierz ścieżkę boustrophedon (już ma enhanced gęste punkty z auto spacing)
        boustrophedon_path = cell_data['waypoints']
        
        # Wyślij TYLKO enhanced boustrophedon ścieżkę
        self.send_path_to_nav2(boustrophedon_path, f"Enhanced Boustrophedon in {cell_name}")

    def send_path_to_nav2(self, path_px, description):
        """Wyślij ścieżkę do Nav2 (enhanced wersja)"""
        if not path_px or len(path_px) == 0:
            self.get_logger().error(f"Pusta ścieżka dla: {description}")
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
        
        self.get_logger().info(f"🛤️ Wysyłam: {description} ({len(path_px)} enhanced punktów)")
        
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
        """Obsługa odpowiedzi Nav2 (enhanced wersja)"""
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
        """Obsługa wyniku wykonania Nav2 z enhanced rozłączonymi fazami"""
        try:
            result = future.result().result
            self.get_logger().info(f"✅ Zakończono: {description}")
            
            # LOGIKA ROZŁĄCZONYCH FAZY
            if self.state == ExecutionState.NAVIGATING_TO_CELL:
                # Zakończono Enhanced A* - przejdź do enhanced boustrophedon
                self.get_logger().info(f"🎯 Enhanced A* do komórki zakończony - rozpoczynam enhanced boustrophedon")
                self.execute_boustrophedon_in_cell()
                
            elif self.state == ExecutionState.EXECUTING_BOUSTROPHEDON:
                # Zakończono enhanced boustrophedon - zapisz czas i przejdź do następnej komórki
                cell_name = self.cell_sequence[self.current_cell_index]
                
                # ZAKOŃCZ POMIAR CZASU KOMÓRKI
                if self.real_start_time:
                    self.cell_completion_times[cell_name] = time.time() - self.real_start_time
                    if cell_name in self.cell_start_times:
                        duration = self.cell_completion_times[cell_name] - self.cell_start_times[cell_name]
                        self.get_logger().info(f"⏱️ Enhanced {cell_name} completed in {duration:.1f}s")
                
                # Przejdź do następnej komórki
                self.current_cell_index += 1

                if self.current_cell_index >= len(self.cell_sequence):  # ostatnia komórka zakończona
                    if hasattr(self, 'coverage_with_inter_astar_start'):
                        time_without_first_astar = time.time() - self.coverage_with_inter_astar_start
                        self.get_logger().info(f"🌀 TIME WITHOUT FIRST A*: {time_without_first_astar:.1f}s (includes inter-zone A*)")
                    if hasattr(self, 'first_astar_start') and hasattr(self, 'coverage_with_inter_astar_start'):
                        first_astar_duration = self.coverage_with_inter_astar_start - self.first_astar_start
                        self.get_logger().info(f"📍 FIRST A* DURATION: {first_astar_duration:.1f}s (excluded from test)")

                self.state = ExecutionState.PLANNING
                self.get_logger().info(f"🔄 Enhanced komórka {cell_name} zakończona - przechodzę do planowania następnej")
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd wyniku: {str(e)}")
            self.handle_navigation_failure()

    def handle_navigation_failure(self):
        """Obsługa błędów nawigacji (enhanced wersja)"""
        if self.state == ExecutionState.NAVIGATING_TO_CELL:
            self.get_logger().warn("⚠️ Błąd Enhanced A* - próbuję bezpośrednio enhanced boustrophedon")
            self.execute_boustrophedon_in_cell()
            
        elif self.state == ExecutionState.EXECUTING_BOUSTROPHEDON:
            self.get_logger().warn("⚠️ Błąd enhanced boustrophedon - przechodzę do następnej komórki")
            
            # Zapisz czas błędu
            if self.current_cell_index < len(self.cell_sequence):
                cell_name = self.cell_sequence[self.current_cell_index]
                if self.real_start_time:
                    self.cell_completion_times[cell_name] = time.time() - self.real_start_time
            
            self.current_cell_index += 1
            self.state = ExecutionState.PLANNING
        else:
            self.get_logger().warn("⚠️ Błąd enhanced nawigacji - kontynuuję")
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
                
                if angle > 25.0:
                    turn_count += 1
        
        # REDUNDANCJA - po prostu sprawdź duplikaty!
        unique_points = len(set(path_px))  # unikalne punkty
        total_points = len(path_px)        # wszystkie punkty
        duplicates = total_points - unique_points
        redundancy_percent = (duplicates / total_points) * 100
        
        # PRINT - ZMIENIONE NA WATER
        print(f"WATER DYSTANS: {total_distance:.2f}m | ZAKRĘTY: {turn_count} | REDUNDANCJA: {redundancy_percent:.1f}% ({duplicates}/{total_points})")

    def calculate_final_path_metrics(self):
        """
        Prosta wersja dla water.py - połącz wszystkie ścieżki i policz metryki
        """
        if not self.cell_sequence or not self.cell_zones:
            self.get_logger().warn("❌ Brak danych do metryk")
            return
        
        # Połącz CAŁĄ trasę: boustrophedon + Enhanced A* roadmaps
        complete_path = []
        
        for i, cell_name in enumerate(self.cell_sequence):
            if cell_name in self.cell_zones:
                # 1. Dodaj ścieżkę boustrophedon w komórce
                cell_waypoints = self.cell_zones[cell_name]['waypoints']
                complete_path.extend(cell_waypoints)
                
                # 2. Dodaj Enhanced A* roadmap do następnej komórki (jeśli istnieje)
                if i < len(self.cell_sequence) - 1:
                    next_cell = self.cell_sequence[i + 1]
                    roadmap_key = f"{cell_name}->{next_cell}"
                    
                    if roadmap_key in self.roadmap_paths:
                        try:
                            roadmap_waypoints = self.roadmap_paths[roadmap_key]['path']
                            if roadmap_waypoints:
                                complete_path.extend(roadmap_waypoints)
                        except Exception as e:
                            self.get_logger().warn(f"⚠️ Błąd Enhanced A* roadmap: {e}")
                            # Bez roadmap - kontynuuj
        
        if not complete_path:
            self.get_logger().warn("❌ Pusta ścieżka końcowa")
            return
        
        self.get_logger().info(f"🎯 Obliczam metryki dla {len(complete_path)} punktów (enhanced boustrophedon + Enhanced A* roadmaps)")
        
        # Wywołaj prostą funkcję metryk (DOKŁADNIE jak w t1.py)
        self.print_path_metrics(complete_path)

def main():
    rclpy.init()
    planner = SimpleBoustrophedonNav2Planner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Zatrzymywanie Enhanced B2 plannera...")
    finally:
        # Zapisz końcowy czas
        if hasattr(planner, 'real_start_time') and planner.real_start_time:
            planner.total_real_time = time.time() - planner.real_start_time
            planner.get_logger().info(f"🏁 FINAL ENHANCED B2 TIME: {planner.total_real_time:.1f}s")
        
        planner.destroy_node()

if __name__ == '__main__':
    main()