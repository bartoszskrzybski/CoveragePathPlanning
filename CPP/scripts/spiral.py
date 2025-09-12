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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from enum import Enum
import time

class SpiralExecutionState(Enum):
    WAITING_FOR_ODOM = 0
    PLANNING = 1
    NAVIGATING_TO_CENTER = 2  # NOWY STAN - A* do środka
    EXECUTING_SPIRAL = 3      # NOWY STAN - spirala
    FINISHED = 4

class SquareSpiralCoveragePlanner(Node):
    def __init__(self):
        super().__init__('square_spiral_coverage_planner')

        # Map name
        self.map_name = "map_test"
        
        # Ustaw parametr use_sim_time na True bezpośrednio w kodzie
        param = rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        self.set_parameters([param])
        
        # Stan wykonania - JAK W WATER.PY
        self.state = SpiralExecutionState.WAITING_FOR_ODOM
        self.robot_position = None
        self.robot_pixel_position = None
        self.map_loaded = False
        
        # DODANE: Rzeczywiste śledzenie czasu (jak w water.py)
        self.real_start_time = None
        self.navigation_start_time = None
        self.spiral_start_time = None
        self.total_real_time = 0.0
        
        # Dodaj zmienną do śledzenia czasu
        self.path_start_time = None
        self.path_start_timestamp = None

        self.get_logger().info("Automatycznie ustawiono use_sim_time=True")
        # Parametry
        self.ROBOT_WIDTH = 0.2
        self.SAFETY_MARGIN = 0.14
        self.SAFETY_CELLS = 0  # Dodatkowe komórki marginesu bezpieczeństwa
        self.MAX_ALLOWED_BLOCKED = 1  # Tolerancja zablokowanych ścieżek
        self.SPIRAL_PATH_MULTIPLIER = 1.7
        
        # Pliki mapy
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.MAP_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.pgm")
        self.YAML_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.yaml")
        
        # Inicjalizacja
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.current_path = None
        self.visited = set()  # Zbiór odwiedzonych punktów
        self.step_size = None  # Zostanie obliczone później
        
        # NOWE: Rozłączone ścieżki (jak w water.py)
        self.path_to_center = None
        self.spiral_path = None
        self.current_navigation_phase = "none"  # "to_center" lub "spiral"
        self.execution_start_time = None
        
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
        
        # Subskrypcje (TYLKO odometria)
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # Wczytaj mapę
        self.load_map()
        
        self.get_logger().info("Planner spirali kwadratowej gotowy! Czekam na dane odometrii...")

    def load_map(self):
        """Wczytaj i opublikuj mapę"""
        try:
            self.get_logger().info(f"Ładowanie mapy z: {self.MAP_FILE}")
            
            if not os.path.exists(self.MAP_FILE):
                raise FileNotFoundError(f"Brak pliku mapy: {self.MAP_FILE}")
            if not os.path.exists(self.YAML_FILE):
                raise FileNotFoundError(f"Brak pliku YAML: {self.YAML_FILE}")
            
            with open(self.YAML_FILE) as f:
                map_meta = yaml.safe_load(f)
                self.map_resolution = map_meta['resolution']
                self.map_origin = map_meta['origin'][:2]
                self.get_logger().info(f"Rozdzielczość: {self.map_resolution}, Origin: {self.map_origin}")
            
            img = cv2.imread(self.MAP_FILE, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Nie można wczytać mapy: {self.MAP_FILE}")
            
            # DODAJ TE 4 LINIE:
            self.original_map_data = np.zeros_like(img, dtype=np.int8)
            self.original_map_data[img == 254] = 0    # wolne
            self.original_map_data[img == 0] = 100    # przeszkody  
            self.original_map_data[img == 205] = -1   # nieznane
            
            self.map_data = np.zeros_like(img, dtype=np.int8)
            self.map_data[img == 254] = 0    # wolne
            self.map_data[img == 0] = 100    # przeszkody
            self.map_data[img == 205] = -1   # nieznane
            
            # Kopia oryginalnej mapy
            original_map = np.copy(self.map_data)
            h, w = self.map_data.shape
            
            # Tworzymy nową mapę
            new_map = np.zeros_like(self.map_data)
            
            # Margines bezpieczeństwa w pikselach
            safety_margin = int((self.ROBOT_WIDTH/2 + self.SAFETY_MARGIN) / self.map_resolution) + 2
            
            # Znajdź wszystkie przeszkody
            obstacles = np.where(original_map == 100)
            obstacle_points = list(zip(obstacles[0], obstacles[1]))
            
            # Parametry do identyfikacji przeszkód
            min_points_for_obstacle = 5  # Minimalna liczba punktów dla przeszkody
            grid_size = 10  # Rozmiar siatki do grupowania przeszkód
            
            # Mapa odwiedzonych punktów
            visited = np.zeros((h, w), dtype=bool)
            
            # Identyfikuj i grupuj przeszkody
            obstacle_groups = []
            
            # Dla każdego piksela przeszkody
            for y, x in obstacle_points:
                if visited[y, x]:
                    continue
                    
                # Wydziel grid, w którym znajduje się piksel
                grid_y = y // grid_size
                grid_x = x // grid_size
                
                # Znajdź wszystkie piksele przeszkód w tym samym gridzie
                grid_obstacle_points = []
                
                for cy in range(grid_y * grid_size, min((grid_y + 1) * grid_size, h)):
                    for cx in range(grid_x * grid_size, min((grid_x + 1) * grid_size, w)):
                        if original_map[cy, cx] == 100 and not visited[cy, cx]:
                            grid_obstacle_points.append((cy, cx))
                            visited[cy, cx] = True
                
                # Jeśli znaleziono wystarczającą liczbę punktów, dodaj jako grupę
                if len(grid_obstacle_points) >= min_points_for_obstacle:
                    obstacle_groups.append(grid_obstacle_points)
            
            # Teraz dla każdej grupy przeszkód tworzymy prostokąt
            for group in obstacle_groups:
                # Znajdź granice grupy
                y_coords, x_coords = zip(*group)
                min_y, max_y = min(y_coords), max(y_coords)
                min_x, max_x = min(x_coords), max(x_coords)
                
                # Sprawdź czy to duża i rozległa przeszkoda (potencjalna ściana)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                
                is_wall = False
                
                # Jeśli przeszkoda dotyka granicy mapy lub jest bardzo długa i wąska, traktuj ją jako ścianę
                if (min_x <= 5 or max_x >= w - 5 or min_y <= 5 or max_y >= h - 5 or
                    (width > 3 * height and width > 30) or (height > 3 * width and height > 30)):
                    is_wall = True
                
                # Ustaw różne marginesy w zależności od typu przeszkody
                if is_wall:
                    # Dla ścian - mniejszy margines
                    margin = max(1, safety_margin // 1)
                else:
                    # Dla zwykłych przeszkód - pełny margines
                    margin = safety_margin
                
                # Dodaj margines bezpieczeństwa
                safe_min_x = max(0, min_x - margin)
                safe_min_y = max(0, min_y - margin)
                safe_max_x = min(w - 1, max_x + margin)
                safe_max_y = min(h - 1, max_y + margin)
                
                # Dodaj prostokąt do nowej mapy
                # Dla ścian, dodaj tylko obramowanie zamiast wypełniać całość
                if is_wall:
                    # Górny i dolny brzeg
                    new_map[safe_min_y:safe_min_y+margin, safe_min_x:safe_max_x+1] = 100
                    new_map[safe_max_y-margin+1:safe_max_y+1, safe_min_x:safe_max_x+1] = 100
                    
                    # Lewy i prawy brzeg
                    new_map[safe_min_y:safe_max_y+1, safe_min_x:safe_min_x+margin] = 100
                    new_map[safe_min_y:safe_max_y+1, safe_max_x-margin+1:safe_max_x+1] = 100
                else:
                    # Dla zwykłych przeszkód - wypełnij cały prostokąt
                    new_map[safe_min_y:safe_max_y+1, safe_min_x:safe_max_x+1] = 100
            
            # Przywróć oryginalne przeszkody, żeby upewnić się, że wszystkie są uwzględnione
            new_map[original_map == 100] = 100
            
            # Przywróć nieznane obszary
            new_map[original_map == -1] = -1
            
            # Zastosuj nową mapę
            self.map_data = new_map
            
            self.publish_costmap()
            self.map_loaded = True
            
        except Exception as e:
            self.get_logger().error(f"Błąd ładowania mapy: {str(e)}")
            raise

    def publish_costmap(self):
        """Publikuje mapę do globalnego i lokalnego costmapa"""
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
        self.get_logger().info("Mapa opublikowana do costmap")

    # ============== MASZYNA STANÓW (JAK W WATER.PY) ==============
    
    def odom_callback(self, msg):
        """Główny callback - MASZYNA STANÓW jak w water.py"""
        if not self.map_loaded:
            return
            
        # Aktualizuj pozycję robota
        self.robot_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        
        # Konwersja współrzędnych z odwróceniem Y
        start_x = int((self.robot_position[0] - self.map_origin[0]) / self.map_resolution)
        start_y = self.map_data.shape[0] - 1 - int((self.robot_position[1] - self.map_origin[1]) / self.map_resolution)
        
        # Upewnij się, że pozycja startowa jest w granicach mapy
        h, w = self.map_data.shape
        start_x = np.clip(start_x, 0, w - 1)
        start_y = np.clip(start_y, 0, h - 1)
        self.robot_pixel_position = (start_x, start_y)

        if self.is_obstacle(start_x, start_y):
            #self.get_logger().warn("Robot na przeszkodzie!")
            return
        
        # MASZYNA STANÓW - JAK W WATER.PY
        if self.state == SpiralExecutionState.WAITING_FOR_ODOM:
            self.state = SpiralExecutionState.PLANNING
            self.execution_start_time = self.get_clock().now()
            # ROZPOCZNIJ RZECZYWISTY POMIAR CZASU
            self.real_start_time = time.time()
            self.get_logger().info("🚀 SPIRAL PLANNER STARTED - Real time measurement began!")
            
        elif self.state == SpiralExecutionState.PLANNING:
            # WYGENERUJ OBIE ŚCIEŻKI (jak w water.py)
            success = self.generate_spiral_paths(start_x, start_y)
            if success:
                self.execute_navigation_to_center()
            else:
                self.get_logger().error("❌ Nie udało się wygenerować ścieżek spirali!")
                self.state = SpiralExecutionState.FINISHED

    def generate_spiral_paths(self, start_x, start_y):
        """Wygeneruj obie ścieżki (jak generate_paths w water.py)"""
        h, w = self.map_data.shape
        
        # 1. Znajdź środek mapy (cel początkowy)
        center_x = w // 2
        center_y = h // 2
        
        # Dostosuj środek, aby uniknąć przeszkód
        center_adjusted = self.find_free_space_near_point(center_x, center_y, max_distance=30)
        if center_adjusted:
            center_x, center_y = center_adjusted
        
        self.get_logger().info(f"📍 Środek mapy: ({center_x}, {center_y})")
        
        # 2. NAJPIERW generuj spiralę od środka
        raw_spiral = self.generate_spiral_from_center(center_x, center_y)
        self.spiral_path = self.fix_spiral_endpoint_crossing(raw_spiral)
        
        if not self.spiral_path or len(self.spiral_path) == 0:
            return False
        
        # 3. TERAZ ścieżka A* do PIERWSZEGO PUNKTU SPIRALI
        spiral_start = self.spiral_path[0]  # <-- PIERWSZY PUNKT SPIRALI!
        self.path_to_center = self.find_path_to_center(start_x, start_y, spiral_start[0], spiral_start[1])
        
        if not self.path_to_center:
            return False
        
        self.get_logger().info(f"✅ A* do początku spirali ({len(self.path_to_center)} pkt), spirala ({len(self.spiral_path)} pkt)")
        self.get_logger().info(f"🎯 A* cel: {spiral_start}, Spirala start: {self.spiral_path[0]}")
        
        # Pokaż wizualizację pełnej ścieżki
        full_path = self.path_to_center + self.spiral_path
        self.print_path_metrics(self.spiral_path)
        self.visualize_path(full_path)
        
        return True

    def execute_navigation_to_center(self):
        """FAZA 1: A* do środka (jak w water.py)"""
        if not self.path_to_center:
            self.get_logger().error("❌ Brak ścieżki do środka!")
            self.handle_navigation_failure()
            return
        
        self.first_astar_start = time.time()
        
        self.state = SpiralExecutionState.NAVIGATING_TO_CENTER
        self.current_navigation_phase = "to_center"
        
        # ROZPOCZNIJ POMIAR CZASU NAWIGACJI
        if self.real_start_time:
            self.navigation_start_time = time.time() - self.real_start_time
        
        self.get_logger().info(f"🎯 A* nawigacja do środka ({len(self.path_to_center)} punktów)")
        
        # Wyślij TYLKO ścieżkę do środka
        self.send_path_to_nav2(self.path_to_center, "A* Navigation to Center")

    def execute_spiral_navigation(self):
        """FAZA 2: Spirala (jak w water.py)"""
        if not self.spiral_path:
            self.get_logger().error("❌ Brak ścieżki spirali!")
            self.handle_navigation_failure()
            return
        
        self.coverage_start = time.time()
        
        self.state = SpiralExecutionState.EXECUTING_SPIRAL
        self.current_navigation_phase = "spiral"
        
        # ROZPOCZNIJ POMIAR CZASU SPIRALI
        if self.real_start_time:
            self.spiral_start_time = time.time() - self.real_start_time
        
        self.get_logger().info(f"🌀 Wykonywanie spirali ({len(self.spiral_path)} punktów)")
        
        # Wyślij TYLKO ścieżkę spirali
        self.send_path_to_nav2(self.spiral_path, "Square Spiral Coverage")

    def send_path_to_nav2(self, path_px, description):
        """Wyślij ścieżkę do Nav2 (jak w water.py)"""
        if not path_px or len(path_px) == 0:
            self.get_logger().error(f"Pusta ścieżka dla: {description}")
            self.handle_navigation_failure()
            return
            
        path_msg = self.create_ros_path(path_px)
        
        self.path_pub.publish(path_msg)
        self.current_path = path_msg
        
        goal_msg = FollowPath.Goal()
        goal_msg.path = path_msg
        
        # Dynamicznie ustawiamy tylko dostępne atrybuty
        if hasattr(goal_msg, 'controller_id'):
            goal_msg.controller_id = "FollowPath"
        
        if hasattr(goal_msg, 'goal_checker_id'):
            goal_msg.goal_checker_id = "general_goal_checker"
        
        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 serwer niedostępny!")
            self.handle_navigation_failure()
            return
        
        self.get_logger().info(f"🛤️ Wysyłam: {description} ({len(path_px)} punktów)")
        
        # Rozpocznij pomiar czasu wykonania
        self.path_start_timestamp = self.get_clock().now().nanoseconds / 1e9
        
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
            
            # Pozycja z odwróceniem Y
            pose.pose.position.x = self.map_origin[0] + x * self.map_resolution
            pose.pose.position.y = self.map_origin[1] + (self.map_data.shape[0] - 1 - y) * self.map_resolution
            pose.pose.position.z = 0.0
            
            # Orientacja
            if i < len(path_px) - 1:
                next_x, next_y = path_px[i+1]
                dx = (next_x - x) * self.map_resolution
                dy = -(next_y - y) * self.map_resolution  # Uwzględniamy odwrócenie Y
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
        """Obsługa odpowiedzi Nav2 (jak w water.py)"""
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
        """Obsługa wyniku wykonania Nav2 z ROZŁĄCZONYMI FAZAMI (jak w water.py)"""
        try:
            result = future.result().result
            
            # Pomiar czasu wykonania
            if self.path_start_timestamp is not None:
                end_timestamp = self.get_clock().now().nanoseconds / 1e9
                execution_time = end_timestamp - self.path_start_timestamp
                self.get_logger().info(f"⏱️ Czas wykonania {description}: {execution_time:.2f}s")
                self.path_start_timestamp = None
            
            self.get_logger().info(f"✅ Zakończono: {description}")
            
            # LOGIKA ROZŁĄCZONYCH FAZY (JAK W WATER.PY)
            if self.state == SpiralExecutionState.NAVIGATING_TO_CENTER:
                # Zakończono A* do środka - przejdź do spirali
                self.get_logger().info(f"🎯 A* do środka zakończony - rozpoczynam spiralę")
                self.execute_spiral_navigation()
                
            elif self.state == SpiralExecutionState.EXECUTING_SPIRAL:
                # Zakończono spiralę - KONIEC
                # ZAKOŃCZ POMIAR CZASU
                if self.real_start_time:
                    self.total_real_time = time.time() - self.real_start_time
                    self.get_logger().info(f"🏁 SPIRAL FINISHED - Total real time: {self.total_real_time:.1f}s")

                if hasattr(self, 'coverage_start'):
                    spiral_only_time = time.time() - self.coverage_start
                    self.get_logger().info(f"🌀 SPIRAL ONLY: {spiral_only_time:.1f}s (exclude A* to center)")
                if hasattr(self, 'first_astar_start') and hasattr(self, 'coverage_start'):
                    astar_to_center_duration = self.coverage_start - self.first_astar_start
                    self.get_logger().info(f"📍 A* TO CENTER: {astar_to_center_duration:.1f}s (excluded from test)")
                
                self.state = SpiralExecutionState.FINISHED
                self.get_logger().info(f"🔄 Spirala zakończona - KONIEC!")
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd wyniku: {str(e)}")
            self.handle_navigation_failure()

    def handle_navigation_failure(self):
        """Obsługa błędów nawigacji (jak w water.py)"""
        if self.state == SpiralExecutionState.NAVIGATING_TO_CENTER:
            self.get_logger().warn("⚠️ Błąd A* do środka - próbuję bezpośrednio spiralę")
            self.execute_spiral_navigation()
            
        elif self.state == SpiralExecutionState.EXECUTING_SPIRAL:
            self.get_logger().warn("⚠️ Błąd spirali - kończę wykonanie")
            
            # Zapisz czas błędu
            if self.real_start_time:
                self.total_real_time = time.time() - self.real_start_time
                self.get_logger().info(f"🏁 SPIRAL FINISHED (with error) - Total real time: {self.total_real_time:.1f}s")
            
            self.state = SpiralExecutionState.FINISHED
        else:
            self.get_logger().warn("⚠️ Błąd nawigacji - kończę wykonanie")
            self.state = SpiralExecutionState.FINISHED

    # ============== ORYGINALNE METODY SPIRALI ==============

    def fix_spiral_endpoint_crossing(self, spiral_path):
        """PROSTE rozwiązanie - po prostu obetnij końcówkę"""
        if len(spiral_path) < 25:
            return spiral_path
        
        # Obetnij 10-15 punktów z końca - na pewno nie będzie przecięć
        cut_points = min(15, len(spiral_path) // 8)  # 10 punktów lub 1/8 spirali
        final_path = spiral_path[:-cut_points]
        
        self.get_logger().info(f"Obcięto {cut_points} punktów z końca: {len(spiral_path)} -> {len(final_path)}")
        return final_path

    def find_free_space_near_point(self, x, y, max_distance=30):
        """Znajduje wolną przestrzeń blisko danego punktu"""
        h, w = self.map_data.shape
        
        # Jeśli punkt jest już wolny, użyj go
        if not self.is_obstacle(x, y):
            return (x, y)
        
        # Przeszukaj obszar wokół punktu w coraz większych pierścieniach
        for distance in range(1, max_distance):
            # Przeszukaj kwadratowy pierścień o danym dystansie
            for dx in range(-distance, distance + 1):
                for dy in [-distance, distance]:  # Góra i dół
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not self.is_obstacle(nx, ny):
                        return (nx, ny)
                    
            for dy in range(-distance + 1, distance):
                for dx in [-distance, distance]:  # Lewo i prawo
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not self.is_obstacle(nx, ny):
                        return (nx, ny)
        
        # Jeśli nie znaleziono wolnego punktu, zwróć None
        return None

    def find_path_to_center(self, start_x, start_y, center_x, center_y):
        """Znajduje ścieżkę od punktu startowego do środka mapy (algorytm A*)"""
        h, w = self.map_data.shape
        
        # Inicjalizacja grid'a dla A*
        closed_set = set()
        open_set = {(start_x, start_y)}
        came_from = {}
        
        # Koszt przejścia od startu do danego punktu
        g_score = {(start_x, start_y): 0}
        
        # Szacowany całkowity koszt od startu do celu przez dany punkt
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, center_x, center_y)}
        
        while open_set:
            # Znajdź punkt z najmniejszym f_score
            current = min(open_set, key=lambda point: f_score.get(point, float('inf')))
            
            # Osiągnięto cel (lub jesteśmy blisko)
            if current == (center_x, center_y) or self.heuristic(*current, center_x, center_y) < 5:
                path = self.reconstruct_path(came_from, current)
                self.get_logger().info(f"Znaleziono ścieżkę do centrum ({len(path)} punktów)")
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Sprawdź sąsiadów (4 kierunki)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                
                if not (0 <= nx < w and 0 <= ny < h) or neighbor in closed_set:
                    continue
                
                # Sprawdź czy sąsiad jest przeszkodą
                if self.is_obstacle(nx, ny):
                    continue
                
                # Sprawdź bezpieczeństwo ścieżki
                if not self.is_safe_path(current[0], current[1], nx, ny):
                    continue
                
                # Oblicz tymczasowy g_score
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                # Ten ścieżka do sąsiada jest najlepsza do tej pory
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(nx, ny, center_x, center_y)
        
        # Jeśli nie znaleziono ścieżki, zwróć pustą listę
        self.get_logger().warn("Nie znaleziono ścieżki do centrum!")
        return []

    def heuristic(self, x1, y1, x2, y2):
        """Odległość Manhattan jako heurystyka dla A*"""
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, came_from, current):
        """Rekonstruuje ścieżkę na podstawie słownika came_from"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        
        # Obróć ścieżkę (od startu do celu)
        total_path.reverse()
        return total_path

    def generate_spiral_from_center(self, center_x, center_y):
        """Generuje ścieżkę spirali kwadratowej od środka mapy z uwzględnieniem przeszkód"""
        path = [(center_x, center_y)]
        self.visited = set([(center_x, center_y)])
        
        h, w = self.map_data.shape
        
        # Określ rozmiar kroku równy szerokości robota w pikselach
        self.step_size = int(self.ROBOT_WIDTH / self.map_resolution * self.SPIRAL_PATH_MULTIPLIER)
        
        # Początkowy kierunek i długość kroku dla spirali
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # prawo, dół, lewo, góra
        current_direction = 0
        step_count = 1
        
        current_x, current_y = center_x, center_y
        turns_completed = 0
        
        # Główna pętla spirali
        max_iterations = w * h // 4  # Ograniczenie liczby iteracji
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Po każdych dwóch zmianach kierunku zwiększamy liczbę kroków
            if turns_completed == 2:
                step_count += 1
                turns_completed = 0
            
            # Wykonaj kroki w bieżącym kierunku
            for i in range(step_count):
                dx, dy = directions[current_direction]
                next_x, next_y = current_x + dx * self.step_size, current_y + dy * self.step_size
                
                # Sprawdź czy punkt jest w granicach mapy i jest bezpieczny
                if not (0 <= next_x < w and 0 <= next_y < h) or self.is_obstacle(next_x, next_y) or not self.is_safe_path(current_x, current_y, next_x, next_y):
                    # Próba znalezienia alternatywnej ścieżki
                    alternative_found = False
                    
                    # Sprawdź wszystkie kierunki
                    for alt_dir in range(4):
                        alt_dx, alt_dy = directions[alt_dir]
                        alt_x, alt_y = current_x + alt_dx * self.step_size, current_y + alt_dy * self.step_size
                        
                        if (0 <= alt_x < w and 0 <= alt_y < h and 
                            not self.is_obstacle(alt_x, alt_y) and 
                            self.is_safe_path(current_x, current_y, alt_x, alt_y) and
                            (alt_x, alt_y) not in self.visited):
                            
                            next_x, next_y = alt_x, alt_y
                            current_direction = alt_dir
                            alternative_found = True
                            break
                    
                    if not alternative_found:
                        # Żadna alternatywa nie działa, kończymy spiralę
                        self.get_logger().info("Koniec spirali - brak bezpiecznych punktów")
                        return path
                        
                # Dodaj punkt do ścieżki
                path.append((next_x, next_y))
                self.visited.add((next_x, next_y))
                current_x, current_y = next_x, next_y
            
            # Zmień kierunek (w prawo w spirali)
            current_direction = (current_direction + 1) % 4
            turns_completed += 1
        
        self.get_logger().info(f"Wygenerowano spiralę kwadratową ({len(path)} punktów)")
        return path

    def add_point(self, path, x, y):
        """Dodaje punkt do ścieżki (jeśli jeszcze go nie było)"""
        if (x, y) not in self.visited:
            path.append((x, y))
            self.visited.add((x, y))

    def is_safe_path(self, x1, y1, x2, y2):
        """Ulepszone sprawdzanie bezpieczeństwa z adaptacyjnym marginesem"""
        if not self.is_path_clear(x1, y1, x2, y2):
            return False
            
        # Pomiń kosztowne sprawdzenia, jeśli nie potrzeba marginesu bezpieczeństwa
        if self.SAFETY_CELLS <= 0:
            return True
            
        # Oblicz kierunek ścieżki, aby zastosować większy margines w kierunku prostopadłym
        dx = x2 - x1
        dy = y2 - y1
        is_horizontal = abs(dx) > abs(dy)
        
        # Dostosuj komórki bezpieczeństwa w oparciu o kierunek ścieżki
        h_cells = self.SAFETY_CELLS if is_horizontal else self.SAFETY_CELLS + 1
        v_cells = self.SAFETY_CELLS + 1 if is_horizontal else self.SAFETY_CELLS
        
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            
            # Zastosuj adaptacyjny margines bezpieczeństwa
            for dy in range(-v_cells, v_cells + 1):
                for dx in range(-h_cells, h_cells + 1):
                    check_x, check_y = x + dx, y + dy
                    if self.is_obstacle(check_x, check_y):
                        return False
        
        return True

    def is_path_clear(self, x1, y1, x2, y2):
        """Sprawdza czy ścieżka jest wolna od przeszkód"""
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            if self.is_obstacle(x, y):
                return False
        return True

    def is_obstacle(self, x, y):
        """Sprawdza czy pozycja jest przeszkodą (próg 50)"""
        h, w = self.map_data.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return True
        return self.map_data[y, x] > 50

    def print_path_metrics(self, path_px):
        """
        Oblicz i wydrukuj podstawowe metryki ścieżki
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
        
        # PRINT
        print(f"DYSTANS: {total_distance:.2f}m | ZAKRĘTY: {turn_count} | REDUNDANCJA: {redundancy_percent:.1f}% ({duplicates}/{total_points})")

    def visualize_path(self, path_px):
        """Wizualizacja ścieżki spirali kwadratowej z okręgami reprezentującymi zasięg skanera"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Mapa i ścieżka
        display_map = np.copy(self.map_data)
        display_map[display_map == -1] = 50
        
        ax1.imshow(display_map, cmap='gray', vmin=0, vmax=100, origin='upper')
        
        if path_px:
            # Podziel ścieżkę na dwie części: do środka i spiralę
            # Znajdź punkt środkowy (ostatni punkt w path_to_center)
            h, w = self.map_data.shape
            center_x, center_y = w // 2, h // 2
            
            # Znajdź punkt najbliższy środkowi (przybliżony punkt rozpoczęcia spirali)
            center_dist = [(i, np.sqrt((x-center_x)**2 + (y-center_y)**2)) 
                        for i, (x, y) in enumerate(path_px)]
            center_idx = min(center_dist, key=lambda x: x[1])[0]
            
            # Rozdziel ścieżkę
            path_to_center = path_px[:center_idx+1]
            spiral_path = path_px[center_idx:]
            
            # Rysuj ścieżkę do środka
            if path_to_center:
                px_to_center, py_to_center = zip(*path_to_center)
                ax1.plot(px_to_center, py_to_center, 'g-', linewidth=2, label='Droga do środka')
                ax1.scatter(px_to_center[0], py_to_center[0], c='green', s=100, marker='o', label='Start')
            
            # Rysuj spiralę
            if spiral_path:
                px_spiral, py_spiral = zip(*spiral_path)
                ax1.plot(px_spiral, py_spiral, 'r-', linewidth=1.5, label='Spirala kwadratowa')
                ax1.scatter(px_spiral[-1], py_spiral[-1], c='red', s=100, marker='x', label='Koniec')
            
            # Rysuj wszystkie punkty dla przejrzystości
            px_all, py_all = zip(*path_px)
            
            # Strzałki kierunku
            arrow_step = max(1, len(path_px) // 20)
            for i in range(0, len(path_px)-1, arrow_step):
                x, y = path_px[i]
                if i+1 < len(path_px):
                    dx = path_px[i+1][0] - x
                    dy = path_px[i+1][1] - y
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx = dx / length * 5
                        dy = dy / length * 5
                        ax1.arrow(x, y, dx, dy, head_width=3, color='blue', alpha=0.6)
        
        # Pozycja robota
        if self.robot_position:
            robot_x = int((self.robot_position[0] - self.map_origin[0]) / self.map_resolution)
            robot_y = self.map_data.shape[0] - 1 - int((self.robot_position[1] - self.map_origin[1]) / self.map_resolution)
            ax1.scatter(robot_x, robot_y, c='magenta', s=150, marker='*', label='Robot', edgecolors='black')
        
        ax1.set_title('Ścieżka Pokrycia Spiralą Kwadratową')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mapa pokrycia - okręgi reprezentujące zasięg skanera
        ax2.imshow(display_map, cmap='gray', vmin=0, vmax=100, origin='upper')
        
        # Średnica robota w pikselach
        robot_diameter_px = int(self.ROBOT_WIDTH / self.map_resolution)
        
        # Średnica zasięgu skanera (2x średnica robota)
        scanner_diameter_px = robot_diameter_px * 2
        scanner_radius = scanner_diameter_px / 2
        
        # Wybierz punkty na ścieżce w regularnych odstępach - DOPASOWANE DO STYKU OKRĘGÓW
        if path_px:
            # Odległość między środkami okręgów = 1.8 * promień
            # To daje częściowe nakładanie się, ale nie za dużo
            circle_spacing = scanner_radius * 1.2
            
            path_distance = 0
            selected_points = [path_px[0]]  # Zawsze dodaj pierwszy punkt
            
            for i in range(1, len(path_px)):
                prev_x, prev_y = path_px[i-1]
                curr_x, curr_y = path_px[i]
                
                # Oblicz odległość między obecnym a poprzednim punktem
                segment_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                path_distance += segment_distance
                
                # Jeśli przeszliśmy wystarczającą odległość, wybierz punkt
                if path_distance >= circle_spacing:
                    selected_points.append(path_px[i])
                    path_distance = 0  # Resetuj licznik odległości
            
            # Dodaj ostatni punkt, jeśli nie został dodany
            if selected_points[-1] != path_px[-1]:
                selected_points.append(path_px[-1])
            
            # Rysuj okręgi dla wybranych punktów
            for x, y in selected_points:
                # Okrąg reprezentujący zasięg skanera
                scanner_circle = plt.Circle((x, y), radius=scanner_radius, color='white', alpha=0.2, fill=True)
                ax2.add_patch(scanner_circle)
                
                # Kontur okręgu
                scanner_outline = plt.Circle((x, y), radius=scanner_radius, color='white', alpha=0.4, fill=False, linewidth=0.5)
                ax2.add_patch(scanner_outline)
                
                # Punkt środkowy
                ax2.scatter(x, y, c='white', s=8, alpha=0.9)
        
        # Pokaż trasę na mapie pokrycia
        if path_px:
            px, py = zip(*path_px)
            ax2.plot(px, py, 'w-', linewidth=0.8, alpha=0.4)
        
        ax2.set_title('Mapa pokrycia - zasięg skanera')
        ax2.grid(True, alpha=0.3)
        
        # Przeliczamy rzeczywiste pokrycie na podstawie okręgów skanera
        coverage_map = np.zeros_like(self.map_data, dtype=np.uint8)
        h, w = self.map_data.shape
        
        # Używamy tych samych punktów co przy wizualizacji
        for x, y in selected_points:
            # Rysujemy okrąg na mapie pokrycia
            scanner_r_int = int(scanner_radius)
            for dy in range(-scanner_r_int - 1, scanner_r_int + 2):
                for dx in range(-scanner_r_int - 1, scanner_r_int + 2):
                    nx, ny = x + dx, y + dy
                    # Sprawdź czy punkt jest wewnątrz okręgu
                    if dx*dx + dy*dy <= scanner_radius**2:
                        if 0 <= nx < w and 0 <= ny < h and not self.is_obstacle(nx, ny):
                            coverage_map[ny, nx] = 1
        
        # Oblicz rzeczywiste pokrycie
        free_cells = np.sum(self.map_data == 0)
        covered_free_cells = np.sum((coverage_map == 1) & (self.map_data == 0))
        coverage_percent = covered_free_cells / free_cells * 100
        
        # Dodatkowa dylatacja dla wypełnienia małych dziur
        kernel = np.ones((2, 2), np.uint8)  # Mniejsze jądro dla delikatniejszej dylatacji
        dilated_coverage = cv2.dilate(coverage_map, kernel, iterations=1)
        coverage_map = dilated_coverage
        
        # Ponowne obliczenie pokrycia po dylatacji
        covered_free_cells = np.sum((coverage_map == 1) & (self.map_data == 0))
        coverage_percent = covered_free_cells / free_cells * 100
        
        # TOTAL COVERAGE
        if hasattr(self, 'original_map_data'):
            total_free_mask = (self.original_map_data == 0)
            total_free_cells = np.sum(total_free_mask)
            covered_total_cells = np.sum(total_free_mask & (coverage_map == 1))
            total_coverage_percent = covered_total_cells / total_free_cells * 100 if total_free_cells > 0 else 0
            
            ax2.set_xlabel(f'Total: {total_coverage_percent:.1f}% | Accessible: {coverage_percent:.1f}%')
        else:
            ax2.set_xlabel(f'Pokrycie: {coverage_percent:.1f}%')
        
        plt.tight_layout()
        plt.show()

def main():
    rclpy.init()
    planner = SquareSpiralCoveragePlanner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Zatrzymywanie Spiral plannera...")
    finally:
        # Zapisz końcowy czas
        if hasattr(planner, 'real_start_time') and planner.real_start_time:
            planner.total_real_time = time.time() - planner.real_start_time
            planner.get_logger().info(f"🏁 FINAL SPIRAL TIME: {planner.total_real_time:.1f}s")
        
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()