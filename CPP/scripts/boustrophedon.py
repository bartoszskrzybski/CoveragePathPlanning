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
import time
import matplotlib.pyplot as plt
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class CompleteCoveragePlanner(Node):
    def __init__(self):
        super().__init__('complete_coverage_planner')
            # Ustaw parametr use_sim_time na True bezpośrednio w kodzie
        param = rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        self.set_parameters([param])
        
        # Map name
        self.map_name = "map_test"

        # Dodaj zmienną do śledzenia czasu
        self.path_start_time = None
        # DODAJ TE LINIE:
        self.real_start_time = None
        self.total_real_time = 0.0

        self.get_logger().info("Automatycznie ustawiono use_sim_time=True")
        # Parametry
        self.ROBOT_WIDTH = 0.2
        self.SAFETY_MARGIN = 0.16
        self.EARLY_TURN_DISTANCE = 10
        self.SAFETY_CELLS = 0  # Dodatkowe komórki marginesu bezpieczeństwa
        self.MAX_ALLOWED_BLOCKED = 1000  # Tolerancja zablokowanych ścieżek
        
        # Pliki mapy
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.MAP_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.pgm")
        self.YAML_FILE = os.path.join(current_dir, f"../maps/{self.map_name}.yaml")
        
        # Inicjalizacja
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.current_path = None
        self.robot_position = None
        self.map_loaded = False
        self.visited = set()  # Zbiór odwiedzonych punktów
        self.step_size = None  # Zostanie obliczone później
        
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
        
        self.get_logger().info("Planner gotowy! Czekam na dane odometrii...")

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
            

            # === DODAJ: ZAPISZ ORYGINALNĄ MAPĘ ===
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

    def odom_callback(self, msg):
        """Główny callback - generuje ścieżkę na podstawie odometrii"""
        if not self.map_loaded:
            return
            
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

        if self.is_obstacle(start_x, start_y):
            #self.get_logger().warn("Robot na przeszkodzie!")
            return
            
        if not self.current_path:
            self.get_logger().info(f"Generuję ścieżkę od pozycji robota: ({start_x}, {start_y})")
            path_px = self.generate_coverage_path(start_x, start_y)
            if path_px:
                self.publish_path(path_px)
                self.send_to_nav2()
                self.visualize_path(path_px)

        # DODAJ TE LINIE:
        if not hasattr(self, 'real_start_time') or self.real_start_time is None:
            self.real_start_time = time.time()
            self.get_logger().info("🚀 BOUSTROPHEDON STARTED - Real time measurement began!")

    def generate_coverage_path(self, start_x, start_y):
        """Generuj ścieżkę pokrycia zigzag z ulepszoną funkcją bezpieczeństwa i obsługą wielu obszarów"""
        path = []
        self.visited = set()  # Reset zbioru odwiedzonych punktów
        
        h, w = self.map_data.shape
        
        # Zwiększamy bazowy rozmiar kroku dla szerszych odstępów
        base_step_size = max(3, int(self.ROBOT_WIDTH / self.map_resolution * 1.7))
        self.get_logger().info(f"Start = ({start_x}, {start_y}), bazowy krok = {base_step_size}")
        
        current_x, current_y = start_x, start_y
        self.add_point(path, current_x, current_y)
        
        # KRYTYCZNA ZMIANA: WYMUSZAMY POCZĄTKOWY RUCH W GÓRĘ (UJEMNE Y)
        direction_y = -1  # -1 = DO GÓRY (mniejsze Y = wyżej)
        
        # Główna pętla
        scan_complete = False
        while not scan_complete:
            # Dynamiczne dostosowanie rozmiaru kroku
            self.step_size = self.calculate_adaptive_step_size(current_x, current_y, base_step_size)
            self.step_size = max(self.step_size, base_step_size // 2)  # Minimalny krok
            
            # Skan w osi Y (góra/dół) - TUTAJ ZACZYNAMY, ZAMIAST OSOBNEGO POCZĄTKOWEGO RUCHU
            vertical_scan_complete = False
            vertical_steps = 0
            
            while not vertical_scan_complete:
                # Limit kroków pionowych
                if vertical_steps > h // self.step_size:
                    vertical_scan_complete = True
                    continue
                    
                # Oblicz krok
                self.step_size = self.calculate_adaptive_step_size(current_x, current_y, base_step_size)
                self.step_size = max(self.step_size, base_step_size // 2)
                
                # Sprawdź wczesny skręt - BEZ DODATKOWYCH WARUNKÓW
                if self.should_early_turn(current_x, current_y, direction_y):
                    vertical_scan_complete = True
                    continue
                
                next_y = current_y + direction_y * self.step_size
                # Sprawdź bezpieczeństwo ruchu pionowego
                if not (0 <= next_y < h) or not self.is_safe_path(current_x, current_y, current_x, next_y):
                    vertical_scan_complete = True
                    continue
                    
                # Dodajemy punkt i aktualizujemy pozycję
                self.add_point(path, current_x, next_y)
                current_y = next_y
                vertical_steps += 1
            
            # Ruch w prawo
            next_x = current_x + self.step_size
            if next_x >= w or not self.is_safe_path(current_x, current_y, next_x, current_y):
                scan_complete = True
                continue
                
            self.add_point(path, next_x, current_y)
            current_x = next_x
            
            # Zmień kierunek (góra <-> dół)
            direction_y *= -1
        
        # Próba znalezienia następnego punktu startowego, jeśli obecna ścieżka kończy się przedwcześnie
        expected_coverage = (w * h) / (base_step_size * base_step_size) * 0.5  # 50% oczekiwanego pokrycia
        if len(path) < expected_coverage:
            next_start = self.find_next_starting_point()
            if next_start:
                next_x, next_y = next_start
                # Generowanie ścieżki przejścia z obecnego punktu końcowego do nowego punktu startowego
                transition_path = self.generate_transition_path(current_x, current_y, next_x, next_y)
                if transition_path:
                    path.extend(transition_path)
                    # Rekurencyjne generowanie ścieżki z nowego punktu startowego
                    remaining_path = self.generate_coverage_path(next_x, next_y)
                    if remaining_path:
                        path.extend(remaining_path)
        
        self.get_logger().info(f"Wygenerowano ścieżkę ({len(path)} punktów)")

        if path:
            self.print_path_metrics(path)

        return path

    def calculate_adaptive_step_size(self, x, y, base_step):
        """Oblicz adaptacyjny rozmiar kroku w oparciu o dostępną przestrzeń"""
        h, w = self.map_data.shape
        
        # Sprawdź dostępną przestrzeń w każdym kierunku
        space_right = 0
        for i in range(1, w):
            if x + i >= w or self.is_obstacle(x + i, y):
                space_right = i - 1
                break
            if i > w/2:  # Ograniczamy sprawdzanie do połowy szerokości mapy
                space_right = i
                break
        
        space_up = 0
        for i in range(1, h):
            if y + i >= h or self.is_obstacle(x, y + i):
                space_up = i - 1
                break
            if i > h/2:  # Ograniczamy sprawdzanie do połowy wysokości mapy
                space_up = i
                break
                
        space_down = 0
        for i in range(1, h):
            if y - i < 0 or self.is_obstacle(x, y - i):
                space_down = i - 1
                break
            if i > h/2:  # Ograniczamy sprawdzanie do połowy wysokości mapy
                space_down = i
                break
        
        # Oblicz adaptacyjny rozmiar kroku (50-100% bazowego rozmiaru kroku)
        available_space = min(space_right, max(space_up, space_down))
        adaptive_step = min(base_step, max(int(base_step * 0.5), available_space))
        
        return max(1, adaptive_step)  # Zapewnienie co najmniej 1

    def find_next_starting_point(self):
        """Znajdź następny nieodwiedzony punkt startowy"""
        h, w = self.map_data.shape
        base_step = max(1, int(self.ROBOT_WIDTH / self.map_resolution * 0.8))
        
        # Utwórz siatkę potencjalnych punktów startowych w oparciu o rozmiar kroku
        for y in range(0, h, base_step):
            for x in range(0, w, base_step):
                if (x, y) not in self.visited and not self.is_obstacle(x, y):
                    # Sprawdź, czy wokół tego punktu jest wystarczająco dużo miejsca
                    if self.has_enough_space(x, y):
                        return (x, y)
        return None

    def has_enough_space(self, x, y):
        """Sprawdź, czy wokół punktu jest wystarczająco dużo miejsca, aby rozpocząć nową ścieżkę"""
        base_step = max(1, int(self.ROBOT_WIDTH / self.map_resolution * 0.8))
        min_area = 3 * base_step  # Minimalna powierzchnia potrzebna do rozpoczęcia nowej ścieżki
        
        free_cells = 0
        for dy in range(-min_area, min_area + 1):
            for dx in range(-min_area, min_area + 1):
                check_x, check_y = x + dx, y + dy
                if not self.is_obstacle(check_x, check_y):
                    free_cells += 1
                    if free_cells >= min_area * min_area * 0.3:  # 30% powierzchni powinno być wolne
                        return True
        return False

    def generate_transition_path(self, start_x, start_y, end_x, end_y):
        """Generuj prostą ścieżkę przejścia między dwoma punktami"""
        path = []
        
        # Używamy algorytmu Bresenhama do generowania linii prostej
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        sx = 1 if start_x < end_x else -1
        sy = 1 if start_y < end_y else -1
        err = dx - dy
        
        x, y = start_x, start_y
        while x != end_x or y != end_y:
            # Sprawdź bezpieczeństwo następnego kroku
            next_x = x + sx if abs(err) >= dy else x
            next_y = y + sy if abs(err) >= dy else y
            
            if self.is_obstacle(next_x, next_y) or not self.is_safe_path(x, y, next_x, next_y):
                return None  # Nie można znaleźć bezpiecznej ścieżki
                
            self.add_point(path, next_x, next_y)
            x, y = next_x, next_y
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return path

    def add_point(self, path, x, y):
        """Dodaje punkt do ścieżki (jeśli jeszcze go nie było)"""
        if (x, y) not in self.visited:
            path.append((x, y))
            self.visited.add((x, y))

    def should_early_turn(self, cx, cy, dir_y):
        """Bardziej tolerancyjna logika wczesnego skrętu z dozwolonym częściowym zablokowaniem"""
        blocked_paths = 0
        
        # Sprawdzenie ścieżki do przodu w krokach
        for i in range(1, self.EARLY_TURN_DISTANCE + 1):
            test_y = cy + i * dir_y * self.step_size
            if not (0 <= test_y < self.map_data.shape[0]) or not self.is_path_clear(cx, cy, cx, test_y):
                blocked_paths += 1
                if blocked_paths > self.MAX_ALLOWED_BLOCKED:
                    return True
            
            # Sprawdź ścieżki po przekątnej i z boku z tolerancją
            test_x = cx + self.step_size
            if test_x < self.map_data.shape[1]:
                diagonal_blocked = not self.is_path_clear(cx, cy, test_x, test_y)
                side_blocked = not self.is_path_clear(cx, test_y, test_x, test_y)
                
                if diagonal_blocked and side_blocked:
                    blocked_paths += 1
                    if blocked_paths > self.MAX_ALLOWED_BLOCKED:
                        return True
        
        # NOWOŚĆ: Sprawdzamy również więcej kroków w prawo
        # To pomoże uniknąć sytuacji gdy robot skręca, a potem wpada na przeszkodę
        for i in range(1, 2):
            test_x = cx + i * self.step_size
            if test_x >= self.map_data.shape[1] or self.is_obstacle(test_x, cy):
                blocked_paths += 1
                if blocked_paths > self.MAX_ALLOWED_BLOCKED:
                    return True
                
        return False

    def is_safe_path(self, x1, y1, x2, y2):
        """Ulepszone sprawdzanie bezpieczeństwa z adaptacyjnym marginesem w oparciu o kierunek ścieżki"""
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

    def publish_path(self, path_px):
        """Publikuj ścieżkę z właściwą orientacją punktów"""
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
            
            # Orientacja
            if i < len(path_px) - 1:
                next_x, next_y = path_px[i+1]
                dx = (next_x - x) * self.map_resolution
                dy = -(next_y - y) * self.map_resolution  # Uwzględniamy odwrócenie Y
                yaw = np.arctan2(dy, dx)
            else:
                yaw = 0.0
            pose.pose.orientation.z = np.sin(yaw / 2)
            pose.pose.orientation.w = np.cos(yaw / 2)
            
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.current_path = path_msg
        self.get_logger().info("Opublikowano ścieżkę na topic /plan")

    def send_to_nav2(self):
        """Wyślij ścieżkę do Nav2"""
        if not self.current_path:
            self.get_logger().error("Brak ścieżki do wysłania!")
            return
                
        self.get_logger().info("Wysyłanie ścieżki do Nav2...")
        
        goal_msg = FollowPath.Goal()
        goal_msg.path = self.current_path
        
        # Sprawdźmy, jakie atrybuty są dostępne w obiekcie goal_msg
        goal_fields = [f for f in dir(goal_msg) if not f.startswith('_')]
        self.get_logger().info(f"Dostępne pola celu: {goal_fields}")
        
        # Dynamicznie ustawiamy tylko dostępne atrybuty
        if hasattr(goal_msg, 'controller_id'):
            goal_msg.controller_id = "FollowPath"
        
        if hasattr(goal_msg, 'goal_checker_id'):
            goal_msg.goal_checker_id = "general_goal_checker"  # Używamy oryginalnej wartości
        
        # Te pola mogą nie być dostępne w Twojej wersji
        if hasattr(goal_msg, 'server_name'):
            goal_msg.server_name = "FollowPath"
        
        if hasattr(goal_msg, 'server_timeout'):
            goal_msg.server_timeout = 10
        
        self.nav2_client.wait_for_server()
        send_goal_future = self.nav2_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.nav2_goal_response)

    def nav2_goal_response(self, future):
        """Obsługa odpowiedzi Nav2"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 odrzucił ścieżkę!")
            return
            
        self.get_logger().info("Nav2 zaakceptował ścieżkę!")
        goal_handle.get_result_async().add_done_callback(self.nav2_result_callback)

    def nav2_result_callback(self, future):
        """Obsługa wyniku wykonania"""
        try:
            result = future.result().result
            
            
            if hasattr(self, 'real_start_time') and self.real_start_time is not None:
                self.total_real_time = time.time() - self.real_start_time
                self.get_logger().info(f"🏁 BOUSTROPHEDON FINISHED - Total real time: {self.total_real_time:.1f}s")
            
            # Sprawdź dostępne pola w wyniku
            if hasattr(result, 'error_msg'):
                error_msg = result.error_msg
                self.get_logger().info(f"Nav2 zakończył wykonanie, komunikat: {error_msg}")
            
            if hasattr(result, 'error_code_id'):
                error_code = result.error_code_id
                self.get_logger().info(f"Nav2 zakończył wykonanie z kodem: {error_code}")
            
            # Jeśli nie ma żadnego z tych pól, wyświetl ogólny komunikat
            if not (hasattr(result, 'error_msg') or hasattr(result, 'error_code_id')):
                self.get_logger().info("Nav2 zakończył wykonanie trasy")
                
        except Exception as e:
            self.get_logger().error(f"Błąd przy odbieraniu wyniku Nav2: {str(e)}")
            # Wyświetl szczegóły błędu
            import traceback
            self.get_logger().error(traceback.format_exc())

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
        """Wizualizacja ścieżki z okręgami reprezentującymi zasięg skanera"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Mapa i ścieżka
        display_map = np.copy(self.map_data)
        display_map[display_map == -1] = 50
        
        ax1.imshow(display_map, cmap='gray', vmin=0, vmax=100, origin='upper')
        
        if path_px:
            px, py = zip(*path_px)
            ax1.plot(px, py, 'r-', linewidth=1.5)
            ax1.scatter(px[0], py[0], c='green', s=100, marker='o', label='Start')
            ax1.scatter(px[-1], py[-1], c='red', s=100, marker='x', label='Koniec')
            
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
        
        ax1.set_title('Ścieżka Pokrycia Zigzag z Adaptacyjnym Krokiem')
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
    planner = CompleteCoveragePlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Zamykanie planner'a...")
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()