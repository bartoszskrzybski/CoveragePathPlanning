# Coverage Path Planning Algorithms - Master's Thesis

Comprehensive implementation and comparative analysis of coverage path planning algorithms for autonomous robots with ROS2 integration and TurtleBot3 simulation. Features a novel Hybrid Algorithm (Watershed + Grid) for flat environment coverage.

# Algorithms Implemented:

### Boustrophedon (ZigZag)

    Classic back-and-forth pattern coverage for efficient coverage of convex areas.
<img width="1661" height="638" alt="image" src="https://github.com/user-attachments/assets/acbd38d9-bc07-4fda-af1f-24d036a9f7d3" />

### Spiral Pattern

    Outward spiral coverage optimized for circular environments.
<img width="1661" height="638" alt="image" src="https://github.com/user-attachments/assets/281baf20-d4b9-43a1-9b87-1f310e0d79f0" />


### Boustrophedon Decomposition

    Area decomposition followed by systematic zigzag coverage in each segment.
    
<img width="1075" height="860" alt="Screenshot from 2025-06-13 18-51-55" src="https://github.com/user-attachments/assets/c6103f0f-6407-48d0-849c-1787f25579d8" />


### Hybrid (Watershed + Grid) Algorithm (Novel Approach)

    Methodology combining watershed decomposition with rectangular grid normalization and advanced safety mechanisms for optimal coverage in complex environments.
    
<img width="1065" height="802" alt="Screenshot from 2025-06-13 20-16-50" src="https://github.com/user-attachments/assets/e0a5523e-7fe1-4603-ae0c-7096a37318e2" />

### Note: 

    Boustrophedon/Spiral: Red line + Blue circles  
    Decomposition Algorithms: Yellow line + Blue circles  
    
    - Lines: Robot's coverage path
    - Circles: Scanner coverage range at each position
    - Coverage: Calculated from scanner footprint overlap

# Hybrid Watershed + Grid Algorithm Details

### Core Concept : The algorithm combines **watershed decomposition** for environment-adaptive segmentation with **rectangular grid normalization** for generating safe, efficient coverage paths.

### Algorithm Architecture:

Phase 1: Watershed Decomposition & Zone Creation
    
    Input: Safe space map, peak_distance, min_cell_area
    Output: Watershed-based cells for coverage planning
    
    1. Calculate distance transform:
       - Apply Euclidean distance transform to safe space
       - Assign obstacle distance values to each pixel
    
    2. Find watershed peaks:
       - Detect local maxima using PEAK_MIN_DISTANCE separation
       - Peaks become centers of future cells
    
    3. Apply watershed segmentation:
       - Use negative distance transform for watershed algorithm
       - Segment space using peaks as seeds
    
    4. Create valid cells:
       - Filter cells by MIN_CELL_AREA threshold
       - Create cell objects from labeled regions

Phase 2: Rectangular Grid Generation with Ultra-Safe Validation
python
    
    Input: Rectangle grid within watershed cells
    Output: Ultra-safe zigzag waypoints with full validation
    
    1. Create safe zigzag pattern with validation:
       - FOR each row of rectangles:
            Determine direction: even rows LEFT_TO_RIGHT, odd rows RIGHT_TO_LEFT
            FOR each rectangle in continuous segments:
                IF rectangle center is ultra-safe AND line is completely safe:
                    Add rectangle center to waypoints
       - Return validated zigzag sequence
    
    2. Ultra-safe point validation:
       - Check all 8 surrounding pixels (3×3 neighborhood)
       - IF any surrounding pixel is unsafe: return FALSE
       - ELSE: return TRUE (ultra-safe)

Phase 3: Path Planning & Advanced Optimization
python

    Input: Watershed cells with internal coverage paths
    Output: Optimized cell sequence, inter-cell transition paths
    
    1. Plan optimal cell sequence using TSP nearest neighbor:
       - Start from first cell
       - WHILE not all cells visited:
            Find nearest unvisited cell by distance to current.end_point
            Add to sequence and update current cell
       - Return optimized cell visiting sequence
    
    2. Create enhanced A* roadmap between cells:
       - FOR each consecutive cell pair:
            Generate A* path from current.end_point to next.start_point
            Apply path densification for smooth transitions
            Store transition path in roadmap
       - Return complete inter-cell roadmap

# 🛡️ Safety Features

    Ultra-Safe Validation: 3×3 neighborhood checking around each point

    Complete Line Safety: Verification of entire path segments

    Adaptive Grid Sizing: Rectangle dimensions adjusted to robot width

    Obstacle-Aware Segmentation: Watershed decomposition based on obstacle proximity

# ⚙️ Parameter Optimization

    PEAK_MIN_DISTANCE: Controls minimum distance between cell centers

        Large maps: 45 pixels

        Small maps: 20 pixels
        
    Note: This parameter requires manual adjustment in the code based on map size

    MIN_CELL_AREA: Filters out excessively small segments

    Safety Margins: Configurable based on robot constraints

# 📊 Performance Metrics Analysis

    Coverage Time - Total mission execution time

    Path Length - Total distance traveled by robot

    Complete Coverage - Coverage percentage without safety margins

    Safe Coverage - Coverage with safety margins considering robot constraints

    Redundancy - Analysis of revisited areas and overlapping coverage

    Turn Count - Number of direction changes

# ⚙️ System Modifications

### Enhanced Nav2 Configuration

    Modified Regulated Pure Pursuit Controller in global Nav2 parameters:

    /opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml

# Environment Mapping Procedure

### Step 1: SLAM with Cartographer

    # Terminal 1: Launch Gazebo simulation
    
    ros2 launch my_gazebo_maps gazebo_launch.py world_name:=<map_name>.world

    # Terminal 2: Start Cartographer SLAM
    
    ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True

    # Terminal 3: Manual robot control
    
    ros2 run turtlebot3_teleop teleop_keyboard

### Step 2: Map Preservation

    # Save generated environment representation
    ros2 run nav2_map_server map_saver_cli -f <map_name>

# 🛠️ Installation & Setup

### Clone repository
    git clone https://github.com/bartoszskrzybski/CoveragePathPlanning.git

### Build only the gazebo maps package
    # Move package to your ROS2 workspace
    cp -r CoveragePathPlanning/my_gazebo_maps ~/ros2_ws/src/
    
    # Build the package
    cd ~/ros2_ws
    colcon build --packages-select my_gazebo_maps
    source install/setup.bash

### Important: Manual configuration required
    
    sudo cp CoveragePathPlanning/nav2_params.yaml /opt/ros/humble/share/nav2_bringup/params/

    Note: Backup original Nav2 parameters first!
    
### All operations from CPP folder

    The CPP/ folder contains all necessary files (scripts, maps) and must be used as the working directory.
    
### Configure environment (~/.bashrc)

    # Basic environment setup
    echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
    echo "source /usr/share/gazebo/setup.sh" >> ~/.bashrc
    
    # Additional critical configurations (adjust paths according to your setup):
    # export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    # export GAZEBO_MODEL_PATH=your_path_to_gazebo_models:${GAZEBO_MODEL_PATH}
    # export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:${GAZEBO_PLUGIN_PATH}
    # source /opt/ros/humble/setup.bash
    # source ~/your_ros2_ws/install/local_setup.bash

    Note: Additional Gazebo and ROS2 environment variables may be required depending on your system configuration. These typically include:    

    Gazebo model paths in specific order

    ROS2 workspace setup scripts

    Gazebo plugin paths
    
# Coverage Execution Workflow
### Three-Terminal Launch System

    Terminal 1 - Simulation Environment
    
    ros2 launch my_gazebo_maps gazebo_launch.py world_name:=<map_name>.world

    Terminal 2 - Navigation Stack

    cd ~/CPP
    ros2 launch turtlebot3_navigation2 navigation2.launch.py \
    use_sim_time:=true \
    map:=maps\<map_name>.yaml \
    params_file:=/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml

    Terminal 3 - Algorithm Execution

    cd ~/CPP
    python3 water.py

Note:  You must adjust the map name in the algorithm file:

    self.map_name = "map_test"  # Change "map_test" to your actual map name
    
https://github.com/user-attachments/assets/50d0df96-536a-41ab-b160-ce5c4777f352

*Video shows the complete 3-terminal system launch and A* navigation to first watershed zone.*

# 📁 Project Structure

    CoveragePathPlanning/
    ├── my_gazebo_maps/                
    │   ├── package.xml                # Package definition
    │   ├── setup.cfg                 
    │   ├── setup.py
    │   ├── launch/                    # Launch files          
    │   │   └── gazebo_launch.py
    │   ├── worlds/                    # Gazebo world files
    │   │   └── map4.world
    │   ├── resource/                  # Custom Gazebo models
    │   ├── test/ 
    │   └── my_gazebo_maps/ 
    ├── CPP/                           # Algorithm implementations folder
    │   ├── scripts/
    │   │   ├── boustrophedon.py       # Zigzag coverage
    │   │   ├── spiral.py              # Spiral pattern
    │   │   ├── choset.py              # Boustrophedon decoposition
    │   │   └── water.py               # Watershed decompostion + grid 
    │   └── maps/                      # Generated environment maps
    │       ├── map4.pgm
    │       └── map4.yaml
    └──nav2_params.yaml               # Modified Nav2 parameters
                     

# 🎓 Academic Contribution

This research presents a novel Hybrid Watershed + Grid Algorithm featuring:

    Intelligent watershed decomposition for environment-adaptive segmentation

    Rectangular grid normalization for safe coverage pattern generation

    Ultra-safe validation mechanisms with 3×3 neighborhood checking

    TSP optimization for optimal cell sequencing

    A roadmap generation for smooth inter-cell transitions

# 📊 Results Package

    Statistical analysis of coverage efficiency across all algorithms

    Visual coverage maps and path visualizations

    Performance comparison metrics

# 📄 License
    MIT License - Academic and research use permitted with attribution.

# Master's Thesis: 
    "Comparative Analysis of Coverage Path Planning Algorithms for Autonomous Mobile Robots with Novel Hybrid Watershed + Grid Approach"

# Author: 
    Bartosz Skrzybski
