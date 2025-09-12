import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    # Argumenty launch
    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='map4.world',
        description='Name of the world file'
    )
    
    # Ścieżki
    pkg_share = get_package_share_directory('turtlebot3_gazebo')
    world_path = PathJoinSubstitution([
        get_package_share_directory('my_gazebo_maps'),
        'worlds',
        LaunchConfiguration('world_name')
    ])

    model_path = os.path.join(
        pkg_share,
        'models',
        'turtlebot3_burger',
        'model.sdf'
    )

    # Proces Gazebo
    gazebo_process = ExecuteProcess(
        cmd=['gazebo', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             '-s', 'libgazebo_ros_laser.so',  # Wymagane dla LiDARa
             world_path],
        output='screen'
    )

    # Proces spawnowania
    spawn_robot = ExecuteProcess(
        cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
             '-entity', 'turtlebot3_burger',
             '-x', '0.0', '-y', '0.0', '-z', '0.2',
             '-Y', '0.0',
             '-file', model_path],
        output='screen'
    )

    # Węzeł robot_state_publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(os.path.join(
                pkg_share,
                'urdf',
                'turtlebot3_burger.urdf'
            )).read()
        }]
    )

    # Static transform dla Cartographera
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint'],
        output='screen'
    )

    return LaunchDescription([
        world_name_arg,
        LogInfo(msg=['World: ', world_path]),
        
        gazebo_process,
        robot_state_publisher,
        static_tf,
        
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=gazebo_process,
                on_start=[
                    LogInfo(msg='Waiting 5 seconds for Gazebo...'),
                    ExecuteProcess(
                        cmd=['sleep', '5'],
                        output='screen'
                    ),
                    spawn_robot
                ]
            )
        )
    ])
