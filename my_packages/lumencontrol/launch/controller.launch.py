from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource

THIS_PACKAGE = "lumencontrol"

def generate_launch_description():

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare(THIS_PACKAGE), 'config', 'params.yaml'
        ]),
        description='Path to config YAML file'
    )
    config_file = LaunchConfiguration('config_file')

    playback_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("audio_stream_tools"),
                    "launch",
                    "playback_interface.launch.py"
                ])
            ),
            launch_arguments={
            'launch_gui' : 'false'
            }.items(),
        )
    
    nodes = [
        Node(
            package=THIS_PACKAGE,
            executable='filtering',
            name='filtering',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package=THIS_PACKAGE,
            executable='features',
            name='features',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package=THIS_PACKAGE,
            executable='beat_detector',
            name='beat_detector',
            output='screen',
        ),
        Node(
            package=THIS_PACKAGE,
            executable='gui',
            name='gui',
            output='screen',
        ),
    ]

    return LaunchDescription([config_file_arg,playback_launch]+nodes)