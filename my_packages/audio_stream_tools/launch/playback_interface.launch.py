from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

THIS_PACKAGE = "audio_stream_tools"

def generate_launch_description():

    nodes = [
        Node(
            package=THIS_PACKAGE,
            executable='playbacker',
            name='playbacker',
            output='screen',
        ),
        Node(
            package=THIS_PACKAGE,
            executable='output',
            name='output',
            output='screen',
        ),
        Node(
            package=THIS_PACKAGE,
            executable='listener_gui',
            name='listener_gui',
            output='screen',
        ),

    ]

    return LaunchDescription(nodes)