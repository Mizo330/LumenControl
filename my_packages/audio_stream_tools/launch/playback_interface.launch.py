from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.conditions import IfCondition


THIS_PACKAGE = "audio_stream_tools"

def generate_launch_description():
    gui_arg = DeclareLaunchArgument(
            "launch_gui",
            default_value="True",
            description="Launch GUI"
        )

    launch_gui = LaunchConfiguration("launch_gui")
    
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
            condition=IfCondition(launch_gui)
        ),

    ]

    return LaunchDescription([gui_arg]+nodes)