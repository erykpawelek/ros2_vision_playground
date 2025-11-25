from launch import LaunchDescription    
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_vision_package',
            executable='image_processor',
            name='image_processor',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'h_upper': 140},
                {'h_lower': 95},
                {'s_lower': 90},
                {'mode' : 'color_rec'}
            ]
        ),

        Node(
            package='camera_ros',
            executable='camera_node',
            name='camera',
            output='screen',
            emulate_tty=True,
            parameters=[]
        )
    ])