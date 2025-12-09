from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package= 'my_vision_cpp',
            name= 'image_processor',
            executable='img_processor_cpp',
            output= 'screen',
            emulate_tty= True,
            parameters= [
                {
                'h_upper': 140,
                'h_lower': 95,
                's_lower': 90,
                'mode' : 'color_rec',
                'benchmark_mode' : False,
                }
            ]
        ),
        Node(
            package='camera_ros',
            executable='camera_node',
            name='camera',
            output='screen',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', 'ERROR'],
            parameters=[
                {
                'camera_auto_detect' : True,
                'camera' : '/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a',
                'format' : 'XRGB8888',
                'width' : 1536,
                'height' : 864,
                'AfMode': 0,
                'LensPosition' : 0.0,
                }
            ]
        )
    ])