import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage,Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_prcessor')

        # Bridge initialization 
        self.bridge = CvBridge()

        # Subscription initialization
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            10)
        # Publisher initialization
        self.publisher_ = self.create_publisher(Image, '/image_processed', 10)
    
    def listener_callback(self, msg):
        try:
            
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            height, width, channels = cv_image.shape

            center_coordinates = (width // 2, height // 2)
            radius = 50
            color = (0, 255, 0)
            thickness = 5
            cv2.circle(cv_image, center_coordinates, radius, color, thickness)

            self.get_logger().info(f'Processed a frame: {width}x{height}')

            output_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

            self.publisher_.publish(output_msg)
        
        except Exception as e:
            self.get_logger().info(f'Processing ERROR: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()