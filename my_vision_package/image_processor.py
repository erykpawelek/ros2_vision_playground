import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(CompressedImage, '/image_processed/compressed', 10) 
    
    def listener_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)              # Converting incoming bytes into np.array object for cv2
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)       # Converting one dimmensional array into multidimensional
            height, width, channels = cv_image.shape                # Collecting data about image processed by cv_bridge
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)   # Converting to HSV because it separates color form brightness, making detection more robust to lighting changes.

            lower_blue = np.array([95, 80, 50])                     # Lower limit for blue color range
            upper_blue = np.array([140, 255, 255])                  # Upper limit for blue color range

            mask = cv2.inRange(hsv_image, lower_blue, upper_blue)   # Generating black-white mask, where blue = white  
            masked_image = cv2.bitwise_and(cv_image, cv_image, mask= mask)

            white_pixels = cv2.countNonZero(mask)
            self.get_logger().info(f'I can see {white_pixels} blue pixels.')

            success, encoded_jpg = cv2.imencode('.jpg', masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            if success:
                output_msg = CompressedImage()
                output_msg.header = msg.header
                output_msg.format = "jpeg"
                output_msg.data = encoded_jpg.tobytes()

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