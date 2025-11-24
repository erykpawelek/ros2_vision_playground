import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    """
    ROS2 node for real time image procesing 

    It subscribes real time camera image and processes color detection and filtration
    in HSV color space definition. It publishes result image which is combination od orginal and bit wise mask
    """
    def __init__(self):
        super().__init__('image_processor')

        # --- System parameters def ---
        self.declare_parameters(
            namespace='', 
            parameters=[
            ('h_upper', 140),
            ('h_lower', 95),
            ('s_lower', 90)])
           
        # --- RO2 objects declaration and image format adjustment ---
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            qos_profile_sensor_data)
        self.publisher_ = self.create_publisher(CompressedImage, '/image_processed/compressed', 10) 
    

    def listener_callback(self, msg: CompressedImage):
        """
        'listener_callback' is core function which is conected to raw image subscription. It is responsible for all image processing 
        as well as for publishing results
        """
        try:
            h_upper = self.get_parameter('h_upper').value           #Parameters mapping
            h_lower = self.get_parameter('h_lower').value
            s_lower = self.get_parameter('s_lower').value

            np_arr = np.frombuffer(msg.data, np.uint8)              # Converting incoming bytes into np.array object for cv2
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)       # Converting one dimmensional array into multidimensional
            height, width, channels = cv_image.shape                # Collecting data about image processed by cv_bridge
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)   # Converting to HSV because it separates color form brightness, making detection more robust to lighting changes.
            center_screen = [width//2, height//2]

            lower_ = np.array([h_lower, s_lower, 50])               # Lower limit for defined color 
            upper_ = np.array([h_upper, 255, 255])                  # Upper limit for defined color
            mask = cv2.inRange(hsv_image, lower_, upper_)           # Generating black-white mask, where defined color = white and other = black
            masked_image = cv2.bitwise_and(cv_image, cv_image, mask= mask)      # Folding raw image with mask

            # Contours recognition along with object mass center solving
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                biggest_contour = max(contours, key=cv2.contourArea)# We obtain largest contour to filter noises or samaller objects
                moments = cv2.moments(biggest_contour)              # Calculating moments of our contour
                if moments['m00'] > 0:
                    cx = int(moments['m10']/moments['m00'])         # From static moments we calculate centre of mass
                    cy = int(moments['m01']/moments['m00'])
                else: 
                    cx, cy = 0, 0                                   # Skip the frame

                center_object = [cx, cy]
                cv2.circle(masked_image, (cx,cy), 7, (0, 0, 255), -1)  
                cv2.drawContours(masked_image, [biggest_contour], -1, (0, 255, 0), 2)

                centre_error_x = center_screen[0] - cx              # Calculating error between screen center and obj center
                centre_error_y = center_screen[1] - cy
                self.get_logger().info(f'\nX error: {centre_error_x},  Y error" {centre_error_y}')
                cv2.line(masked_image, center_screen, center_object, (255, 255, 255), 2)

            white_pixels = cv2.countNonZero(mask)                               # Counting pixels containing detected color
            self.get_logger().info(f'I can see {white_pixels} defined clor pixels.')    

            # Compressing precessed image along with preparing packages for publishing
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