import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import psutil
import csv



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
            ('s_lower', 90),
            ('mode', 'color_rec'),
            ('benchmark_mode', False),
            ('benchmark_start', False),
            ('benchmark_duration', 60.0),])
           
        # --- RO2 objects declaration and image format adjustment ---
        self.bridge = CvBridge()

        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            qos_policy)
        
        self.publisher_ = self.create_publisher(CompressedImage, '/image_processed/compressed', qos_policy) 

        # --- MediaPipe neural network initialization ---
        package_share_dir = get_package_share_directory('my_vision_package')
        model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')

        base_options = python.BaseOptions(model_asset_path=model_path) # Packing model to be readable for mediapipe
        options = vision.HandLandmarkerOptions(                        # Defining handlandmarker options
            base_options = base_options,
            running_mode = vision.RunningMode.LIVE_STREAM,             # Async mode
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            result_callback = self.neural_net_mediapipe_callback)      # Asynchronus result
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.latest_results = None

       # --- Benchmarking setup ---
        self.benchmark_last_frame_time = time.time()
        self.benchmark_start_time = self.benchmark_last_frame_time
        self.benchmark_running = False
        # File initialization
        if self.get_parameter('benchmark_mode').value:
            self.csv_file = open('benchmark_results_async.csv', 'w', newline='') 
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'FPS', 'CPU_Usage_Percent', 'RAM_Usage_MB'])
            self.process = psutil.Process() 
            self.get_logger().warn("Benchmark mode initializated")
    
    def listener_callback(self, msg: CompressedImage):
        """
        This function is core function which manages all processing modes, with subscription to raw compressed image as well as for publishing results
        """
        mode =  self.get_parameter('mode').value
        # Frame skipping logic:
        # We process only every 3rd frame to prevent queue buildup (lag).
        # By doing this, we trade FPS for lower latency, ensuring that 
        # the displayed results are synchronized with real-time movement.
        if mode == 'neural_net_mediapipe':
            if not hasattr(self, 'frame_count'): self.frame_count = 0
            self.frame_count += 1
            if self.frame_count % 3 != 0: return 
           
        try:
           
            np_arr = np.frombuffer(msg.data, np.uint8)              # Converting incoming bytes into np.array object for cv2
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)       # Converting one dimmensional array into multidimensional
            height, width, channels = cv_image.shape                # Collecting data about image processed by cv_bridge
        
            # Switching between modes
            if mode == 'color_rec':
                processed_image = self.color_rec(cv_image,height, width)
            elif mode == 'neural_net_mediapipe':
                processed_image = self.neural_net_mediapipe(cv_image, height, width)
            else:
                self.get_logger().info('Invalid mode! \n Please choose between: \n color_rec\n neural_net_mediapipe')
                processed_image = cv_image
                    
            # Compressing precessed image along with preparing packages for publishing
            success, encoded_jpg = cv2.imencode('.jpg', processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if success:
                output_msg = CompressedImage()
                output_msg.header = msg.header
                output_msg.format = "jpeg"
                output_msg.data = encoded_jpg.tobytes()

                self.publisher_.publish(output_msg)
                
                self.benchmark()
            

        except Exception as e:
            self.get_logger().info(f'Processing ERROR: {e}')

    def color_rec(self, cv_image, height, width):
        """
        This function is responsible for processing image and object detection using color recognition and open cv2 library.
        """
        h_upper = self.get_parameter('h_upper').value           #Parameters mapping
        h_lower = self.get_parameter('h_lower').value
        s_lower = self.get_parameter('s_lower').value

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
            self.get_logger().info(f'X error: {centre_error_x},  Y error" {centre_error_y}')
            cv2.line(masked_image, center_screen, center_object, (255, 255, 255), 2)

        white_pixels = cv2.countNonZero(mask)                   # Counting pixels containing detected color
        self.get_logger().info(f'I can see {white_pixels} defined clor pixels.')    

        return masked_image
    
    def neural_net_mediapipe_callback(self, result, output_image, timestamp_ms):
        """
        This is callback for HandLandmarker class to return results, due to documentation specyfication it needs to by provided with 3 arguments.
        """
        self.latest_results = result

    def neural_net_mediapipe(self, cv_image, height, width):
        """
        This function is core for MediaPipe neural network model processing. It passes images with stamps to queue for processing, and draws points in place of hand landmarks. 
        """
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # Converting image ro RGB, which is necessary for medipipe
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_image) 
        timestamp_ms = int(time.time() * 1000)                # Creating timestamp to conect each proccesed image with uniqe stamp
        self.landmarker.detect_async(mp_image, timestamp_ms)
        # Poloting results
        if self.latest_results and self.latest_results.hand_landmarks:
            for hand_landmarks in self.latest_results.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(cv_image, (x, y), 5, (0, 255, 255), -1)
                # Using old API method due to poor new API documentation
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    start_point = hand_landmarks[connection[0]]
                    end_point = hand_landmarks[connection[1]]
                    x1, y1 = int(start_point.x * width), int(start_point.y * height)
                    x2, y2 = int(end_point.x * width), int(end_point.y * height)

                    cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                   
        # Returning image with landmarks drawed. Hint: Not every iteration will return image with drawed landmarsk due tu async processing 
        return cv_image

    def benchmark(self):
        """
        This function is responsible for benchmarking the process. It logs fps data in real time and with parameters start values it logs data to csv file.
        """
        stop_time = time.time()
        duration = stop_time - self.benchmark_last_frame_time
        self.benchmark_last_frame_time = stop_time
        if duration > 0:
            fps = 1.0 / duration
            self.get_logger().info(f'Current FPS: {fps:.1f}')
        
        if self.get_parameter('benchmark_start').value:
            if not self.benchmark_running:
                self.benchmark_start_time = time.time()
                self.benchmark_running = True
            cpu_usage = self.process.cpu_percent()
            ram_usage = self.process.memory_info().rss / 1024 / 1024
            elapsed = stop_time - self.benchmark_start_time

            self.csv_writer.writerow([f"{elapsed:.2f}", f"{fps:.2f}", f"{cpu_usage:.1f}", f"{ram_usage:.1f}"])

            if elapsed >= self.get_parameter('benchmark_duration').value:
                self.get_logger().warning("BENCHMARK COMPLETE. Saving and shutting down.")
                self.csv_file.close()
                rclpy.shutdown()

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