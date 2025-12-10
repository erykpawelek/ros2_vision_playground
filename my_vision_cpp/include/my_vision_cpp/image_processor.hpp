#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <memory>
#include <string>
#include <fstream>
#include <chrono>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
/**
 * @brief ROS node responsible for image processing.
 * * Class handles receiving camra image, processing in two modes:
 * 1. Color detection classical computer vision image processing.
 * 2. AI YOLOv8 nano neural network processing.
 * * Additionally node has benchmarking functionality (CPU, RAM, FPS)
 */
class ImageProcessor : public rclcpp::Node
{
public: 
    /**
     * @brief ImageProcessor node contructor.
     * * Initialization of ROS interfaces along with external tools.
     * @param node_name Name of ROS node.
     * @param model_path Full system path to model (.onnx) file.
     */
    ImageProcessor(const std::string & node_name, const std::string & model_path);

    /**
     * @brief Node destructor.
     * * Save benchmarking data into .csv file.
     */
    ~ImageProcessor();

private:
    // --- ROS Interfaces ---

    /**Compressed image subscription. */
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;

    /**Processed image publisher. */
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
    
    /**A callback handler that handles parameters change on the fly. */
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;

    /**Quality of service policy for video stream */
    rclcpp::QoS qos_policy_;

    // --- Parameters and state variables ---

    int h_upper_;///< Top limit for hue (hsv) value.
    int h_lower_;///< Bottom limit for hue (hsv) value.
    int s_lower_;///< Botom limit for saturation (hsv) value.
    std::string mode_;///< Program state parameter

    // --- Benchmarking --- 

    bool benchmark_start_;///< Benchmark start flag.
    std::chrono::steady_clock::time_point benchmark_last_frame_time_; 
    std::chrono::steady_clock::time_point benchmark_start_time_;
    std::chrono::seconds benchmark_duration_;
    bool benchmark_running_;
    std::ofstream csv_file_;
    // System log data collection asisting variables
    std::clock_t last_cpu_time_ ;
    std::chrono::steady_clock::time_point last_sys_time_;
    // System log data colection containers
    std::vector<double> log_timestamps_;
    std::vector<double> log_fps_;
    std::vector<double> log_cpu_;
    std::vector<double> log_ram_;

    // --- Dynamic file path ---

    std::string onnx_model_path_;

    // --- ONNX Runtime ---
    Ort::Env ort_env_;
    std::shared_ptr<Ort::Session> ort_session_;
    Ort::SessionOptions ort_session_options_;
    std::vector<std::string> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_;
    std::vector<std::string> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_;
    size_t input_tensor_size_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    /**
     * @brief Parameter on change on fly callback.
     * * @param parameters Vector of changed parameters.
     * @return Operation result.
     */
    rcl_interfaces::msg::SetParametersResult parameters_callback(
        const std::vector<rclcpp::Parameter> & parameters);
    
    /**
     * @brief Main function responsible for receiving camera data.
     *  Converts image to OpenCV, and decides about processing type (based on param `mode`), and publish results.
     * * @param msg Shared pointer to image message.
     */
    void listener_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

    /**
     * @brief Processes image using color detection algorithm (hsv).
     * * @param imported_image Orginal frame after cv convertion in BGR format.
     * @return Image with contours of the detected object.
     */
    cv::Mat color_rec(cv::Mat & imported_image);

    /**
     * @brief Initialization of metadata onnx sesion.
     * * Dynamiclly aquires model data from it's file.
     * It's being run only once in constructor.
     */
    void onnx_init();

    /**
     * @brief Perform neural network inference on single frame.
     * * Prepares blob, starts ONNX sesion and passes outpu for plotting.
     * * @param imported_image Orginal frame after cv convertion in RGB format.
     * @return Processed image with detected object boundries drawed.
     */
    cv::Mat neural_net_onnx(cv::Mat & imported_image);

    /**
     * @brief Processes raw output tensor values and draws boundries on object.
     * * Performs post-processing (NMS - Non Maximum Supression) along with vizualization.
     * * @param output_tensor Nural net processing result.
     * @param imported_image Image desired to be drawn on.
     * @return Image with vizualization.
     */
    cv::Mat onnx_draw_boundries(const std::vector<Ort::Value> & output_tensor, cv::Mat & imported_image);

    /**
    * @brief Calculates and logs performance statistics (FPS, CPU, RAM).
    * * If `benchmark_start_` is true, collects data for later recording.
    */
    void benchmark();

    /**
    * @brief Reads the current RAM usage of the process.
    * @return Memory usage in MB.
    */
    double get_ram_usage();

    /**
     @brief Calculates the percentage of CPU usage by a process.
    * @return CPU usage in %.
    */
    double get_cpu_usage();

    /**
    * @brief Saves the collected benchmark data to a CSV file.
    * * The file is saved in the node's working directory under the name "benchmark_results_cpp.csv".
    */
    void save_benchmark_data();
};
#endif //IMAGE_PROCESSOR_HPP