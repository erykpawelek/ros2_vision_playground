#include <string>
#include "my_vision_cpp/image_processor.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto onnx_model_path = 
    ament_index_cpp::get_package_share_directory("my_vision_cpp") +
    "/models/industrial_signs_yolo_nano_rev21.onnx";

    auto image_processor = std::make_shared<ImageProcessor>(
        "image_processor", onnx_model_path);

    rclcpp::spin(image_processor);
    rclcpp::shutdown();
    return 0;
}