#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/qos.hpp"

#include <opencv2/opencv.hpp>

class ImageProcessor : public rclcpp::Node
{
private:
    
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
    // Parameters callback ptr
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
    
    rclcpp::QoS qos_policy_;
    // Parameters values
    int h_upper_, h_lower_, s_lower_;
    std::string mode_;

void listener_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg){
    // Converting image format to array style formato on which cv library operates
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(*msg, "bgr8");
    cv::Mat imported_image = cv_image->image, output_image;

    // Mode selection
    if (mode_ == "color_rec"){
        output_image = color_rec(imported_image);
    }else{
        output_image = imported_image;
    }
    
    // Packing msg for publisher
    cv_bridge::CvImage output_msg;
    output_msg.header = msg->header;
    output_msg.encoding = "bgr8";
    output_msg.image = output_image;
    
    publisher_->publish(*output_msg.toCompressedImageMsg());
}

public:
    ImageProcessor()
    :   Node("image_processor"),
        // Setting up QoS policy for camera stream
        qos_policy_(rclcpp::QoS(1).best_effort().durability_volatile())
    {   
        this->declare_parameter("h_upper", 140);
        this->declare_parameter("h_lower", 95);
        this->declare_parameter("s_lower", 90);
        this->declare_parameter("mode", "color_rec");
        
        h_upper_ = this->get_parameter("h_upper").as_int();
        h_lower_ = this->get_parameter("h_lower").as_int();
        s_lower_ = this->get_parameter("s_lower").as_int();
        mode_ = this->get_parameter("mode").as_string();
        // Registering the callback to handle dynamic parameter updates
        params_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&ImageProcessor::parametersCallback,
            this,
            std::placeholders::_1));
        
        publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/camera/image_processed/compressed",
            qos_policy_);
        
        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/image_raw/compressed",
            qos_policy_,
            std::bind(&ImageProcessor::listener_callback, this, std::placeholders::_1));
    }
    /**
     * @brief Performs color-based object detection and position estimation.
     * * Processes the input image to isolate objects of a specific color (defined by parameters),
     * calculates their centroids, and draws visual overlays (contours, center points).
     *
     * @param imported_image Input video frame in BGR format (cv::Mat).
     * @return Processed image with background masked out and debug visualizations drawn.
     */
    cv::Mat color_rec(cv::Mat & imported_image){
        
        cv::Size img_size  = imported_image.size();     // Extracting dimmensions for center moment calculation
        int width = img_size.width;
        int height = img_size.height;
        cv::Point screen_centre(width/2, height/2); 
        
        cv::Mat hsv_image;       // Converting format to hsv for more roboust color filtration selction
        cv::cvtColor(imported_image, hsv_image, cv::COLOR_BGR2HSV);
    
        // Defining color to filter through
        cv::Scalar lower = cv::Scalar(h_lower_, s_lower_, 50);
        cv::Scalar upper = cv::Scalar(h_upper_, 255, 255);
        cv::Mat mask,masked_image;
        cv::inRange(hsv_image, lower, upper, mask);
        cv::bitwise_and(imported_image, imported_image, masked_image, mask); // Bitwise and filter our image, leaving only defined above color

        std::vector<std::vector<cv::Point> > contours, contours_to_draw;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE); // Finding all countours in our filtred single chanel image 
        if (contours.size() > 0){                           
            auto max_countour_iterator = std::max_element(     // Finding biggest countour
            contours.begin(),
            contours.end(),
            [] (const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
            {return cv::contourArea(a) < cv::contourArea(b);});
            std::vector<cv::Point> biggest_contour = *max_countour_iterator;
            contours_to_draw.push_back(biggest_contour);
            
            cv::Point contour_centre;       // Finding mas centre of our countour
            cv::Moments moments = cv::moments(biggest_contour);
            if (moments.m00 > 0){
                contour_centre.x = moments.m10/moments.m00;
                contour_centre.y = moments.m01/moments.m00;
            } else {
                contour_centre.x = 0;
                contour_centre.y = 0;
            }
            cv::Point centre_error(screen_centre.x-contour_centre.x, screen_centre.y-contour_centre.y);
            RCLCPP_INFO(this->get_logger(), "X error: %d, Y error: %d", centre_error.x, centre_error.y);

            cv::circle(masked_image, contour_centre, 7, cv::Scalar(0,0,255), -1);   // Visualizing results
            cv::line(masked_image, screen_centre, contour_centre, cv::Scalar(255,255,255),2);
            cv::drawContours(masked_image, contours_to_draw, -1, cv::Scalar(0, 255, 0), 2); 
        } 
        return masked_image;
    }

    rcl_interfaces::msg::SetParametersResult parametersCallback(
        const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        int val;
        std::string val_str;
        for(const auto &param : parameters)
        {
            if (param.get_name() == "h_upper"){
                val = param.as_int();
                if (val >= 0 && val <= 179){
                    result.successful = true;
                    this->h_upper_ = val;
                    result.reason = "Success!";
                }else{
                    result.successful = false;
                    result.reason = "HSV hue value is in range 0-179!";
                }
            }else if (param.get_name() == "h_lower"){
                val = param.as_int();
                if (val >= 0 && val <= 179){
                    result.successful = true;
                    this->h_lower_ = val;
                    result.reason = "Success!";
                }else{
                    result.successful = false;
                    result.reason = "HSV hue value is in range 0-179!";
                }
            }else if (param.get_name() == "s_lower"){
                val = param.as_int();
                if (val >= 0 && val <= 255){
                    result.successful = true;
                    this->s_lower_ = val;
                    result.reason = "Success!";
                }else{
                    result.successful = false;
                    result.reason = "HSV saturation value is in range 0-255!";
                }
            }else if (param.get_name() == "mode"){
                val_str = param.as_string();
                if (val_str == "color_rec"){
                    result.successful = true;
                    result.reason = "Success!";
                    this->mode_ = val_str;
                }else{
                    result.successful = false;
                    result.reason = "Invalid mode:" + val_str;
                }
            }
        }
        return result;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProcessor>());
    rclcpp::shutdown();
    return 0;
}

