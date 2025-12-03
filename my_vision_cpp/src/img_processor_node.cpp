#include <memory>

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

    rclcpp::QoS qos_policy_;

void listener_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg){
    // Converting image format to array style formato on which cv library operates
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(*msg, "bgr8");
    cv::Mat imported_image = cv_image->image;

    cv::Mat output_image = color_rec(imported_image);
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
        

        publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/camera/image_processed/compressed",
            qos_policy_);

        

        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/image_raw/compressed",
            qos_policy_,
            std::bind(&ImageProcessor::listener_callback, this, std::placeholders::_1));
    }

    cv::Mat color_rec(cv::Mat imported_image){
        // Extracting dimmensions for center moment calculation
        cv::Size img_size  = imported_image.size();
        int width = img_size.width;
        int height = img_size.height;
        cv::Point screen_centre(width/2, height/2); 
        // Converting format to hsv
        cv::Mat hsv_image;
        cv::cvtColor(imported_image, hsv_image, cv::COLOR_BGR2HSV);
        // Bitwise and with our mask to filter all objects with different colours than defined
        
        int h_upper = 140, h_lower = 95,  s_lower = 90;
        cv::Scalar lower = cv::Scalar(h_lower, s_lower, 50);
        cv::Scalar upper = cv::Scalar(h_upper, 255, 255);
        cv::Mat mask,masked_image;
        cv::inRange(hsv_image, lower, upper, mask);
        cv::bitwise_and(imported_image, imported_image, masked_image, mask);

        std::vector<std::vector<cv::Point> > contours, contours_to_draw;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
        if (contours.size() > 0){                           
            auto max_countour_iterator = std::max_element(
            contours.begin(),
            contours.end(),
            [] (const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
            {return cv::contourArea(a) < cv::contourArea(b);});
            std::vector<cv::Point> biggest_contour = *max_countour_iterator;
            contours_to_draw.push_back(biggest_contour);
            
            cv::Point contour_centre;
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

            cv::circle(masked_image, contour_centre, 7, cv::Scalar(0,0,255), -1);
            cv::line(masked_image, screen_centre, contour_centre, cv::Scalar(255,255,255),2);
            cv::drawContours(masked_image, contours_to_draw, -1, cv::Scalar(0, 255, 0), 2); 
        } 
        return masked_image;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProcessor>());
    rclcpp::shutdown();
    return 0;
}

