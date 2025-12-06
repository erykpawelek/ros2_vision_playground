#include <memory>
#include <string>
#include <fstream>
#include <chrono>
#include <unistd.h>

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
    // Qos policy
    rclcpp::QoS qos_policy_;
    // Parameters values
    int h_upper_, h_lower_, s_lower_;
    bool benchmark_start_;
    std::string mode_;
    // Benchmarking
    std::chrono::steady_clock::time_point benchmark_last_frame_time_; 
    std::chrono::steady_clock::time_point benchmark_start_time_;
    std::chrono::seconds benchmark_duration_;
    bool benchmark_running_;
    std::ofstream csv_file_;
    // System log data collection asisting variables
    std::clock_t last_cpu_time_ ;
    std::chrono::steady_clock::time_point last_sys_time_;
    // Data colection containers
    std::vector<double> log_timestamps_;
    std::vector<double> log_fps_;
    std::vector<double> log_cpu_;
    std::vector<double> log_ram_;


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
    
    benchmark();
    publisher_->publish(*output_msg.toCompressedImageMsg());
}

public:
    ImageProcessor()
    :   Node("image_processor"),
        // Setting up QoS policy for camera stream
        qos_policy_(rclcpp::QoS(1).best_effort().durability_volatile()),
        last_sys_time_(std::chrono::steady_clock::now()),
        benchmark_last_frame_time_(std::chrono::steady_clock::now()),
        benchmark_running_(false)

    {   
        this->declare_parameter("h_upper", 140);
        this->declare_parameter("h_lower", 95);
        this->declare_parameter("s_lower", 90);
        this->declare_parameter("mode", "color_rec");
        this->declare_parameter("benchmark_mode", false);
        this->declare_parameter("benchmark_start", false);
        this->declare_parameter("benchmark_duration", 60);
        
        h_upper_ = this->get_parameter("h_upper").as_int();
        h_lower_ = this->get_parameter("h_lower").as_int();
        s_lower_ = this->get_parameter("s_lower").as_int();
        mode_ = this->get_parameter("mode").as_string();
        benchmark_start_ = this->get_parameter("benchmark_start").as_bool();
        
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

    ~ImageProcessor(){
        if(!log_timestamps_.empty()){
            save_benchmark_data();
        }
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
        const std::vector<rclcpp::Parameter> & parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        int val_int;
        std::string val_str;
        bool val_bool;

        for(const auto &param : parameters)
        {
            if (param.get_name() == "h_upper"){
                val_int = param.as_int();
                if (val_int >= 0 && val_int <= 179){
                    result.successful = true;
                    this->h_upper_ = val_int;
                    result.reason = "Success!";
                }else{
                    result.successful = false;
                    result.reason = "HSV hue value is in range 0-179!";
                }
            }else if (param.get_name() == "h_lower"){
                val_int = param.as_int();
                if (val_int >= 0 && val_int <= 179){
                    result.successful = true;
                    this->h_lower_ = val_int;
                    result.reason = "Success!";
                }else{
                    result.successful = false;
                    result.reason = "HSV hue value is in range 0-179!";
                }
            }else if (param.get_name() == "s_lower"){
                    val_int = param.as_int();
                if (val_int >= 0 && val_int <= 255){
                    result.successful = true;
                    this->s_lower_ = val_int;
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
            }else if (param.get_name() == "benchmark_start"){
                this->benchmark_start_ = param.as_bool();
                result.successful = true;
                result.reason = "Success!";
            }
        }
        return result;
    }

    void benchmark(){
        std::chrono::steady_clock::time_point stop_time =std::chrono::steady_clock::now();
        auto cycle_duration  = stop_time - benchmark_last_frame_time_;
        double cycle_duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(cycle_duration).count();
        benchmark_last_frame_time_ = stop_time;
        double fps;
        if (cycle_duration_seconds > 0){
            fps = 1 / cycle_duration_seconds;
            RCLCPP_INFO(this->get_logger(),"Current FPS: %.2f", fps);
        }

        if (benchmark_start_){
            if (!benchmark_running_){
                benchmark_start_time_ = std::chrono::steady_clock::now();
                benchmark_running_ = true;
            }
            auto cpu_usage = get_cpu_usage();
            auto ram_usage = get_ram_usage();
            auto elapsed = stop_time - benchmark_start_time_;
            double elapsed_f = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
            log_timestamps_.push_back(elapsed_f);
            log_cpu_.push_back(cpu_usage);
            log_ram_.push_back(ram_usage);
            log_fps_.push_back(fps);
        }
    }

    /**
     * @brief Loads data about RAM usage of process from system logs.
     * @return Value of RAM usage in Mb.
     */
    double get_ram_usage(){
        // Creating input file stream
        std::ifstream stat_stream("/proc/self/status");
        std::string line;
        long rss_kb = 0;

        while (std::getline(stat_stream,line)){
            // Looking for line Vm Resident Set Size
            if (line.substr(0, 6) == "VmRSS:"){
                std::stringstream ss(line);
                std::string label;
                // Extracting RAM kb usage
                ss >> label;
                ss >> rss_kb;
                break;
            }
        }
        return rss_kb / 1024.0;
    }
    /**
     * @brief Loads data about CPU usage of process from system logs and performs neccessary calculation to obtain result in %.
     * @return Procentage value of processor usage.
     */
    double get_cpu_usage(){
        // Creating input file stream
        std::ifstream stat_file("/proc/self/stat");
        // Due to "/proc/self/stat" file structure we can place it's whole content into one 'line'.
        std::string line;
        std::getline(stat_file, line);
        std::stringstream ss(line);
        std::string trash;
        // Looking for utime and stime which are on 14th and 15th position of this file. 
        // User time is time which processor spend on this proces.
        // System time is time which processor spend in kernel mode reading files, alocating memory, networc protocols...
        for(int i=0; i<13; ++i) ss >> trash;

        long unsigned int utime, stime;
        ss >> utime >> stime;
        // We want all cycle time including network processes and others
        long total_cpu_ticks = utime + stime;
        auto now = std::chrono::steady_clock::now();
        double cpu_percent = 0.0;
        // Calculating ticks and time between cycles
        if (last_cpu_time_ > 0){
            long ticks_diff = total_cpu_ticks - last_cpu_time_;

            std::chrono::duration<double> time_diff = now - last_sys_time_;
            double seconds = time_diff.count();
            // Converting ticks to time to find how much of time our processor were occupied by process
            long clk_tck = sysconf(_SC_CLK_TCK);
            if (seconds > 0){
                // Calculating procesor usage by formula: busy_time_of_procesor / sumaric_time_of_cycle, *hint: sysconf(_SC_CLK_TCK) is amount of time for one processors tick
                cpu_percent = (double(ticks_diff) / clk_tck) / seconds * 100;
            }
        }
        // Actualizing ticks and time for next cycle
        last_cpu_time_ = total_cpu_ticks;
        last_sys_time_ = now;
        return cpu_percent;
    }

    void save_benchmark_data(){
        RCLCPP_WARN(this->get_logger(),"Saving benchmark results to file");
        std::ofstream benchmark_file("benchmark_results_cpp.csv");
        // Header
        if (benchmark_file.is_open()) {
            benchmark_file << "Timestamp,FPS,CPU_Usage_Percent,RAM_Usage_MB\n";
            for(size_t i = 0; i < log_timestamps_.size(); ++i){
                benchmark_file
                << log_timestamps_[i] << ","
                << log_fps_[i] << ","
                << log_cpu_[i] << ","
                << log_ram_[i] << "\n";
            }
            benchmark_file.close();
            RCLCPP_WARN(this->get_logger(), "Data save successfull!");
        }else{
            RCLCPP_ERROR(this->get_logger(), "Data hasn't been saved!");
        }
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProcessor>());
    rclcpp::shutdown();
    return 0;
}

