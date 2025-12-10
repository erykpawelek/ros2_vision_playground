
#include <string>
#include <chrono>
#include <memory>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <unistd.h>

#include "my_vision_cpp/image_processor.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/qos.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


ImageProcessor::ImageProcessor(const std::string & node_name, const std::string & model_path)
:   Node(node_name),
    // Setting up QoS policy for camera stream
    qos_policy_(rclcpp::QoS(1).best_effort().durability_volatile()),
    // Benchmarking
    benchmark_last_frame_time_(std::chrono::steady_clock::now()),
    benchmark_running_(false),
    last_sys_time_(std::chrono::steady_clock::now()),
    // Dynamic paths
    onnx_model_path_(model_path),
    // Onnx evironment
    ort_env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_yolo8n"),
    ort_session_(nullptr),
    ort_session_options_()
{   
    this->declare_parameter("h_upper", 140);
    this->declare_parameter("h_lower", 95);
    this->declare_parameter("s_lower", 90);
    this->declare_parameter("mode", "color_rec");
    this->declare_parameter("benchmark_mode", false);
    this->declare_parameter("benchmark_start", false);
    this->declare_parameter("benchmark_duration", 60.0);
    
    h_upper_ = this->get_parameter("h_upper").as_int();
    h_lower_ = this->get_parameter("h_lower").as_int();
    s_lower_ = this->get_parameter("s_lower").as_int();
    mode_ = this->get_parameter("mode").as_string();
    benchmark_start_ = this->get_parameter("benchmark_start").as_bool();

    // Loading neural net path to enviornment
    try{
        ort_session_options_.SetInterOpNumThreads(1);
        ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        // Starting sesion
        ort_session_ = std::make_shared<Ort::Session>(ort_env_, onnx_model_path_.c_str(), ort_session_options_);
        onnx_init();
        RCLCPP_INFO(this->get_logger(), "Successfully loaded ONNX model from: %s", onnx_model_path_.c_str());
    } catch (const std::exception &e){
        RCLCPP_ERROR(this->get_logger(), "FATAL ERROR: Could not load ONNX model. Check path! Error: %s", e.what());
    }

    // Registering the callback to handle dynamic parameter updates
    params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&ImageProcessor::parameters_callback,
        this,
        std::placeholders::_1));

    //Publisher/ subscriber declarations
    publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "/camera/image_processed/compressed",
        qos_policy_);
    
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/camera/image_raw/compressed",
        qos_policy_,
        std::bind(&ImageProcessor::listener_callback, this, std::placeholders::_1));
};

ImageProcessor::~ImageProcessor(){
    if(!log_timestamps_.empty()){
        save_benchmark_data();
    }
}

void ImageProcessor::listener_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg){
    // Converting image format to array style formato on which cv library operates
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(*msg, "bgr8");
    cv::Mat imported_image = cv_image->image, output_image;

    // Mode selection
    if (mode_ == "color_rec"){
        output_image = color_rec(imported_image);
    }else if (mode_ == "neural_net_onnx"){
        output_image = neural_net_onnx(imported_image);
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

rcl_interfaces::msg::SetParametersResult ImageProcessor::parameters_callback(
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
            if (val_str == "color_rec" || val_str == "neural_net_onnx"){
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

cv::Mat ImageProcessor::color_rec(cv::Mat & imported_image){
    // Extracting dimmensions for center moment calculation
    cv::Size img_size  = imported_image.size();     
    int width = img_size.width;
    int height = img_size.height;
    cv::Point screen_centre(width/2, height/2); 
    // Converting format to hsv for more roboust color filtration selction
    cv::Mat hsv_image;       
    cv::cvtColor(imported_image, hsv_image, cv::COLOR_BGR2HSV);
    // Defining color to filter through
    cv::Scalar lower = cv::Scalar(h_lower_, s_lower_, 50);
    cv::Scalar upper = cv::Scalar(h_upper_, 255, 255);
    cv::Mat mask,masked_image;
    cv::inRange(hsv_image, lower, upper, mask);
    // Bitwise and filter our image, leaving only defined above color
    cv::bitwise_and(imported_image, imported_image, masked_image, mask); 

    std::vector<std::vector<cv::Point> > contours, contours_to_draw;
    std::vector<cv::Vec4i> hierarchy;
    // Finding all countours (not: for contour function image has to be single chanel)
    cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE); 
    if (contours.size() > 0){
        // Finding biggest countour                           
        auto max_countour_iterator = std::max_element(     
        contours.begin(),
        contours.end(),
        [] (const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
        {return cv::contourArea(a) < cv::contourArea(b);});
        std::vector<cv::Point> biggest_contour = *max_countour_iterator;
        contours_to_draw.push_back(biggest_contour);

        // Finding mas centre of our countour using static moments
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
        // Visualizing results
        cv::circle(masked_image, contour_centre, 7, cv::Scalar(0,0,255), -1);   
        cv::line(masked_image, screen_centre, contour_centre, cv::Scalar(255,255,255),2);
        cv::drawContours(masked_image, contours_to_draw, -1, cv::Scalar(0, 255, 0), 2); 
    } 
    return masked_image;
}

void ImageProcessor::onnx_init(){
    // Dynamic model data loading
    // Creating memory allocator due to fact that onnx in build on C language.
    OrtAllocator* allocator = Ort::AllocatorWithDefaultOptions();
    for (size_t i = 0; i < ort_session_.get()->GetInputCount(); ++i){
        Ort::AllocatedStringPtr ptr = ort_session_.get()->GetInputNameAllocated(i, allocator);    
        input_node_names_.push_back(std::string(ptr.get()));

        Ort::TypeInfo type_info = ort_session_->GetInputTypeInfo(i);
        // GetInputTypeInfo method returns read-only file so it is neccesery to use ConstTensorTypeAndShape
        Ort::ConstTensorTypeAndShapeInfo type_shape_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> current_shape = type_shape_info.GetShape();
        input_node_dims_.push_back(current_shape);
    }

    for (size_t i = 0; i < ort_session_.get()->GetOutputCount(); ++i){
        Ort::AllocatedStringPtr ptr = ort_session_.get()->GetOutputNameAllocated(i, allocator);
        output_node_names_.push_back(std::string(ptr.get()));

        Ort::TypeInfo type_info = ort_session_->GetOutputTypeInfo(i);
        Ort::ConstTensorTypeAndShapeInfo type_shape_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> current_shape = type_shape_info.GetShape();
        output_node_dims_.push_back(current_shape);
    }
    
        input_tensor_size_ = std::accumulate(
        input_node_dims_[0].begin(),
        input_node_dims_[0].end(),
        (size_t) 1,
        [] (int64_t a, int64_t b){return a * b;});
    
    for (size_t i =0; i < input_node_names_.size(); ++i){
        input_names_.push_back(input_node_names_[i].c_str());
    }
    for (size_t i=0; i < output_node_names_.size(); ++i){
        output_names_.push_back(output_node_names_[i].c_str());
    }
    // By deviding loops we avoid memory relocation while getting const char* for Run() function
}

cv::Mat ImageProcessor:: neural_net_onnx(cv::Mat & imported_image){
    // Converting standard cv::Mat to 4-dimensional Mat with NCHW dimensions order
    cv::Mat blob_prep = cv::dnn::blobFromImage(
        imported_image,
        1.0/255.0,
        cv::Size(cv::Point(640, 640)),
        cv::Scalar(0,0,0),
        true);

    // Finding RAM memory adress
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // CV -> Ort, bridge
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        blob_prep.ptr<float>(), // Start adress
        input_tensor_size_, 
        input_node_dims_[0].data(), 
        input_node_dims_[0].size());

    try {
        auto output_tensor = ort_session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            input_names_.size(),
            output_names_.data(),
            output_node_names_.size());

        imported_image = onnx_draw_boundries(output_tensor, imported_image);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "ONNX runtime error %s", e.what());
    }
    return imported_image;

}

cv::Mat ImageProcessor::onnx_draw_boundries(
    const std::vector<Ort::Value> & output_tensor,
     cv::Mat & imported_image)
{
    // Obtaining ptr to first object in result structure
    const Ort::Value & output = output_tensor.front();
    auto data = output.GetTensorData<float>();
    auto type_size = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape  = type_size.GetShape();
    cv::Mat output_matrix(shape[1], shape[2], CV_32F, const_cast<float*>(data));
    // Transposing matrix for better handling of results
    cv::Mat transposed = output_matrix.t();
    // Recactor cooefficents to remap image from input ex. 640x640 to ex. 1920x1080
    float model_heigh = static_cast<float>(input_node_dims_[0][2]);
    float model_width = static_cast<float>(input_node_dims_[0][3]);
    float x_factor = imported_image.cols / model_width;
    float y_factor = imported_image.rows / model_heigh;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> indices;
    float intersection_union = 0.6;
    float coeficient_factor = 0.5;

    for (int i = 0; i < transposed.rows; ++i){
        // Due to row major convention we obtain here adress of desired row
        float* row_ptr = transposed.ptr<float>(i);
        // Finding max_class propability returnet by network 0, 1, 2, 3 positions are x, y, h, w
        auto max_class_ptr = std::max_element(
            row_ptr + 4,
            row_ptr + transposed.cols);
        // Finding max_class index
        int max_class_id = std::distance(
        row_ptr + 4,
        max_class_ptr);
        // Checking cenrtainity cofactor
        float score = *max_class_ptr;
        if (score > 0.5f){
            float cx = row_ptr[0];
            float cy = row_ptr[1];
            float w = row_ptr[2];
            float h = row_ptr[3];

            int left = static_cast<int>((cx - 1.0/2.0 * w) * x_factor);
            int top = static_cast<int>((cy - 1.0/2.0 * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(score);
            class_ids.push_back(max_class_id);
        }
    }
    cv::dnn::NMSBoxes(boxes, confidences, coeficient_factor, intersection_union, indices);
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        float confidence = confidences[idx];
        u_int class_id = class_ids[idx];

        cv::Scalar color = cv::Scalar(0, 255, 0);
        cv::rectangle(imported_image, box, color, 2); 

        std::string class_name = "Class ";
        std::string label = class_name + std::to_string(class_id) + ": " + cv::format("%.2f", confidence);
        
        cv::rectangle(imported_image, box, color, 2);
        cv::putText(
            imported_image, 
            label, 
            cv::Point(box.x, box.y- 40),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            color);
        }
    return imported_image;
}

void ImageProcessor::benchmark(){
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
        if (elapsed_f >= this->get_parameter("benchmark_duration").as_double()){
            rclcpp::shutdown();
        }
    }
}

double ImageProcessor::get_ram_usage(){
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

double ImageProcessor::get_cpu_usage(){
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

void ImageProcessor::save_benchmark_data(){
    RCLCPP_WARN(this->get_logger(),"Saving benchmark results to file");
    std::ofstream benchmark_file("benchmark_results_cpp.csv");
    // Header for csv file
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