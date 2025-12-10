cmake_minimum_required(VERSION 3.8)
project(my_vision_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Finding packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ament_index_cpp)

# Finding external packages
find_package(OpenCV REQUIRED)

# Finding localy installed library
find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)

# Defining library to build
add_library(image_processor_lib src/image_processor.cpp)
# Adding dependencies ROS2
ament_target_dependencies(image_processor_lib
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "rcl_interfaces")

# Show to compiler where are hedder files (.hpp) OpenCV
target_include_directories(image_processor_lib PRIVATE
  ${OpenCV_INCLUDE_DIRS})
# Manual linking ONNX and open CV
target_link_libraries(image_processor_lib
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB})

target_include_directories(image_processor_lib PUBLIC
  include
  )

target_include_directories(image_processor_lib PUBLIC
  /usr/local/include/onnxruntime 
)

add_executable(image_processor_node src/main.cpp)
ament_target_dependencies(image_processor_node 
  "rclcpp"
  "ament_index_cpp")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

target_link_libraries(image_processor_node
  image_processor_lib)

install(TARGETS 
  image_processor_node
  image_processor_lib
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  )

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
