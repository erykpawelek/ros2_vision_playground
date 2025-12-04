cmake_minimum_required(VERSION 3.8)
project(my_vision_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
# Standard executable definition
add_executable(img_processor_cpp src/img_processor_node.cpp)
ament_target_dependencies(img_processor_cpp 
  "rclcpp"
  "sensor_msgs"
  "cv_bridge")

# Show to compiler where are hedder files (.hpp) OpenCV
target_include_directories(img_processor_cpp PRIVATE
  ${OpenCV_INCLUDE_DIRS})
# Manual linking of external libraries
target_link_libraries(img_processor_cpp
  ${OpenCV_LIBS})

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

# Telling where to place compiled files
install(TARGETS
  img_processor_cpp
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  )


ament_package()
