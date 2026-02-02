cmake_minimum_required(VERSION 3.8)
project(my_vision_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Finds the core ROS 2 build system package.
find_package(ament_cmake REQUIRED)

# Finds standard ROS 2 C++ client libraries.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rcl_interfaces REQUIRED)

# Finds the package used for handling resource paths (ament_index).
find_package(ament_index_cpp)

# Finding external packages
find_package(OpenCV REQUIRED)
# Finds the ONNX Runtime shared library specifically named 'onnxruntime'.
find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)

# Creates a shared library named 'image_processor_lib' from the source file.
# This separates the implementation logic from the executable entry point.
add_library(image_processor_lib src/image_processor.cpp)
# Links ROS 2 dependencies to the library.
# 'ament_target_dependencies' is a wrapper that handles both linking libraries
# and including header directories for ROS packages automatically.
ament_target_dependencies(image_processor_lib
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "rcl_interfaces")

# Includes OpenCV headers ONLY for this library (PRIVATE).
# 'PRIVATE' means that targets linking against this lib won't inherit these paths.
target_include_directories(image_processor_lib PUBLIC
  ${OpenCV_INCLUDE_DIRS})
# Links external (non-ROS) libraries to our library.
target_link_libraries(image_processor_lib
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB})
# Exposes the 'include' directory to this library and any target that links to it.
# 'PUBLIC' is used because the header file 'image_processor.hpp' is in 'include/',
# and it is part of the library's public interface.
target_include_directories(image_processor_lib PUBLIC
  include
  )
# Exposes the ONNX Runtime headers.
target_include_directories(image_processor_lib PUBLIC
  /usr/local/include/onnxruntime 
)
# Creates the executable binary named 'image_processor_node'.
add_executable(image_processor_node src/main.cpp)

# Adds ROS 2 dependencies to the executable.
ament_target_dependencies(image_processor_node 
  "rclcpp"
  "ament_index_cpp")

# Links the executable against our custom library created above.
# This allows 'main.cpp' to use the 'ImageProcessor' class.
target_link_libraries(image_processor_node
  image_processor_lib)

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

# Installs the executable and the library to the package's 'lib' directory.
# This makes them available to 'ros2 run'.
install(TARGETS 
  image_processor_node
  image_processor_lib
  DESTINATION lib/${PROJECT_NAME})

# Installs header files.
# This allows other packages to find your .hpp files.
install(
  DIRECTORY include/
  DESTINATION include/
)

# Installs launch files and configuration files to the 'share' directory.
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  )
install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
