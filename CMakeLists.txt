cmake_minimum_required(VERSION 3.5)
project(3d_object_ros)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(bboxes_ex_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(3d_object_ros_node src/3d_object_ros_node.cpp)

ament_target_dependencies(3d_object_ros_node
  rclcpp
  rclcpp_components
  vision_msgs
  bboxes_ex_msgs
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  message_filters
  pcl_conversions
)

target_link_libraries(3d_object_ros_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

install(TARGETS
  3d_object_ros_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
