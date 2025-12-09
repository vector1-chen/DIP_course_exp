cmake_minimum_required(VERSION 3.0.2)
project(final_exp)

# 确保找到必要的 ROS 组件和 OpenCV
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  image_transport
)

find_package(OpenCV REQUIRED)

catkin_package()

# 包含路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 构建可执行文件
add_executable(final_exp src/final_exp.cpp)
target_link_libraries(final_exp
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)