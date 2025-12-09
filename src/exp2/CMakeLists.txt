cmake_minimum_required(VERSION 3.0.2)
project(exp2)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge

)


find_package(OpenCV REQUIRED)



catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(exp2 src/exp2.cpp)

target_link_libraries(exp2
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
)
