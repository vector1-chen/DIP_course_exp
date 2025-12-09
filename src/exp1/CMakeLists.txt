cmake_minimum_required(VERSION 3.0.2)
project(exp1)
 
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

add_executable(exp1 src/exp1.cpp)
add_executable(car_demo src/car_demo.cpp)

target_link_libraries(car_demo
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
)

target_link_libraries(exp1
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
)