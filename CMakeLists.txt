cmake_minimum_required(VERSION 3.16)
project(global_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-parameter)
endif()

find_package(PCL REQUIRED COMPONENTS common io filters registration)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(small_gicp REQUIRED)

include_directories(
  bbs3d/include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(cpu_bbs3d
  STATIC
  bbs3d/src/cpu_bbs3d/bbs3d.cpp
  bbs3d/src/cpu_bbs3d/voxelmaps.cpp
  bbs3d/src/cpu_bbs3d/voxelmaps_io.cpp
)

set_target_properties(cpu_bbs3d PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cpu_bbs3d
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

fins_add_node(${PROJECT_NAME}
  node.cpp
)

fins_link_ros_dependencies(${PROJECT_NAME}
  geometry_msgs
  nav_msgs
  tf2_eigen
)

target_compile_options(${PROJECT_NAME} PRIVATE
  -O3 -ffast-math -DNDEBUG
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  cpu_bbs3d
  small_gicp::small_gicp
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)