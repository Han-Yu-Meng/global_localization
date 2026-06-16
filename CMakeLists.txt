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

# 1. 定义 cpu_bbs3d 静态库
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

# 2. 【关键修正】为 cpu_bbs3d 静态库应用最高级别的 CPU 优化选项和硬件矢量化
target_compile_options(cpu_bbs3d PRIVATE
  -O3
  -ffast-math
  -DNDEBUG
  -march=native  # 允许编译器利用当前编译机器支持的所有 CPU 特性（AVX2, AVX512, SSE4 等）
)

# 3. 定义并编译 ROS 节点
fins_add_node(${PROJECT_NAME}
  node.cpp
)

fins_link_ros_dependencies(${PROJECT_NAME}
  geometry_msgs
  nav_msgs
  tf2_eigen
)

# 4. 【关键修正】同样为节点目标应用最高级别优化和矢量化
target_compile_options(${PROJECT_NAME} PRIVATE
  -O3
  -ffast-math
  -DNDEBUG
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  cpu_bbs3d
  small_gicp::small_gicp
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)