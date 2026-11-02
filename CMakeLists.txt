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

target_link_libraries(cpu_bbs3d PRIVATE
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

# 2. SIMD 选项必须由所有包含 bbs3d.hpp 的目标共享：
#    Eigen 的 EIGEN_MAX_ALIGN_BYTES 会随 ISA 选项变化，两个 TU 看到的
#    alignof(BBS3D) 不一致时，new 出来的对象按较低对齐分配，构造函数却按
#    更高对齐写入，直接 SIGSEGV。
#    这里刻意不使用 -march=native：保持 x86-64 通用指令集，保证不同 x64 机器
#    上编译/运行行为一致（可迁移性优先于峰值性能）。
add_library(bbs3d_simd_flags INTERFACE)
target_compile_options(bbs3d_simd_flags INTERFACE
  -O3
  -ffast-math
  -DNDEBUG
)

target_link_libraries(cpu_bbs3d PRIVATE bbs3d_simd_flags)

# 3. 定义并编译 ROS 节点
fins_add_node(${PROJECT_NAME}
  node.cpp
)

fins_link_ros_dependencies(${PROJECT_NAME}
  geometry_msgs
  nav_msgs
  tf2_eigen
  visualization_msgs
)

# 4. 节点目标与 cpu_bbs3d 使用完全相同的 SIMD 选项（切勿只为静态库单独加 ISA 选项）
target_link_libraries(${PROJECT_NAME} PRIVATE
  bbs3d_simd_flags
  cpu_bbs3d
  small_gicp::small_gicp
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)