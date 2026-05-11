
cmake_minimum_required(VERSION 3.16)
project(fins_wrapper LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/home/steven/.fins/install;/home/steven/.fins/install/lib;/home/steven/.fins/dependencies/install/small_gicp/1.0.0-0f5a6e88/lib;/home/steven/.fins/dependencies/install/Eigen3/3.4.0-fb94cfdb/lib")

set(FINS_DEP_PATHS "/home/steven/.fins/dependencies/install/small_gicp/1.0.0-0f5a6e88;/home/steven/.fins/dependencies/install/Eigen3/3.4.0-fb94cfdb")
if(FINS_DEP_PATHS)
	list(INSERT CMAKE_PREFIX_PATH 0 ${FINS_DEP_PATHS})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
# Globally force define FMT_HEADER_ONLY to prevent subproject overrides
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC -DFMT_HEADER_ONLY")
add_compile_definitions(FMT_HEADER_ONLY)

# Ensure internal fmt header path has highest priority
include_directories(BEFORE "/home/steven/.fins/install/include/fins/third_party/fmt/include")
include_directories("/home/steven/.fins/install/include")

# --- FINS PRE-LOAD DEPENDENCIES ---
message(STATUS "[FINS] Pre-loading dependency: Eigen3")
find_package(Eigen3 REQUIRED)
message(STATUS "   -- Locked Eigen3 to: ${Eigen3_DIR}")
message(STATUS "[FINS] Pre-loading dependency: small_gicp")
find_package(small_gicp REQUIRED)
message(STATUS "   -- Locked small_gicp to: ${small_gicp_DIR}")

# ----------------------------------


if(EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop")
    message(STATUS "WSL Environment Detected: Sanitizing environment...")

    set(_current_path "$ENV{PATH}")
    string(REPLACE ":" ";" _path_list "${_current_path}")
    set(_clean_path_list "")
    
    foreach(_p ${_path_list})
        if(NOT "${_p}" MATCHES "^/mnt/")
            list(APPEND _clean_path_list "${_p}")
        endif()
    endforeach()
    
    list(JOIN _clean_path_list ":" _clean_path)
    set(ENV{PATH} "${_clean_path}")
    message(STATUS "PATH sanitized: Windows paths removed from CMake visibility.")

    list(APPEND CMAKE_SYSTEM_IGNORE_PATH "/mnt/c" "/mnt/d" "/mnt/e" "/mnt/f" "/mnt/g" "/mnt")
    list(APPEND CMAKE_IGNORE_PATH "/mnt/c" "/mnt/d" "/mnt/e" "/mnt/f" "/mnt/g" "/mnt")
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "/mnt/c" "/mnt/d" "/mnt/e" "/mnt/f" "/mnt/g" "/mnt")

    get_cmake_property(_cacheVars CACHE_VARIABLES)
    foreach(_var ${_cacheVars})
        get_property(_val CACHE ${_var} PROPERTY VALUE)
        if("${_val}" MATCHES "^/mnt/")
            if("${_var}" MATCHES "(DIR|LIBRARY|INCLUDE|FILE|PATH|PROGRAM)$")
                message(STATUS "Clearing polluted cache var: ${_var} = ${_val}")
                unset(${_var} CACHE)
            endif()
        endif()
    endforeach()
endif()



function(fins_link_ros_dependencies target)
    foreach(pkg ${ARGN})
        find_package(${pkg} REQUIRED)
        message(STATUS "[FINS] Linking ${pkg} to ${target}")

        if(TARGET ${pkg}::${pkg})
            target_link_libraries(${target} PRIVATE ${pkg}::${pkg})
        
        elseif(TARGET ${pkg})
            target_link_libraries(${target} PRIVATE ${pkg})
        
        else()
            if(DEFINED ${pkg}_INCLUDE_DIRS)
                target_include_directories(${target} PRIVATE ${${pkg}_INCLUDE_DIRS})
            elseif(DEFINED ${pkg}_INCLUDE_DIR)
                target_include_directories(${target} PRIVATE ${${pkg}_INCLUDE_DIR})
            endif()

            if(DEFINED ${pkg}_LIBRARIES)
                target_link_libraries(${target} PRIVATE ${${pkg}_LIBRARIES})
            elseif(DEFINED ${pkg}_LIBRARY)
                target_link_libraries(${target} PRIVATE ${${pkg}_LIBRARY})
            endif()
        endif()
    endforeach()
endfunction()

macro(fins_optional_ros_dependency target pkg_name)
    find_package(${pkg_name} QUIET)
    if(${pkg_name}_FOUND)
        message(STATUS ">> Optional Package '${pkg_name}': FOUND. Enabling...")
        fins_link_ros_dependencies(${target} ${pkg_name})
        string(TOUPPER ${pkg_name} _PKG_UPPER)
        target_compile_definitions(${target} PRIVATE WITH_${_PKG_UPPER})
    else()
        message(STATUS ">> Optional Package '${pkg_name}': NOT FOUND. Skipping.")
    endif()
endmacro()


# === USE PRE-BUILT SDK STATIC LIBRARY ===
add_library(fins_sdk STATIC IMPORTED GLOBAL)
set_target_properties(fins_sdk PROPERTIES
    IMPORTED_LOCATION "/home/steven/.fins/install/lib/libfins_sdk_static.a"
    INTERFACE_INCLUDE_DIRECTORIES "/home/steven/.fins/install/include;/home/steven/.fins/install/include/fins/third_party/fmt/include"
)

target_link_libraries(fins_sdk INTERFACE 
    Threads::Threads 
    ${CMAKE_DL_LIBS}
)

add_library(fins_shared ALIAS fins_sdk)

macro(fins_add_node _target)
    add_library(${_target} SHARED ${ARGN})
    target_link_libraries(${_target} PRIVATE 
        fins_sdk
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(${_target} PRIVATE FMT_HEADER_ONLY FINS_NODE)
    set_target_properties(${_target} PROPERTIES 
        OUTPUT_NAME "${PKG_SOURCE}_${_target}"
        POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH "/home/steven/.fins/install;/home/steven/.fins/install/lib;/home/steven/.fins/dependencies/install/small_gicp/1.0.0-0f5a6e88/lib;/home/steven/.fins/dependencies/install/Eigen3/3.4.0-fb94cfdb/lib"
    )
endmacro()

add_compile_definitions(PKG_NAME="${FINS_META_NAME}")
add_compile_definitions(PKG_VERSION="${FINS_META_VERSION}")
add_compile_definitions(PKG_MAINTAINER="${FINS_META_MAINTAINER}")
add_compile_definitions(PKG_DESCRIPTION="${FINS_META_DESC}")
add_compile_definitions(PKG_SOURCE="${FINS_META_SOURCE}")

add_subdirectory("/home/steven/nav_ws/global_localization" "${CMAKE_BINARY_DIR}/node_build")

if(TARGET ${FINS_META_NAME})
    set_target_properties(${FINS_META_NAME} PROPERTIES OUTPUT_NAME "${FINS_META_SOURCE}_${FINS_META_NAME}")
endif()
