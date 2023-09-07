cmake_minimum_required(VERSION 3.10)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(grasp_synthesis)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(ryml REQUIRED)
link_directories(${CMAKE_PREFIX_PATH}/lib) #loguru

find_package(PCL CONFIG REQUIRED)

# Realsense
if (WIN32)
	set(RS_LIB "C:/Program Files (x86)/Intel RealSense SDK 2.0/lib/x64")
	set(RS_INC "C:/Program Files (x86)/Intel RealSense SDK 2.0/include")
else()
	set(LIBS_PREFIX "/usr/local")
	if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
		set(LIBS_PREFIX "/opt/homebrew") # Apple ARM macs
	endif()
	set(RS_LIB "${LIBS_PREFIX}/lib")
	set(RS_INC "${LIBS_PREFIX}/include")
endif()
find_library(REALSENSE2_FOUND realsense2 HINTS ${RS_LIB} REQUIRED)
message(STATUS "REALSENSE2_FOUND=${REALSENSE2_FOUND}")
# find_package(realsense2 REQUIRED) # This method is convenient but does not work on Windows, and finds the wrong librealsense on the robot

# OpenCV
if (WIN32)
	set(OpenCV_DIR "C:\\tools\\opencv\\build\\x64\\vc16\\lib")
endif()
# NOTE: does not work on Windows with clang; i.e. undefine CC=clang.exe, CXX=clang++.exe
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Executable
set(SRC_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/src/depth_to_cloud.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/rs_camera.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/config_file.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/camera_task.cpp
)

add_executable(grasp_synthesis ${SRC_FILES})
target_include_directories(grasp_synthesis PUBLIC ${RS_INC})
target_link_libraries(grasp_synthesis PUBLIC Threads::Threads Eigen3::Eigen ryml::ryml loguru)
target_link_libraries(grasp_synthesis PUBLIC ${REALSENSE2_FOUND})
target_link_libraries(grasp_synthesis PUBLIC ${OpenCV_LIBS})
target_link_libraries(grasp_synthesis PUBLIC ${PCL_LIBRARIES})
target_compile_definitions(grasp_synthesis PRIVATE -DUSE_CAMERAS_ON)

# FIXME: this needs to be integrated
add_executable(6D_pose_estimation_method1
	${CMAKE_CURRENT_SOURCE_DIR}/src/6D_pose_estimation_method1.cpp)
target_link_libraries(6D_pose_estimation_method1 PUBLIC ${OpenCV_LIBS} ${PCL_LIBRARIES} ${REALSENSE2_FOUND} Eigen3::Eigen)
target_include_directories(6D_pose_estimation_method1 PUBLIC ${RS_INC})

if (WIN32) # If using clang on Windows, if (MSVC) doesn't work
	target_compile_options(grasp_synthesis PRIVATE -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
endif()
