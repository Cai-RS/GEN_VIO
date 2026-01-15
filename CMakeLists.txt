cmake_minimum_required(VERSION 3.16)
project(GEN_VIO LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "release")
# set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CUDA_BIN_PATH "/usr/local/cuda-11.6")

#set(CUDA_GEN_CODE "-gencode=arch=compute_86, code=sm_86")

set(CUDA_TOOLKIT_ROOT_DIR  "/usr/local/cuda-11.6")
set(TENSORRT_DIR "/usr/local/TensorRT/TensorRT-8.5.3.1")
set(OpenCV_DIR   "/usr/local/opencv-4.6.0")
set(CAMERA_MODEL "camera_models")
set(Ceres_DIR /usr/local/include/ceres_1.14.0/lib/cmake/Ceres)
set(CMAKE_CXX_FLAGS_DEGUG "-std=c++11 -O2 -Wall;")

# 所有文件都进行堆栈错误监测
#[[if (NOT WIN32 AND NOT HAIKU)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstack-protector")
endif()]]

# 不是对所有的文件都进行ASAN监测，因为其与CUDA有冲突
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

add_definitions(-w)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
#find_package(catkin REQUIRED COMPONENTS
  #cv_bridge
  #geometry_msgs
  #image_transport
  #nav_msgs
  #roscpp
  #rospy
  #std_msgs
  #sensor_msgs
#)

find_package(OpenCV 4.6 REQUIRED)
#FindCUDAToolkit(-DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR})
find_package(CUDA 11.6 REQUIRED)
find_package(Boost REQUIRED)
find_package(Ceres 1.14.0 REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(Eigen3 REQUIRED)

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT/TensorRT-8.5.3.1/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/ NO_DEFAULT_PATH)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64 NO_DEFAULT_PATH)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64 NO_DEFAULT_PATH)
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64 NO_DEFAULT_PATH)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER} ${TENSORRT_LIBRARY_PLUGIN})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")


#find_package(CUDA QUIET REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(
  ${PROJECT_SOURCE_DIR}/src/CudaSift
  ${PROJECT_SOURCE_DIR}/src/estimator
  ${PROJECT_SOURCE_DIR}/src/factor
  ${PROJECT_SOURCE_DIR}/src/featureTracker
  ${PROJECT_SOURCE_DIR}/src/GPUProcess
  ${PROJECT_SOURCE_DIR}/src/initial
  ${PROJECT_SOURCE_DIR}/src/utility
  ${PROJECT_SOURCE_DIR}/src/TensorRT/include
  ${PROJECT_SOURCE_DIR}/src/evaluate
  #${PROJECT_SOURCE_DIR}/${CAMERA_MODEL}/build
  ${PROJECT_SOURCE_DIR}/camera_models/include
  ${OpenCV_DIR}/include/opencv4
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  #${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${TENSORRT_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

link_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${OpenCV_DIR}/lib
  ${PROJECT_SOURCE_DIR}/${CAMERA_MODEL}/build
)

#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}; -std=c++11; -Wall; -O0; -Wfatal-errors; -pthread; -w; -g")

#set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -O3 -fPIC" )
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -std=c++11; -O0; -Xcompiler; -fPIC")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")

file(GLOB_RECURSE cuda_srcs 
${PROJECT_SOURCE_DIR}/src/GPUProcess/*.cu 
${PROJECT_SOURCE_DIR}/src/CudaSift/cudaImage.cu
${PROJECT_SOURCE_DIR}/src/CudaSift/cudaSiftH.cu
${PROJECT_SOURCE_DIR}/src/CudaSift/matching.cu)

add_library(plugin_list SHARED ${cuda_srcs})
target_link_libraries(plugin_list ${TENSORRT_LIBRARY})
target_link_libraries(plugin_list cuda cublas cudart cudnn -lcudadevrt)
target_link_libraries(plugin_list pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})
target_link_libraries(plugin_list opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs -lffi)
# 在usr/lib路经下创建该文件的软链接即可
#target_link_libraries(plugin_list /usr/lib/x86_64-linux-gnu/libf77blas.so)
#target_link_libraries(plugin_list TensorRTBuffer)
set_target_properties(plugin_list PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(plugin_list PROPERTIES CUDA_ARCHITECTURES "86")
# https://blog.csdn.net/limanjihe/article/details/115309659
# -g代表对程序进行调试性编译，可能会导致opencv的部分函数在运行时不可用. -O2代表编译器对代码的优化等级，分为0-4级，0级就是基本不优化
target_compile_options(plugin_list PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17; -O0; -Xcompiler; -fPIC; -g; -w>)

## 给定路径以便可以找到自己编译的第三方库 CUDASift 和 plugin_list_CudaSift的h文件
##include_directories(/home/crs/GEN_VIO/src/dependencies/CudaSift-Pascal/build)

## 把使用CUDA的所有文件放在一起进行编译，这部分不使用ASAN监测（因为CUDA与ASAN库有冲突，最好的办法就是把需要和不需要被监测的文件分开来编译）
add_library(${PROJECT_NAME}_1_lib SHARED
  src/CudaSift/Sift.cpp
  src/CudaSift/geomFuncs.cpp
  src/CudaSift/geomFuncs.cpp
  src/GPUProcess/stereo_depth.cpp
  src/GPUProcess/yolov8_seg.cpp
  src/GPUProcess/rapidflow.cpp
  src/TensorRT/src/cuda_tools.cpp
  src/TensorRT/src/ilogger.cpp
  src/TensorRT/src/logger.cpp
)

target_link_libraries(${PROJECT_NAME}_1_lib
  plugin_list
  ${CAMERA_MODEL}
  ${TENSORRT_LIBRARY}
  ${OpenCV_LIBRARIES}
  #${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES}
  /usr/lib/x86_64-linux-gnu/libffi.so.7
)

target_link_libraries(${PROJECT_NAME}_1_lib cuda cublas cudart cudnn)
target_link_libraries(${PROJECT_NAME}_1_lib pthread)
target_link_libraries(${PROJECT_NAME}_1_lib opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs -lffi)
target_compile_options(${PROJECT_NAME}_1_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17; -Wall; -O0; -Wfatal-errors; -pthread; -Werror=return-type; -g>)
#target_compile_options(${PROJECT_NAME}_1_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17; -Wall; -fstack-protector-all; -Wstack-protector; -O2; -Wfatal-errors; -pthread; -Werror=return-type; -g>)

# 接下来编译需要进行ASAN监测的文件
add_library(${PROJECT_NAME}_2_lib SHARED
  src//estimator/parameters.cpp
  src/estimator/estimator.cpp
  src/estimator/feature_manager.cpp
  src/estimator/utils.cc
  src/factor/pose_local_parameterization.cpp
  src/factor/projectionTwoFrameOneCamFactor.cpp
  src/factor/projectionTwoFrameTwoCamFactor.cpp
  src/factor/projectionOneFrameTwoCamFactor.cpp
  src/factor/marginalization_factor.cpp
  src/utility/utility.cpp
  #src/utility/visualization.cpp
  #src/utility/CameraPoseVisualization.cpp
  src/initial/solve_5pts.cpp
  src/initial/initial_aligment.cpp
  src/initial/initial_sfm.cpp
  src/initial/initial_ex_rotation.cpp
  src/featureTracker/feature_tracker.cpp
  src/evaluate/evaluate_odometry.cpp
  src/evaluate/matrix.cpp
  
  #[[src/CudaSift/Sift.cpp
  src/CudaSift/geomFuncs.cpp
  src/GPUProcess/stereo_depth.cpp
  src/GPUProcess/yolov8_seg.cpp
  src/GPUProcess/rapidflow.cpp
  src/TensorRT/src/cuda_tools.cpp
  src/TensorRT/src/ilogger.cpp
  src/TensorRT/src/logger.cpp]]
  #[[camera_models/src/chessboard/Chessboard.cc
  camera_models/src/calib/CameraCalibration.cc
  camera_models/src/camera_models/Camera.cc
  camera_models/src/camera_models/CameraFactory.cc
  camera_models/src/camera_models/CostFunctionFactory.cc
  camera_models/src/camera_models/PinholeCamera.cc
  camera_models/src/camera_models/PinholeFullCamera.cc
  camera_models/src/camera_models/CataCamera.cc
  camera_models/src/camera_models/EquidistantCamera.cc
  camera_models/src/camera_models/ScaramuzzaCamera.cc
  camera_models/src/sparse_graph/Transform.cc
  camera_models/src/gpl/gpl.cc
  camera_models/src/gpl/EigenQuaternionParameterization.cc]]
)

## 给定路径以便可以找到自己编译的第三方库 CUDASift 和 plugin_list_CudaSift
## link_directories(/home/crs/GEN_VIO/src/dependencies/CudaSift-Pascal/build)

## 注意，在为库 或者 可执行文件 添加依赖链接库时，如果需要依赖库a和b，而a又同时依赖于b，则链接顺序应该是先a后b，即依赖越多，放得越前！
target_link_libraries(${PROJECT_NAME}_2_lib
  ${PROJECT_NAME}_1_lib
  plugin_list
  ${CAMERA_MODEL}
  ${TENSORRT_LIBRARY}
  ${OpenCV_LIBRARIES}
  #${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES}
  /usr/lib/x86_64-linux-gnu/libffi.so.7
)

target_link_libraries(${PROJECT_NAME}_2_lib cuda cublas cudart cudnn)
target_link_libraries(${PROJECT_NAME}_2_lib pthread)
target_link_libraries(${PROJECT_NAME}_2_lib opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs -lffi)

#O0代表不进行编译的自动优化
target_compile_options(${PROJECT_NAME}_2_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17; -Wall; -O0; -Wfatal-errors; -pthread; -Werror=return-type; -g>)
# 可以对这部分文件开启ASAN监测。compile和link的命令中都要添加-fsanitize=address指令
#target_compile_options(${PROJECT_NAME}_2_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17; -Wall; -fstack-protector-all; -Wstack-protector; -fsanitize=address; -O2; -Wfatal-errors; -pthread; -Werror=return-type; -g>)
#target_link_options(${PROJECT_NAME}_2_lib PRIVATE -fsanitize=address)

##add_executable(${PROJECT_NAME} main.cpp)
##target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib ${catkin_LIBRARIES})

##add_executable(${PROJECT_NAME}_ros ros_main.cpp)
##target_link_libraries(${PROJECT_NAME}_ros ${PROJECT_NAME}_lib ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME} src/KITTITracking.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_2_lib ${PROJECT_NAME}_1_lib plugin_list /usr/lib/x86_64-linux-gnu/libffi.so.7)
target_link_libraries(${PROJECT_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudadevrt.a)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17; -Wall; -O0; -Wfatal-errors; -pthread; -Werror=return-type; -g>)

#target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17; -Wall; -fstack-protector-all; -Wstack-protector; -fsanitize=address; -O2; -Wfatal-errors; -pthread; -Werror=return-type; -g>)
#target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
