cmake_minimum_required(VERSION 3.19)
project(zhang)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-O0 -g)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O0
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
    -gencode arch=compute_70,code=sm_70
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_80,code=sm_80
    -gencode arch=compute_86,code=sm_86
)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-g  -std=c++11 -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")



set(HOST_PATH /data1/virgo_3rdparty)
if (EXISTS ${HOST_PATH})
    set(OpenCV_INCLUDE_DIRS 
    ${HOST_PATH}/opencv/include
    ${HOST_PATH}/opencv/include/opencv4
    ${HOST_PATH}/opencv/include/opencv4/opencv2)
    set(OpenCV_LIBS
    ${HOST_PATH}/opencv/lib/libopencv_core.so
    ${HOST_PATH}/opencv/lib/libopencv_imgproc.so
    ${HOST_PATH}/opencv/lib/libopencv_ximgproc.so
    ${HOST_PATH}/opencv/lib/libopencv_highgui.so
    ${HOST_PATH}/opencv/lib/libopencv_imgcodecs.so
    ${HOST_PATH}/opencv/lib/libopencv_cudafilters.so
    ${HOST_PATH}/opencv/lib/libopencv_videoio.so
    ${HOST_PATH}/opencv/lib/libopencv_video.so
    ${HOST_PATH}/opencv/lib/libopencv_dnn.so
    ${HOST_PATH}/opencv/lib/libopencv_cudaimgproc.so
    ${HOST_PATH}/opencv/lib/libopencv_cudaarithm.so
    ${HOST_PATH}/opencv/lib/libopencv_tracking.so
    ${HOST_PATH}/opencv/lib/libopencv_cudawarping.so
    ${HOST_PATH}/opencv/lib/libopencv_optflow.so
    # ${HOST_PATH}/opencv/lib/libopencv_cudaoptflow.so
    ${HOST_PATH}/opencv/lib/libopencv_cudaimgproc.so
    ${HOST_PATH}/opencv/lib/libopencv_calib3d.so
    ${HOST_PATH}/opencv/lib/libopencv_features2d.so
    ${HOST_PATH}/opencv/lib/libopencv_cudacodec.so
    ${HOST_PATH}/opencv/lib/libopencv_photo.so
    ${HOST_PATH}/opencv/lib/libopencv_xfeatures2d.so
)
ELSE()
    find_package(OpenCV REQUIRED)
ENDIF()

# openGL安装教程 https://blog.csdn.net/chengde6896383/article/details/88028826
# find_package获得opengl的头文件和library.我们得到四个变量OPENGL_INCLUDE_DIR, GLUT_INCLUDE_DIR, OPENGL_LIBRARIES, GLUT_LIBRARIES.

# https://blog.argcv.com/articles/3853.c
# find_package(OpenGL REQUIRED)
# find_package(GLUT REQUIRED)

# if(NOT GLUT_FOUND)
#     message(ERROR " GLUT not found!")
# endif(NOT GLUT_FOUND)

# if(GLUT_FOUND)
#     include_directories(${GLUT_INCLUDE_DIR})
# endif(GLUT_FOUND)


include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda/include
    # ${OPENGL_INCLUDE_DIR}
    
    )


set(ONNXRUNTIME_ROOT_PATH ${HOST_PATH}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
    ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
    ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
    )
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)


include_directories(${ONNXRUNTIME_INCLUDE_DIRS})


include_directories("${PROJECT_SOURCE_DIR}/myGmm")
add_subdirectory(myGmm)

add_executable(main main.cpp )
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main mygmm)
target_link_libraries(main cumygmm)


include_directories("${PROJECT_SOURCE_DIR}/cuGmm")
add_subdirectory(cuGmm)
add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test cumygmm)
target_link_libraries(test mygmm)

# add_subdirectory(testscript)

add_subdirectory(threadPool)

add_subdirectory(ort)

add_subdirectory(kmeans_gmm)

cuda_add_executable(meansquare meansquare.cpp)
target_link_libraries(meansquare ${OpenCV_LIBS})



