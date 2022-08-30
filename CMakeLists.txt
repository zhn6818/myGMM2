cmake_minimum_required(VERSION 3.19)
project(zhang)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-O0 -g)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_75,code=sm_75
)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-g  -std=c++11 -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")



set(HOST_PATH /data1/gcy/software/3rdparty)
if (EXISTS ${HOST_PATH})
    set(OpenCV_INCLUDE_DIRS /data1/gcy/software/3rdparty/opencv_cuda_400)
    set(OpenCV_LIBS
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_core.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_imgproc.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_ximgproc.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_highgui.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_imgcodecs.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_videoio.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_video.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_dnn.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_cudaimgproc.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_cudaarithm.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_tracking.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_cudawarping.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_optflow.so
    # ${HOST_PATH}/opencv_cuda_400/lib/libopencv_cudaoptflow.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_cudaimgproc.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_calib3d.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_features2d.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_cudacodec.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_photo.so
    ${HOST_PATH}/opencv_cuda_400/lib/libopencv_xfeatures2d.so
    ${HOST_PATH}/ffmpeg/libavcodec.so.58
    ${HOST_PATH}/ffmpeg/libavformat.so.58
    ${HOST_PATH}/ffmpeg/libavutil.so.56
    ${HOST_PATH}/ffmpeg/libswscale.so.5
    ${HOST_PATH}/ffmpeg/libavdevice.so.58 
    ${HOST_PATH}/ffmpeg/libswresample.so.3)
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


set(ONNXRUNTIME_ROOT_PATH /data1/code/_onnxruntime/onnxruntime)
set(TENSORRT_HOME /data1/soft/TensorRT-8.4.2.4/lib)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
        # ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/core/providers/cuda

        )
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)


include_directories(${ONNXRUNTIME_INCLUDE_DIRS})


include_directories("${PROJECT_SOURCE_DIR}/myGmm")
add_subdirectory(myGmm)

add_executable(main main.cpp )
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main mygmm)

include_directories("${PROJECT_SOURCE_DIR}/cuGmm")
add_subdirectory(cuGmm)
add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test cumygmm)

add_subdirectory(testscript)

add_subdirectory(threadPool)

add_subdirectory(ort)


