cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-O0 -g)


add_executable(ort main.cpp Onnx.h Onnx.cpp UTIL.h UTIL.cpp)
target_link_libraries(ort ${ONNXRUNTIME_LIB})
target_link_libraries(ort ${OpenCV_LIBS})