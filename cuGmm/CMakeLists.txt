
cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-O0 -g)
set(CMAKE_CXX_FLAGS "-g  -std=c++11 -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
AUX_SOURCE_DIRECTORY(. cumyGmm)
cuda_add_library(cumygmm ${cumyGmm})
