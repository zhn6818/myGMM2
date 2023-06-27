cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-O0 -g)


add_executable(kmeans kmeans-test.cpp KMeans.cpp KMeans.h)

add_executable(kmeans_main kmeans_main.cpp)

add_executable(EM opencvEM.cpp)
target_link_libraries(EM ${OpenCV_LIBS})