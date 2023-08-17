cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-O0 -g)

add_executable(testFaiss testFaiss.cpp)
target_link_libraries(testFaiss ${FAISS_LIB})

add_executable(Flat_test Flat_test.cpp)
target_link_libraries(Flat_test ${FAISS_LIB})

add_executable(faissST faissST.cpp)
target_link_libraries(faissST ${FAISS_LIB})