

add_executable(thread test.cpp threadpool.h)
target_link_libraries(thread ${OpenCV_LIBS}  pthread)

add_executable(thread2 test2.cpp threadpool.h)
target_link_libraries(thread2 ${OpenCV_LIBS}  pthread)

add_executable(mutextest mutextest.cpp)
target_link_libraries(mutextest ${OpenCV_LIBS}  pthread)