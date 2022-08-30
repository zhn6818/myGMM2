

add_executable(texture textTrue.cpp)
target_link_libraries(texture ${OpenCV_LIBS})
target_link_libraries(texture cumygmm)


cuda_add_executable(testmallocpich testmallocpich.cu)

cuda_add_executable(malloc3D malloc3D.cu)

# if(GLUT_FOUND)
# add_executable(openGL openGL.cpp)
# target_link_libraries(openGL ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
# endif(GLUT_FOUND)