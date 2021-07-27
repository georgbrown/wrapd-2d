add_executable( param ${CMAKE_CURRENT_SOURCE_DIR}/param.cpp )
target_link_libraries( param wrapd ${LINKED_LIBS} )