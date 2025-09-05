cmake_minimum_required(VERSION 3.12.0)
project(code LANGUAGES CXX C)
include (../common/cmake-common.cmake)
add_executable(main-1D src/main-1D.cpp src/BRAIN-1D.cpp)
deal_ii_setup_target(main-1D) 

add_custom_target(remove_vtu_files
        COMMAND ${CMAKE_COMMAND} -E remove
        "${CMAKE_BINARY_DIR}/*.pvtu"
        "${CMAKE_BINARY_DIR}/*.vtu"
        "${CMAKE_BINARY_DIR}/*.vtk"

        COMMENT "Removing all .pvtu and .vtu files from the build directory"
)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
        "${CMAKE_BINARY_DIR}/*.pvtu"
        "${CMAKE_BINARY_DIR}/*.vtu"
        "${CMAKE_BINARY_DIR}/*.vtk"

)

add_dependencies(main-1D remove_vtu_files)