cmake_minimum_required(VERSION 3.12.0)
project(code LANGUAGES CXX C)
include (../common/cmake-common.cmake)
add_executable(main src/main.cpp src/BRAIN.cpp)
deal_ii_setup_target(main) 

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

add_dependencies(main remove_vtu_files)