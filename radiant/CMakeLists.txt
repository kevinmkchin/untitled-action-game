cmake_minimum_required(VERSION 3.20)

project(Radiant)

add_library(Radiant SHARED radiant.cpp radiant.h)

set_target_properties(Radiant PROPERTIES OUTPUT_NAME "radiant")
target_include_directories(Radiant PUBLIC ${CMAKE_SOURCE_DIR}/optix/include)

# CUDA dependency for NVIDIA OptiX

find_package(CUDAToolkit 5.0 REQUIRED)
target_include_directories(
        ${PROJECT_NAME} PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
)
message(${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        CUDA::cudart
)

# Build the Optix IR shader

include(ExternalProject)

ExternalProject_Add(MakeOptixIRProgram
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/optix
    BINARY_DIR ${CMAKE_BINARY_DIR}/optix
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_ARGS 
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
)

# Ensure the external project is built before Radiant
add_dependencies(Radiant MakeOptixIRProgram)