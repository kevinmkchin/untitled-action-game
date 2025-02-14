#
# CMakeLists.txt equivalent of build.bat
# Exists for compatibility with IDEs like CLion
# This project only compiles with MSVC!!!
# Must set both toolchain and generator in CLion to Visual Studio
# This may not be kept up to date with build.bat
#

cmake_minimum_required(VERSION 3.20)

project(game)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   message(FATAL_ERROR "Requires MSVC")
endif()

set (INTERNAL_BUILD 1)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
set (INTERNAL_BUILD 0)
endif()

add_executable(${PROJECT_NAME} code/main.cpp)

file(WRITE  "code/BUILDINFO.H" "#pragma once\n")
file(APPEND "code/BUILDINFO.H" "#define PROJECT_WORKING_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/wd/\"\n")
file(APPEND "code/BUILDINFO.H" "#define MESA_WINDOWS 1\n")
file(APPEND "code/BUILDINFO.H" "#define INTERNAL_BUILD ${INTERNAL_BUILD}\n")

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/ext
        ${CMAKE_SOURCE_DIR}/ext/gl3w
        ${CMAKE_SOURCE_DIR}/ext/sdl/include
        ${CMAKE_SOURCE_DIR}/ext/sdl2_mixer/include
        ${CMAKE_SOURCE_DIR}/ext/assimp/include
        ${CMAKE_SOURCE_DIR}/ext/jolt
        ${CMAKE_SOURCE_DIR}/ext/recast/Includes
)

set(MSVC_COMPILE_OPTIONS -nologo /std:c++17 /EHsc /W3 /we4239 /wd4996 /MP)
set(MSVC_COMPILE_DEBUG_OPTIONS ${MSVC_COMPILE_OPTIONS} /Zi /MTd /DJPH_OBJECT_STREAM)
set(MSVC_COMPILE_RELEASE_OPTIONS ${MSVC_COMPILE_OPTIONS} /O2 /DNDEBUG /DJPH_OBJECT_STREAM)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:${MSVC_COMPILE_DEBUG_OPTIONS}>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:${MSVC_COMPILE_RELEASE_OPTIONS}>)

set(MSVC_LINKER_OPTIONS
        /incremental:no /opt:ref /subsystem:console shell32.lib opengl32.lib dwmapi.lib ole32.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/sdl/lib/x64" SDL2.lib SDL2main.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/sdl2_mixer/lib/x64" SDL2_mixer.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/assimp/lib" assimp-vc142-mt.lib)
set(MSVC_LINKER_DEBUG_OPTIONS ${MSVC_LINKER_OPTIONS}
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/jolt/Debug" jolt.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/recast/Debug" Recast-d.lib Detour-d.lib DebugUtils-d.lib)
set(MSVC_LINKER_RELEASE_OPTIONS ${MSVC_LINKER_OPTIONS}
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/jolt/Release" /LTCG jolt.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/recast/Release" Recast.lib Detour.lib DebugUtils.lib)
target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:${MSVC_LINKER_DEBUG_OPTIONS}>)
target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:${MSVC_LINKER_RELEASE_OPTIONS}>)

# Copy DLLs after build
set(DLL_FILES
        ${CMAKE_SOURCE_DIR}/ext/sdl/lib/x64/SDL2.dll
        ${CMAKE_SOURCE_DIR}/ext/sdl2_mixer/lib/x64/SDL2_mixer.dll
        ${CMAKE_SOURCE_DIR}/ext/assimp/lib/assimp-vc142-mt.dll
        ${CMAKE_SOURCE_DIR}/ext/renderdoc.dll
)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLL_FILES}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying DLLs to output directory"
)
