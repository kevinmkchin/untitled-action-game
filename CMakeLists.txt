#
# This project only compiles with MSVC!!!
# Must set both toolchain and generator in CLion or Visual Studio to Visual Studio
#

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Distribution" CACHE STRING "" FORCE)

project(game)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
   message(FATAL_ERROR "Requires MSVC")
endif()

add_subdirectory(jolt/Build)
add_subdirectory(recast/Recast)
add_subdirectory(recast/Detour)
add_subdirectory(recast/DebugUtils)
add_subdirectory(recast/DetourTileCache)

add_executable(${PROJECT_NAME}
        # should exist
        code/mem.cpp
        code/common.cpp
        code/filedialog.cpp
        code/utility.cpp
        code/shaders.cpp
        code/instanced.cpp

        code/renderer.cpp
        # could refactor
        code/main.cpp
        code/gui.cpp
        code/cam.cpp
        code/anim.cpp
        code/gpu_resources.cpp
        code/resources.cpp
        code/game_assets.cpp
        code/physics.cpp
        code/physics_debug.cpp
        code/facebatch.cpp
        code/shader_helpers.cpp
        code/winged.cpp
        code/primitives.cpp
        code/levelentities.cpp
        code/particles.cpp
        code/lightmap.cpp
        code/saveloadlevel.cpp
)
target_precompile_headers(${PROJECT_NAME} PRIVATE code/pch.h)
target_link_libraries(${PROJECT_NAME} PRIVATE Jolt Recast Detour DebugUtils)

file(WRITE  "code/BUILDINFO.H" "#pragma once\n")
file(APPEND "code/BUILDINFO.H" "#define PROJECT_WORKING_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/wd/\"\n")
file(APPEND "code/BUILDINFO.H" "#define MESA_WINDOWS 1\n")
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:INTERNAL_BUILD=1>
    $<$<NOT:$<CONFIG:Debug>>:INTERNAL_BUILD=0>
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/ext
        ${CMAKE_SOURCE_DIR}/ext/gl3w
        ${CMAKE_SOURCE_DIR}/ext/SDL3/include
        ${CMAKE_SOURCE_DIR}/ext/SDL3_mixer/include
        ${CMAKE_SOURCE_DIR}/ext/assimp/include
        ${CMAKE_SOURCE_DIR}/jolt
        ${CMAKE_SOURCE_DIR}/recast/Recast/Include
        ${CMAKE_SOURCE_DIR}/recast/Detour/Include
        ${CMAKE_SOURCE_DIR}/recast/DebugUtils/Include
        ${CMAKE_SOURCE_DIR}/recast/DetourTileCache/Include
        ${CMAKE_SOURCE_DIR}/radiant
)

set(MSVC_COMPILE_OPTIONS -nologo /std:c++17 /EHsc /W3 /we4239 /wd4996 /MP)
set(MSVC_COMPILE_DEBUG_OPTIONS ${MSVC_COMPILE_OPTIONS} /ZI /MTd)
set(MSVC_COMPILE_RELEASE_OPTIONS ${MSVC_COMPILE_OPTIONS} /O2 /MT /DNDEBUG)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:${MSVC_COMPILE_DEBUG_OPTIONS}>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:${MSVC_COMPILE_RELEASE_OPTIONS}>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Distribution>:${MSVC_COMPILE_RELEASE_OPTIONS}>)

set(MSVC_LINKER_OPTIONS
        /subsystem:console shell32.lib opengl32.lib dwmapi.lib ole32.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/assimp/lib" assimp-vc142-mt.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/radiant/Release" radiant.lib
)
set(MSVC_LINKER_DEBUG_OPTIONS ${MSVC_LINKER_OPTIONS}
        /debug:full /opt:noicf
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/SDL3/Debug" SDL3.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/SDL3_mixer/Debug" SDL3_mixer.lib
)
set(MSVC_LINKER_RELEASE_OPTIONS ${MSVC_LINKER_OPTIONS}
        /incremental:no /opt:ref /LTCG
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/SDL3/Release" SDL3.lib
        /LIBPATH:"${CMAKE_SOURCE_DIR}/ext/SDL3_mixer/Release" SDL3_mixer.lib
)
target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:${MSVC_LINKER_DEBUG_OPTIONS}>)
target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:${MSVC_LINKER_RELEASE_OPTIONS}>)
target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Distribution>:${MSVC_LINKER_RELEASE_OPTIONS}>)

# Copy DLLs after build
set(DLL_FILES
        ${CMAKE_SOURCE_DIR}/ext/SDL3/Release/SDL3.dll
        ${CMAKE_SOURCE_DIR}/ext/SDL3_mixer/Release/SDL3_mixer.dll
        ${CMAKE_SOURCE_DIR}/ext/assimp/lib/assimp-vc142-mt.dll
        ${CMAKE_SOURCE_DIR}/ext/renderdoc.dll
        ${CMAKE_SOURCE_DIR}/radiant/Release/radiant.dll
)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${DLL_FILES}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying DLLs to output directory"
)
