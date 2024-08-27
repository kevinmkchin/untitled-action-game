cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 14)

project(game)
set(PROJECT_WORKING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/wd/")

if (WIN32)
    set(MESA_WINDOWS 1)
    set(MESA_MACOSX 0)
else(APPLE)
    set(MESA_WINDOWS 0)
    set(MESA_MACOSX 1)
endif()

set(INTERNAL_BUILD 1)

set(SOURCE_FILES
    code/GAME.CPP
)


add_executable(${PROJECT_NAME} ${SOURCE_FILES})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ext/BUILDINFO.H.IN" "${CMAKE_CURRENT_SOURCE_DIR}/code/BUILDINFO.H")

# Build options, additional include directories, library linking, copying DLLs to executable directory

if(INTERNAL_BUILD)
    message(STATUS "INTERNAL build: " ${PROJECT_BUILD_VERSION})
else()
    message(STATUS "External build: " ${PROJECT_BUILD_VERSION})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ext)

## Windows
if(WIN32)

    if(MSVC)
      add_definitions(/MP -D_CRT_SECURE_NO_WARNINGS)
      target_compile_options(${PROJECT_NAME} PUBLIC "/W3" "/EHsc" "/we4239")
    endif()

    # Find OpenGL
    find_package(OpenGL REQUIRED)
    if (OPENGL_FOUND)
       target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIR})
       target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_gl_LIBRARY})
    endif()

    set(SDL2_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl/include")
    set(SDL2_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl/lib/x64/SDL2.lib;${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl/lib/x64/SDL2main.lib")
    set(SDL_DLL "${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl/lib/x64/SDL2.dll")

    set(SDL2MIXER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl2_mixer/include")
    set(SDL2MIXER_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl2_mixer/lib/x64/SDL2_mixer.lib")
    set(SDL2MIXER_DLL "${CMAKE_CURRENT_SOURCE_DIR}/ext/sdl2_mixer/lib/x64/SDL2_mixer.dll")

    set(ASSIMP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ext/assimp/include;")
    set(ASSIMP_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/ext/assimp/lib/assimp-vc142-mt.lib")
    set(ASSIMP_DLL "${CMAKE_CURRENT_SOURCE_DIR}/ext/assimp/lib/assimp-vc142-mt.dll")

    # Copy and rename dlls
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL_DLL}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/SDL2.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2MIXER_DLL}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/SDL2_mixer.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ASSIMP_DLL}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assimp-vc142-mt.dll")

    target_include_directories(${PROJECT_NAME} 
        PUBLIC 
        ${SDL2_INCLUDE_DIRS}
        ${SDL2MIXER_INCLUDE_DIRS}
        ext/gl3w 
        ${ASSIMP_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} 
        PUBLIC 
        ${SDL2_LIBRARIES} 
        ${SDL2MIXER_LIBRARIES}
        ${ASSIMP_LIBRARIES} 
        dwmapi.lib)

## MacOS
else(APPLE)

    find_package(SDL2 REQUIRED)
    if(SDL2_FOUND)
        message(STATUS "SDL2 found.")
        include_directories(${SDL2_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})
    else()
    	message(FATAL_ERROR "CMake couldn't find SDL2 on MacOSX.")
    endif()

    find_package(OpenGL REQUIRED)
    if(OpenGL_FOUND)
        message(STATUS "OpenGL found.")
        target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})
    else()
        message(FATAL_ERROR "CMake couldn't find OpenGL on MacOSX.")
    endif()

endif()