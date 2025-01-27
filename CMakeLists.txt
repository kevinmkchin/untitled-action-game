cmake_minimum_required(VERSION 3.20)


project(game)


set(PROJECT_WORKING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/wd/")
set(MESA_WINDOWS 1)
set(INTERNAL_BUILD 1)

add_executable(${PROJECT_NAME} code/main.cpp)




#configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ext/BUILDINFO.H.IN" "${CMAKE_CURRENT_SOURCE_DIR}/code/BUILDINFO.H")
##pragma once
#cmakedefine PROJECT_WORKING_DIR "@PROJECT_WORKING_DIR@"
#cmakedefine MESA_WINDOWS @MESA_WINDOWS@
#cmakedefine MESA_MACOSX @MESA_MACOSX@
#cmakedefine INTERNAL_BUILD @INTERNAL_BUILD@


target_include_directories(${PROJECT_NAME} PUBLIC ext)

add_definitions(/MP -D_CRT_SECURE_NO_WARNINGS)
target_compile_options(${PROJECT_NAME} PUBLIC "/W3" "/EHsc" "/we4239")

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

# # Copy and rename dlls
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${SDL_DLL}"
#     "$<TARGET_FILE_DIR:${PROJECT_NAME}>/SDL2.dll")
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${SDL2MIXER_DLL}"
#     "$<TARGET_FILE_DIR:${PROJECT_NAME}>/SDL2_mixer.dll")
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${ASSIMP_DLL}"
#     "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assimp-vc142-mt.dll")
# 
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${CMAKE_CURRENT_SOURCE_DIR}/ext/renderdoc.dll"
#     "$<TARGET_FILE_DIR:${PROJECT_NAME}>/renderdoc.dll")
# 
# target_include_directories(${PROJECT_NAME} 
#     PUBLIC 
#     ${SDL2_INCLUDE_DIRS}
#     ${SDL2MIXER_INCLUDE_DIRS}
#     ext/gl3w 
#     ${ASSIMP_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} 
#     PUBLIC 
#     ${SDL2_LIBRARIES} 
#     ${SDL2MIXER_LIBRARIES}
#     ${ASSIMP_LIBRARIES} 
#     dwmapi.lib)

