file(GLOB SOURCES Source/*.cpp)
add_library(Recast ${SOURCES})

add_library(RecastNavigation::Recast ALIAS Recast)
set_target_properties(Recast PROPERTIES DEBUG_POSTFIX -d)

set(Recast_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Include")

target_include_directories(Recast PUBLIC
    "$<BUILD_INTERFACE:${Recast_INCLUDE_DIR}>"
)

# 2025-03-27 NOTE(Kevin): required to avoid mismatch between runtime library flag
target_compile_options(Recast PRIVATE
    $<$<CONFIG:Debug>:/MTd>
    $<$<CONFIG:Release>:/MT>
    $<$<CONFIG:Distribution>:/MT>
)

set_target_properties(Recast PROPERTIES
        SOVERSION ${SOVERSION}
        VERSION ${LIB_VERSION}
        COMPILE_PDB_OUTPUT_DIRECTORY .
        COMPILE_PDB_NAME "Recast-d"
        )

install(TARGETS Recast
        EXPORT recastnavigation-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT library
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} ${CMAKE_INSTALL_INCLUDEDIR}/recastnavigation
        )

file(GLOB INCLUDES Include/*.h)
install(FILES ${INCLUDES} DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/recastnavigation)
if(MSVC)
    install(FILES "$<TARGET_FILE_DIR:Recast>/Recast-d.pdb" CONFIGURATIONS "Debug" DESTINATION "lib" OPTIONAL)
endif()
