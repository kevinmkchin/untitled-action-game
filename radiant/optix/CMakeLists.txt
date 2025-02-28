# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on VERSION_GREATER_EQUAL which requires CMake v. 3.7.
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

# This sets up the name of our project.  For our purposes the main thing this controls is
# the name of the VS solution file.
project(OptiX-Samples)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# As of CMake 2.6 policies were introduced in order to provide a mechanism for
# adding backwards compatibility one feature at a time.  We will just specify
# that all policies will use version 2.8.12 semantics.
cmake_policy(VERSION 2.8.12)

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

##########
# Process our custom setup scripts here.

# Enable C++11. Needs to be done before the include of ConfigCompilerFlags.cmake below.
set(GCC_LIBSTDCPP11 ON)

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Passing the --use-local-env option to NVCC can dramatically speed up CUDA compilation
if(WIN32)
  option(CUDA_USE_LOCAL_ENV "Pass the '--use-local-env' option to NVCC; only applies to initial configure" ON)
endif()

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# If NVRTC was enabled/disabled, reset OPTIXIR/PTX to default
set( CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile OPTIXIR/PTX at run-time instead of NVCC at build-time" )
if( NOT ( NOT ( NOT CUDA_NVRTC_ENABLED ) ) EQUAL ( NOT ( NOT CUDA_NVRTC_ENABLED_INTERNAL ) ) )
  message( STATUS "Resetting OPTIXIR/PTX support" )
  unset( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT CACHE )
endif()
set( CUDA_NVRTC_ENABLED_INTERNAL ${CUDA_NVRTC_ENABLED} CACHE INTERNAL "Previous configured value (NVRTC)" FORCE )

if( CUDA_NVRTC_ENABLED )
  if( CUDA_VERSION VERSION_LESS 12.0 )
    if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
      message( SEND_ERROR "CUDA_NVRTC_ENABLED is not compatible with SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT in CUDA versions less than 12.0" )
    else()
      option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
    endif()
  else()
    option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
  endif()
else()
  if( CUDA_VERSION VERSION_LESS 11.7 )
    if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
      message( SEND_ERROR "SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7" )
    else()
      option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
    endif()
  else()
    option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
  endif()
endif()

# This code looks funny, but CMake doesn't have an equality operator for boolean types
# (only integer and string). By doing NOT NOT VAL, you can force the original value into 0
# or 1 and allow the EQUAL operator to function correctly.
if( NOT ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
  message( STATUS "Unsetting values associated with OptiX code generation" )
  # This allows us to reset dependent options if you change it.
  unset( SAMPLES_INPUT_GENERATE_OPTIXIR CACHE )
  unset( SAMPLES_INPUT_GENERATE_PTX CACHE )
  unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value (OPTIXIR)" FORCE)

if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

# Determine if we are going to use the static CRT on windows.
if(WIN32)
  option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" ON)
endif()

# Helper for fixing compiler flags
function(replace_flag var old_flag new_flag)
  string(REPLACE "${old_flag}" "${new_flag}" ${var} ${${var}})
  set(${var} "${${var}}" CACHE STRING "Default compiler flags" FORCE)
endfunction()

function(replace_flags old_flag new_flag)
  foreach(build "" _DEBUG _MINSIZEREL _RELEASE _RELWITHDEBINFO)
    replace_flag(CMAKE_C_FLAGS${build}   "${old_flag}" "${new_flag}")
    replace_flag(CMAKE_CXX_FLAGS${build} "${old_flag}" "${new_flag}")
  endforeach()
endfunction()

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endfunction()

function(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config( "" ${flag} )
endfunction()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")

  # CMAKE_CONFIGURATION_TYPES is only defined for multi-config build systems like
  # MSVC and Ninja, but we need to generate flags for each configuration
  # regardless.
  if( DEFINED CMAKE_CONFIGURATION_TYPES )
    set( OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} )
  else()
    set( OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
  endif()

  foreach( config ${OPTIX_CONFIGURATION_TYPES} )
    if( ${config} STREQUAL "Debug" )
      optix_add_cuda_flag_config( _${config} "-G" )
      optix_add_cuda_flag_config( _${config} "-O0" )
    endif()
  endforeach()

  if( CUDA_VERSION VERSION_LESS "3.0" )
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CUDA_USE_LOCAL_ENV)
    optix_add_cuda_flag("--use-local-env")
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()

  if( NOT DEFINED CMAKE_CONFIGURATION_TYPES )
    if( NOT CMAKE_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE_PREVIOUS )
      message( STATUS "Resetting CUDA_NVRTC_FLAGS" )
      unset( CUDA_NVRTC_FLAGS CACHE )
    endif()
    set( CMAKE_BUILD_TYPE_PREVIOUS ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Previous configured value (CMAKE_BUILD_TYPE)" FORCE )

    set( configs "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
    foreach( config ${configs} )
      if( ${config} STREQUAL "Debug" )
        set( SAMPLES_NVRTC_DEBUG "-G" )
      else()
        set( SAMPLES_NVRTC_DEBUG "-lineinfo" )
      endif()

      string( TOUPPER ${config} config_upper )
      set( CUDA_NVRTC_FLAGS_${config_upper} ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} ${SAMPLES_NVRTC_DEBUG} -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )

      if( ${config} STREQUAL ${CMAKE_BUILD_TYPE} )
        set( CUDA_NVRTC_FLAGS ${CUDA_NVRTC_FLAGS_${config_upper}} CACHE STRING "List of NVRTC options just for the samples" )
      endif()
    endforeach()
  else()
    set( CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -lineinfo -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
    set( CUDA_NVRTC_FLAGS_DEBUG ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -G -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
  endif()
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# # Search for the OptiX libraries and include files.
# find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  # "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

# my shit

set(OptixModules ../lm_bake.cu)

set_source_files_properties(${OptixModules} PROPERTIES LANGUAGE CUDA)
get_filename_component(CXX_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
set(CUDA_HOST_COMPILER ${CXX_COMPILER_DIR})

set( target_name Target )

# Separate the sources from the CMake and CUDA options fed to the macro.  This code
# comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
# code here, so that we can use our own name for the target.  target_name is used in the
# creation of the output file names, and we want this to be unique for each target in
# the SDK.
CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options 
  ${OptixModules}
)

# Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
set(cu_obj_source_files)
set(cu_optix_source_files)
foreach(file ${source_files})
  get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
  if(${_cuda_source_format} MATCHES "OBJ")
    list(APPEND cu_obj_source_files ${file})
  else()
    list(APPEND cu_optix_source_files ${file})
  endif()
endforeach()

# Create the rules to build the OBJ from the CUDA files.
CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )

# Create the rules to build the PTX and/or OPTIX files.
if( SAMPLES_INPUT_GENERATE_OPTIXIR )
  CUDA_WRAP_SRCS( ${target_name} OPTIXIR generated_files2 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
  list(APPEND generated_files ${generated_files2})
endif()
if( SAMPLES_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
  CUDA_WRAP_SRCS( ${target_name} PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
  list(APPEND generated_files ${generated_files3})
endif()

# Here is where we create the rule to make the executable.  We define a target name and
# list all the source files used to create the target.  In addition we also pass along
# the cmake_options parsed out of the arguments.
add_executable(${target_name}
  ${source_files}
  ${generated_files}
  ${cmake_options}
)

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)

find_package(Python3 REQUIRED)

add_custom_command(
    TARGET ${target_name}
    POST_BUILD
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bin_to_cpp.py ${generated_files} ${CMAKE_SOURCE_DIR}/../optixir_binary.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python script after building MakeOptixIRProgram"
)



