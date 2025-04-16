

== Build for Windows ==

Requires Visual Studio 2022 with the following workload and components:
  - Desktop development with C++:
    - MSVC v143 - VS 2022 C++ x64/x86 build tools
    - Windows 11 SDK (10.0.22621.0)
    - C++ CMake tools for Windows

cmake -S . -B build
cmake --build build --config Debug

Build configs: Debug, Release, Distribution
Debug          : output to build\Debug\ with debug mode /Zi flag and creates PDB
Release        : output to build\Release\ with full optimizations /O2 flag
Distribution   : shippable build

== Dependencies ==

SDL3 - platform https://github.com/libsdl-org/SDL
Jolt - physics https://github.com/jrouwe/JoltPhysics
Recast - navigation mesh https://github.com/recastnavigation/recastnavigation
NVIDIA OptiX - RTX ray tracing support for lightmapping

See ext\ for any other external libraries.
