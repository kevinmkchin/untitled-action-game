

== Build for Windows ==

Requires MSVC

cmake -S . -B build
cmake --build build --config Debug

Build configs: Debug, Release, Distribution
Debug          : output to build\Debug\ with debug mode /Zi flag and creates PDB
Release        : output to build\Release\ with full optimizations /O2 flag
Distribution   : shippable build


== Dependencies ==

Jolt - physics https://github.com/jrouwe/JoltPhysics
Recast - navigation mesh https://github.com/recastnavigation/recastnavigation
       - built with their default configs without any conflicts
NVIDIA OptiX - RTX ray tracing support for lightmapping
gmath.h - math
stb_ds.h - dynamic arrays

See ext\ for any other external libraries.

Consider https://github.com/rxi/cmixer to replace SDL_mixer
