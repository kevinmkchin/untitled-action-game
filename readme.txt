

== Build for Windows ==

BUILD.BAT calls MSVC cl.exe

build           : output to build\Debug\ with debug mode /Zi flag and creates PDB
build release   : output to build\Release\ with full optimizations /O2 flag

There is also a CMakeLists.txt but this might not be kept up to date. There primarily for IDEs.


== Dependencies ==

Jolt - physics https://github.com/jrouwe/JoltPhysics
     - built with JPH_DEBUG_RENDERER only for Debug config
     - otherwise, mostly default configs in their CMakeLists were used
Recast - navigation mesh https://github.com/recastnavigation/recastnavigation
       - built with their default configs without any conflicts
NVIDIA OptiX - RTX ray tracing support for lightmapping
gmath.h - math
stb_ds.h - dynamic arrays

See ext\ for any other external libraries.

Consider https://github.com/rxi/cmixer to replace SDL_mixer
