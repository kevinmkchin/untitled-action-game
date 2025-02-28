
Radiant is a ray tracing lightmapping program. It leverages NVIDIA OptiX and RTX hardware to model global illumination and bake lightmaps incredibly fast.

The optix/ directory contains the CMakeLists project to compile the OptiX shader into the OptiX Intermediate Representation. It also contains a python script to generate a C++ byte array from the optixir binary so that it can be embedded into the program instead of loading at runtime.

Requires CUDA Toolkit

I'm not sure if OptiX SDK needs to be installed or if I've copied all the required dependencies to this folder.  

== How to use ==

cmake -S . -B build
cmake --build build --config Release

Include radiant.h in your project
Load the output dll in your project
