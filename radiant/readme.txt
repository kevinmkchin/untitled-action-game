
Radiant is a ray tracing lightmapper program. It leverages NVIDIA OptiX and RTX hardware to model global illumination and bake lightmaps incredibly fast.

The optix/ directory contains the CMakeLists project to compile the OptiX shader into the OptiX Intermediate Representation. It also contains a python script to generate a C++ byte array from the optixir binary so that it can be embedded into the program instead of loading at runtime.

Requires CUDA Toolkit

I'm not sure if OptiX SDK needs to be installed or if I've copied all the required dependencies to this folder.  

Radiant should be compiled as a shared library so that any client code does not also require the same CUDA and OptiX dependencies.

== How to use ==

cmake -S . -B build
cmake --build build --config Release

Include radiant.h in your project
Link with the .lib dll import library
Place the .dll next to your executable

If CUDA Toolkit is not installed on the target machine, the executable will probably need cudart64_12.dll as well. The NVIDIA driver also needs to support the OptiX version (8.1) that this library was built with, so update.



