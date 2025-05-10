# Untitled FPS Game (work in progress)

This is a quick feature overview of my custom engine and untitled game.

TODO show gameplay gif here

### Lightmapping
![image](https://github.com/user-attachments/assets/0d52e813-ab61-477a-99b6-8bc2fbc5a414)
![image](https://github.com/user-attachments/assets/d286b2a2-fcb6-44e6-bdaf-bc3b4e78a584)
![image](https://github.com/user-attachments/assets/f550ef33-4c04-4dcb-a987-880d16768d00)

### Light Probes
![image](https://github.com/user-attachments/assets/de486d90-5e64-487b-9104-b7f41ed12ff6)
TODO show model with and without light probes

### Level Editor
[Watch Demo on YouTube](https://www.youtube.com/watch?v=EjHV1p95SDo&ab_channel=KevinChin)
![image](https://github.com/user-attachments/assets/ed32a890-59f8-4205-bd0f-1c129a93e864)

### Particle System
TODO show gib gif here

### Skeletal Animation
![skeletal-animation](https://github.com/user-attachments/assets/f406453c-8adc-49ff-b4cc-332da241c0c4)

### Instanced model rendering
![image](https://github.com/user-attachments/assets/7be8d9d5-5b9f-4051-9f92-0b07f3c52529)

### How to Build
```
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
     - precompiled, but should be easy to include it as a subproject
Jolt - physics https://github.com/jrouwe/JoltPhysics
Recast - navigation mesh https://github.com/recastnavigation/recastnavigation
NVIDIA OptiX - RTX ray tracing support for lightmapping

See ext\ for any other external libraries.
```


