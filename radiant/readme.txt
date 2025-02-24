
CMakeLists.txt is for compiling cuda files to optixir.
Taken from OptiX 8.1 SDK and modified to work. CUDA_HOST_COMPILER was not resolving Visual Studio properties like $(VCInstallDir) so I had to set it manually.
