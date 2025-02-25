@echo off

if not exist build mkdir build
cmake -S . -B build
cmake --build build
rem if exist trace_radiance.optixir mv trace_radiance.optixir trace_radiance_old.optixir
mv build/lib/ptx/Debug/Target_generated_trace_radiance.cu.optixir trace_radiance.optixir