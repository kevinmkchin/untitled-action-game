@echo off

if not exist build mkdir build
echo Build started: %time%

cmake -S . -B build

if "%1" == "clean" goto clean
if "%1" == "release" goto release

goto debug

:clean
cmake --build build --config Debug --target clean
cmake --build build --config Release --target clean
goto end

:debug
cmake --build build --config Debug
goto end

:release
cmake --build build --config Release
goto end

:end
echo Build finished: %time%
