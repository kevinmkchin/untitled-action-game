@echo off 

:: BUILD.BAT
::   Invoke MSVC to build project 64-bit build
::   Copies required DLLs to executable directory
::   Sets up BUILDINFO.H


:: BUILDINFO.H ::
set IsInternalBuild=1 
set CurrentDir=%cd%
set CurrentDir=%CurrentDir:\=/%
echo #pragma once                                        > code\BUILDINFO.H
echo #define PROJECT_WORKING_DIR "%CurrentDir%/wd/"     >> code\BUILDINFO.H
echo #define MESA_WINDOWS 1                             >> code\BUILDINFO.H
echo #define INTERNAL_BUILD %IsInternalBuild%           >> code\BUILDINFO.H

:: MSVC FLAGS ::
set CommonCompilerFlags=-nologo /EHsc /W3 /we4239 /wd4996 /MP /I"..\ext" /I"..\ext\gl3w" /I"..\ext\sdl\include" /I"..\ext\sdl2_mixer\include" /I"..\ext\assimp\include"
set CommonLinkerFlags=/incremental:no /opt:ref /subsystem:console shell32.lib opengl32.lib dwmapi.lib ole32.lib /LIBPATH:"..\ext\sdl\lib\x64" SDL2.lib SDL2main.lib /LIBPATH:"..\ext\sdl2_mixer\lib\x64" SDL2_mixer.lib /LIBPATH:"..\ext\assimp\lib" assimp-vc142-mt.lib

:: Debug or Release
set OutputExecutable=GAME.exe
if "%1"=="release" (
    set OutputFolder=Release
    set CommonCompilerFlags=%CommonCompilerFlags% /O2 /DNDEBUG
    echo Build %OutputExecutable% in Release mode
) else (
    set OutputFolder=Debug
    set CommonCompilerFlags=%CommonCompilerFlags% /Zi
    echo Build %OutputExecutable% in Debug mode
)

:: COPY DLLS ::
if not exist build mkdir build
if not exist build\%OutputFolder% mkdir build\%OutputFolder%
if not exist build\%OutputFolder%\SDL2.dll copy ext\sdl\lib\x64\SDL2.dll build\%OutputFolder%\SDL2.dll > nul
if not exist build\%OutputFolder%\SDL2_mixer.dll copy ext\sdl2_mixer\lib\x64\SDL2_mixer.dll build\%OutputFolder%\SDL2_mixer.dll >nul
if not exist build\%OutputFolder%\assimp-vc142-mt.dll copy ext\assimp\lib\assimp-vc142-mt.dll build\%OutputFolder%\assimp-vc142-mt.dll >nul
if not exist build\%OutputFolder%\renderdoc.dll copy ext\renderdoc.dll build\%OutputFolder%\renderdoc.dll >nul

:: CL ::
:: echo Compiler Flags: %CommonCompilerFlags%
:: echo Linker Flags: %CommonLinkerFlags%
echo Build started: %time%
if %IsInternalBuild%==0 (echo INTERNAL_BUILD 0) else (echo INTERNAL_BUILD 1) 
pushd build
cl %CommonCompilerFlags% /Fe%OutputFolder%\%OutputExecutable% ..\code\GAME.CPP /link %CommonLinkerFlags%
popd
if %errorlevel% neq 0 (
    exit /b %errorlevel%
) else (
    echo Output to build\%OutputFolder%\%OutputExecutable%
    echo Build finished: %time%
)


