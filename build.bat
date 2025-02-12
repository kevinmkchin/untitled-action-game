@echo off 

:: BUILD.BAT
::   Invoke MSVC to build project 64-bit build
::   Copies required DLLs to executable directory
::   Sets up BUILDINFO.H


set ShowMSVCOptions=0
set ReleaseMode=0

:ParamCheck :: Loop through all parameters
if [%1] == [] goto EndParamCheck
if /I "%1" == "showoptions" (set ShowMSVCOptions=1)
if /I "%1" == "-s"          (set ShowMSVCOptions=1)
if /I "%1" == "release"     (set ReleaseMode=1)
if /I "%1" == "-r"          (set ReleaseMode=1)
:: Shift the parameters to the left
shift 
goto ParamCheck
:EndParamCheck

:: BUILDINFO.H ::
set CurrentDir=%cd%
set CurrentDir=%CurrentDir:\=/%
:: maybe InternalBuild should be always if not Release mode?
if %ReleaseMode%==0 (
    set IsInternalBuild=1
) else (
    set IsInternalBuild=0
)
echo #pragma once                                        > code\BUILDINFO.H
echo #define PROJECT_WORKING_DIR "%CurrentDir%/wd/"     >> code\BUILDINFO.H
echo #define MESA_WINDOWS 1                             >> code\BUILDINFO.H
echo #define INTERNAL_BUILD %IsInternalBuild%           >> code\BUILDINFO.H

:: MSVC FLAGS ::
set CommonCompilerFlags=-nologo /std:c++17 /EHsc /W3 /we4239 /wd4996 /MP /I"..\ext" /I"..\ext\gl3w" /I"..\ext\sdl\include" /I"..\ext\sdl2_mixer\include" /I"..\ext\assimp\include" /I"..\ext\jolt" /I"..\ext\recast\Includes"
set CommonLinkerFlags=/incremental:no /opt:ref /subsystem:console shell32.lib opengl32.lib dwmapi.lib ole32.lib /LIBPATH:"..\ext\sdl\lib\x64" SDL2.lib SDL2main.lib /LIBPATH:"..\ext\sdl2_mixer\lib\x64" SDL2_mixer.lib /LIBPATH:"..\ext\assimp\lib" assimp-vc142-mt.lib

:: Debug or Release ::
set OutputExecutable=GAME.exe
if %ReleaseMode%==0 (
    set OutputFolder=Debug
    set CommonCompilerFlags=%CommonCompilerFlags% /Zi /MTd /DJPH_OBJECT_STREAM
    set CommonLinkerFlags=%CommonLinkerFlags% /LIBPATH:"..\ext\jolt\Debug" jolt.lib /LIBPATH:"..\ext\recast\Debug" Recast-d.lib Detour-d.lib DebugUtils-d.lib
    echo Build %OutputExecutable% in Debug mode
) else (
    set OutputFolder=Release
    set CommonCompilerFlags=%CommonCompilerFlags% /O2 /DNDEBUG /DJPH_OBJECT_STREAM
    set CommonLinkerFlags=%CommonLinkerFlags% /LIBPATH:"..\ext\jolt\Release" /LTCG jolt.lib /LIBPATH:"..\ext\recast\Release" Recast.lib Detour.lib DebugUtils.lib
    echo Build %OutputExecutable% in Release mode
)

:: COPY DLLS ::
if not exist build mkdir build
if not exist build\%OutputFolder% mkdir build\%OutputFolder%
if not exist build\%OutputFolder%\SDL2.dll copy ext\sdl\lib\x64\SDL2.dll build\%OutputFolder%\SDL2.dll > nul
if not exist build\%OutputFolder%\SDL2_mixer.dll copy ext\sdl2_mixer\lib\x64\SDL2_mixer.dll build\%OutputFolder%\SDL2_mixer.dll >nul
if not exist build\%OutputFolder%\assimp-vc142-mt.dll copy ext\assimp\lib\assimp-vc142-mt.dll build\%OutputFolder%\assimp-vc142-mt.dll >nul
if not exist build\%OutputFolder%\renderdoc.dll copy ext\renderdoc.dll build\%OutputFolder%\renderdoc.dll >nul

:: cl.exe ::
if %ShowMSVCOptions%==1 (
    echo:
    echo Compiler Flags:
    echo %CommonCompilerFlags%
    echo:
    echo Linker Flags: 
    echo %CommonLinkerFlags%
    echo:
)
set BuildStartTime=%time%
if %IsInternalBuild%==0 (echo INTERNAL_BUILD 0) else (echo INTERNAL_BUILD 1) 
echo:
pushd build
cl %CommonCompilerFlags% /Fe%OutputFolder%\%OutputExecutable% ..\code\main.cpp /link %CommonLinkerFlags%
popd
echo:
if %errorlevel% neq 0 (
    exit /b %errorlevel%
) else (
    echo Output to build\%OutputFolder%\%OutputExecutable%
    call timer.cmd %BuildStartTime% %time%
)


