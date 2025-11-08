@echo off 

set DebugConfig=0
set ReleaseConfig=0
set DistributionConfig=0

if [%1] == [] set DebugConfig=1

:ParamCheck :: Loop through all parameters
if [%1] == [] goto EndParamCheck
if /I "%1" == "Debug"           (set DebugConfig=1)
if /I "%1" == "debug"           (set DebugConfig=1)
if /I "%1" == "Release"         (set ReleaseConfig=1)
if /I "%1" == "release"         (set ReleaseConfig=1)
if /I "%1" == "Distribution"    (set DistributionConfig=1)
if /I "%1" == "distribution"    (set DistributionConfig=1)
:: Shift the parameters to the left
shift 
goto ParamCheck
:EndParamCheck


set BuildStartTime=%time%

rem cmake -S . -B build
if %DebugConfig%==1 cmake --build build --config Debug
if %ReleaseConfig%==1 cmake --build build --config Release
if %DistributionConfig%==1 cmake --build build --config Distribution

if %errorlevel% neq 0 (
    exit /b %errorlevel%
) else (
    call timer.cmd %BuildStartTime% %time%
)
