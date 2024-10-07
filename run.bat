@echo off

IF "%1"=="vs" goto run-vs
IF "%1"=="subl" goto run-subl
IF "%1"=="r" goto run-release
IF "%1"=="release" goto run-release
IF "%1"=="-r" goto run-release
goto run-debug

:run-vs
devenv build\Debug\game.exe
goto end

:run-subl
subl game.sublime-project
goto end

:run-debug
START build\Debug\game.exe
goto end

:run-release
START build\Release\game.exe
goto end

:end
