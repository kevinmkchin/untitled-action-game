@echo off

IF "%1"=="vs" goto run-vs
IF "%1"=="subl" goto run-subl
IF "%1"=="release" goto run-release
goto run-bin

:run-vs
REM build\Debug\game.exe
devenv build\game.sln
goto end

:run-subl
subl source.sublime-project
goto end

:run-bin
START build\Debug\game.exe
goto end

:run-release
START build\Release\game.exe
goto end

:end
