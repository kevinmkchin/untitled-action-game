@echo off

CALL build.bat
START /b C:\remedybg\remedybg.exe -g -q build\Debug\session.rdbg
