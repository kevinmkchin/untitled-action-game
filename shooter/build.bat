@echo off

set CommonCompilerFlags=-MTd -nologo -WX -W4 -wd4201 -wd4100 -wd4189 -wd4505 -wd4324 -D_CRT_SECURE_NO_WARNINGS -DHANDMADE_INTERNAL=1 -DHANDMADE_SLOW=1 -FC -Z7 -I ..\ext
set CommonLinkerFlags=-incremental:no -opt:ref -subsystem:windows user32.lib gdi32.lib ole32.lib winmm.lib shcore.lib

set ts=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
set ts=%ts: =0%%time:~3,2%%time:~6,2%

IF NOT EXIST build mkdir build
pushd build
del *.pdb > NUL 2> NUL
REM Optimization switches /O2 /Oi /fp:fast
echo WAITING FOR PDB > lock.tmp
cl %CommonCompilerFlags% -LD ..\code\shooter.cpp -Fmshooter.map /link -incremental:no -PDB:shooter_%ts%.pdb -EXPORT:GameUpdateAndRender -EXPORT:GameGetSoundSamples
del lock.tmp
cl %CommonCompilerFlags% ..\code\win32_shooter.cpp -Fmwin32_shooter.map /link %CommonLinkerFlags%
popd

:: COPY DLLS ::
set OutputFolder=.
rem IF NOT EXIST build\%OutputFolder% MKDIR build\%OutputFolder%
IF NOT EXIST build\%OutputFolder%\renderdoc.dll copy ext\renderdoc.dll build\%OutputFolder%\renderdoc.dll >nul
