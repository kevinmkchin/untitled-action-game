

== Build ==

Using CMake to generate build for MSVC.

build           : build debug
build clean     : clean target
build debug     : output to build\Debug\
build release   : output to build\Release\. No debug info. Faster.

Copies required dlls from ext\
Inject data into BUILDINFO.H


== RenderDoc Integration ==

https://renderdoc.org/docs/in_application_api.html

RenderDoc graphics debugger is linked dynamically at program start. The RenderDoc overlay should appear in program.

API:
    RDOCAPI->LaunchReplayUI(1, "");
    if (RDOCAPI) RDOCAPI->StartFrameCapture(NULL, NULL);
    if (RDOCAPI) RDOCAPI->EndFrameCapture(NULL, NULL);

Captures saved to C:\Users\Kevin\AppData\Local\Temp\RenderDoc but can be changed with RDOCAPI->SetCaptureFilePathTemplate.


== Debugging ==

PDB file should be generated when building in debug mode.
For some reason, data breakpoints don't work when opening Visual Studio at CMake source dir, but it works if open generated solutions or open executable file.


== Development ==

gmath.h - math
stb_ds.h - dynamic arrays

See ext\ for any other external libraries.

