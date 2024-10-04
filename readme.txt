

== Build for Windows ==

BUILD.BAT calls MSVC cl.exe

build           : output to build\Debug\ with debug mode /Zi flag and creates PDB
build release   : output to build\Release\ with full optimizations /O2 flag

There is also a CMakeLists.txt but this might not be kept up to date. There primarily for IDEs.


== RenderDoc Integration ==

https://renderdoc.org/docs/in_application_api.html

RenderDoc graphics debugger is linked dynamically at program start. The RenderDoc overlay should appear in program. 

Press Home key to launch debugger UI.

API:
    RDOCAPI->LaunchReplayUI(1, "");
    if (RDOCAPI) RDOCAPI->StartFrameCapture(NULL, NULL);
    if (RDOCAPI) RDOCAPI->EndFrameCapture(NULL, NULL);

Captures saved to C:\Users\Kevin\AppData\Local\Temp\RenderDoc but can be changed with RDOCAPI->SetCaptureFilePathTemplate.


== Development ==

gmath.h - math
stb_ds.h - dynamic arrays

See ext\ for any other external libraries.

Consider https://github.com/rxi/cmixer to replace SDL_mixer

