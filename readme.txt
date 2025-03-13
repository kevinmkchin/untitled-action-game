

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


== Dependencies ==

Jolt - physics https://github.com/jrouwe/JoltPhysics
     - built with JPH_DEBUG_RENDERER only for Debug config
     - otherwise, mostly default configs in their CMakeLists were used
Recast - navigation mesh https://github.com/recastnavigation/recastnavigation
       - built with their default configs without any conflicts
NVIDIA OptiX - RTX ray tracing support for lightmapping
gmath.h - math
stb_ds.h - dynamic arrays

See ext\ for any other external libraries.

Consider https://github.com/rxi/cmixer to replace SDL_mixer


== Development Resources ==

https://web.archive.org/web/20130328024036/http://gafferongames.com/game-physics/fix-your-timestep/


== 3D Models and Characters ==

Use Blender for creating character models. 
- Use the +X axis in Blender as the forward direction of the character model (the character model should be facing the +X direction). When performing actions such as Symmetrize on an Armature, rotate the armature to the correct Blender forward orientation, apply, Symmetrize, rotate back to match the character model, and apply.
- Export in GLTF 2.0 binaries (so that materials are packed together with the model). Make sure to set +Y as the up axis in export settings. 
