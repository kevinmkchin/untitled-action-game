
== Build Instructions ==

Requires Visual Studio 2022 with the following workload and components:
  - Desktop development with C++
    - MSVC v143 - VS 2022 C++ x64/x86 build tools
    - Windows 11 SDK (10.0.22621.0)

MSVC x64 cl.exe must be in PATH environment variable. You can use the Developer
Command Prompts installed with Visual Studio. Run `build.bat` to build the game.

Open the compiled executable with Visual Studio to debug it. At the moment, you
must first right click the executable in the Solution Explorer, go to Properties,
and set the Working Directory parameter to the `data` directory. Press F5 to debug.

