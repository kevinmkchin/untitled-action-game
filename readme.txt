

== Build ==

Using CMake to generate build for MSVC.

build           : build debug
build clean     : clean target
build debug     : output to build\Debug\
build release   : output to build\Release\. No debug info. Faster.

Copies required dlls from ext\
Inject data into BUILDINFO.H


