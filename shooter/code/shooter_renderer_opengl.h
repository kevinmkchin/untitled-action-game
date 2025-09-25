#ifndef SHOOTER_RENDERER_OPENGL_H
#define SHOOTER_RENDERER_OPENGL_H

#include "shooter_renderer.h"

struct open_gl
{
    renderer Header;

    bool32 SupportsSRGBFramebuffer;

    //
    // NOTE(Kevin): OpenGL specific state goes here e.g. FBO handles, SSBOs probably
    //
};



#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);

void Win32EndFrame(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_OPENGL_H
