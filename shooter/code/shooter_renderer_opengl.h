#ifndef SHOOTER_RENDERER_OPENGL_H
#define SHOOTER_RENDERER_OPENGL_H

#include "../ext/gl3w.h"
#include "shooter_renderer.h"

// TODO(Kevin): Framebuffer objects

//
// NOTE(Kevin): Rendering backend services to the rendering frontend
//

struct open_gl_mesh
{
    uint32 VAO;
    uint32 VBO;
    uint32 IBO;
    uint32 IndicesCount;
};

struct open_gl_texture
{
    uint32 Id;
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
    GLenum TargetFormat;
    GLenum SourceFormat;
    GLenum PixelDataType;
};

struct open_gl
{
    renderer Header;

    bool32 SupportsSRGBFramebuffer;

    //
    // NOTE(Kevin): OpenGL specific state goes here e.g. FBO handles, SSBOs probably
    //
    open_gl_mesh HandleToMesh[1024];
    mesh_handle FreeMeshHandles[1024];
    uint32 FreeMeshHandlesCount;

    open_gl_texture HandleToTexture[512];
    mesh_handle FreeTextureHandles[512];
    uint32 FreeTextureHandlesCount;

    struct open_gl_shader *TestShader;
};

mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
void OpenGLDrawMesh(open_gl *OpenGL, mesh_handle MeshHandle);
void OpenGLDeleteMesh(open_gl *OpenGL, mesh_handle MeshHandle);

texture_handle OpenGLUploadTexture2D(open_gl *OpenGL, void *bitmap, uint32 w, uint32 h, 
                                     pixel_format Format, bool32 LinearFiltering);
void OpenGLBindTexture2D(open_gl *OpenGL, uint32 Slot, texture_handle TextureHandle);
void OpenGLUpdateTexture2D(open_gl *OpenGL, texture_handle TextureHandle, void *bitmap, uint32 w, uint32 h);
void OpenGLDeleteGPUTexture2D(open_gl *OpenGL, texture_handle TextureHandle);

#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);
void Win32EndFrame(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_OPENGL_H
