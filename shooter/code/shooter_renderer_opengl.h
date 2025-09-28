#ifndef SHOOTER_RENDERER_OPENGL_H
#define SHOOTER_RENDERER_OPENGL_H

#include "../ext/gl3w.h"
#include "shooter_renderer.h"
#include "shooter_renderer_opengl_shaders.h"

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
    GLenum TargetFormat;
    GLenum SourceFormat;
    GLenum PixelDataType;
};

struct alignas(16) open_gl_model_instance
{
    //
    // NOTE(Kevin): Per-instance data for models to be drawn in a 3D world
    //              Struct must be passable to GLSL shader. See:
    //              https://community.khronos.org/t/ssbo-std430-layout-rules/109761
    //              std430 Layout
    //

    mat4 WorldFromModel;            // 64      : 64
    vec4 PointLightsPos[4];         // 64      : 128
    real32 AmbientCube[6];          // 4*6=24  : 152
    int32 DoSunLight;               // 4       : 156
    int32 PointLightsCount;         // 4       : 160
    real32 PointLightsAttLin[4];    // 4*4=16  : 176
    real32 PointLightsAttQuad[4];   // 4*4=16  : 192
    uint64 _reserved_;                 // 8       : 200
    uint64 _padding_;                  // 8       : 208

    bool operator<(const open_gl_model_instance &Other) const
    {
        // TODO(Kevin): Sorting...uhhh maybe we just sort render_items instead?
        //              Cuz everything we want to draw with instancing are drawn in one pipeline!
        return true;
        // if (!CorpseModel)
        //     return false;
        // if (!Other.CorpseModel)
        //     return true;
        // return CorpseModel->MT_ID < Other.CorpseModel->MT_ID;
    }
};

struct ssbo_frame_chunk
{
    void *WriteTo;
    umm BaseOffset; // BaseInstanceIndex = BaseOffset / sizeof(ElementStoredInSSBO)
};

struct n_buffered_ssbo
{
    GLuint BufferObject;
    void *MappedPtr;
    umm FrameChunkSize;
    umm TotalSize;
    uint32 NumFrames;
    uint32 CurrentFrame;
    GLsync FrameSyncObjects[5];
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

    open_gl_shader *Shaders[16];
    uint32 ShadersCount;

    n_buffered_ssbo PerInstanceDataSSBO;
};

void OpenGLActivatePipeline(open_gl *OpenGL, rendering_pipeline Pipeline);

mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV1_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
void OpenGLDrawMesh(open_gl *OpenGL, mesh_handle MeshHandle);
void OpenGLDrawMeshInstanced(open_gl *OpenGL, mesh_handle MeshHandle, uint32 Count);
void OpenGLDeleteMesh(open_gl *OpenGL, mesh_handle MeshHandle);

texture_handle OpenGLUploadTexture2D(open_gl *OpenGL, void *bitmap, uint32 w, uint32 h, 
                                     pixel_format Format, bool32 LinearFiltering);
void OpenGLBindTexture2D(open_gl *OpenGL, uint32 Slot, texture_handle TextureHandle);
void OpenGLUpdateTexture2D(open_gl *OpenGL, texture_handle TextureHandle, void *bitmap, uint32 w, uint32 h);
void OpenGLDeleteGPUTexture2D(open_gl *OpenGL, texture_handle TextureHandle);

void OpenGLSSBOInitialize(n_buffered_ssbo *SSBO, memory_index FrameChunkSizeBytes, uint32 NumFrames = 3);
void OpenGLSSBODelete(n_buffered_ssbo *SSBO);
void OpenGLSSBOBindTo(n_buffered_ssbo *SSBO, GLuint BindingIndex);
ssbo_frame_chunk OpenGLSSBOBeginFrame(n_buffered_ssbo *SSBO);
void OpenGLSSBOEndFrame(n_buffered_ssbo *SSBO);

#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);
void Win32EndFrame(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_OPENGL_H
