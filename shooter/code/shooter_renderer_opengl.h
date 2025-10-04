#ifndef SHOOTER_RENDERER_OPENGL_H
#define SHOOTER_RENDERER_OPENGL_H

#include "gl3w.h"
#include "shooter_renderer.h"

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
    uint32 ID;
    GLenum TargetFormat;
    GLenum SourceFormat;
    GLenum PixelDataType;
};

struct open_gl_framebuffer
{
    uint32 FBO;
    open_gl_texture ColorTex;
    open_gl_texture DepthTex;
    uint32 Width;
    uint32 Height;
};

struct vertex_attribute
{
    GLenum Format;
    uint32 NumComponents;
    uint32 Offset;
};

struct vertex_descriptor
{
    size_t ByteSize; // stride
    vertex_attribute Attrib[5];
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
    render_descriptor Descriptor;   // 12      : 204
    uint32 _padding_;               // 4       : 208

    // bool operator<(const open_gl_model_instance &Other) const
    // {
    //     // TODO(Kevin): Sorting...uhhh maybe we just sort render_items instead?
    //     //              Cuz everything we want to draw with instancing are drawn in one pipeline!
    //     return true;
    //     // if (!CorpseModel)
    //     //     return false;
    //     // if (!Other.CorpseModel)
    //     //     return true;
    //     // return CorpseModel->MT_ID < Other.CorpseModel->MT_ID;
    // }
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

struct vstream_frame_chunk
{
    void *WriteTo;
    memory_index ChunkSizeBytes;
};

struct persistent_vertex_stream
{
    GLuint VAO;
    GLuint VBO;
    memory_index VertexSize;
    static constexpr GLuint BindingIndex = 0;

    void *MappedPtr;
    umm FrameChunkSize;
    umm TotalSize;
    uint32 NumFrames;
    uint32 CurrentFrame;
    GLsync FrameSyncObjects[5];
};

struct gui_vstream_item
{
    umm OffsetBytes;
    u32 VertexCount;
    texture_handle ColorTexture;
    s32 ScissorX, ScissorY;
    u32 ScissorW, ScissorH;
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

    struct open_gl_shader *Shaders[16];
    uint32 ShadersCount;
    struct open_gl_shader *FinalCompositeShader;

    open_gl_framebuffer RenderTargetGame;
    open_gl_framebuffer RenderTargetGui;

    n_buffered_ssbo             PerInstanceDataSSBO;
    ssbo_frame_chunk            PerInstanceDataFrameChunk;
    uint32                      PerInstanceDataCount;

    persistent_vertex_stream    GuiVertexStream;
    vstream_frame_chunk         GuiVertexStreamFrameChunk;
    gui_vstream_item            GuiVertexStreamItems[1024];
    uint32                      GuiVertexStreamItemsCount;
};

void OpenGLActivatePipeline(open_gl *OpenGL, rendering_pipeline Pipeline);

void OpenGLGetPerInstanceDataArray(open_gl *OpenGL, open_gl_model_instance **OutArray, uint32 **OutCount);
void OpenGLGetGuiVertexStreamFrame(open_gl *OpenGL, vstream_frame_chunk *OutVertexStreamFrame);
void OpenGLGetGuiVertexStreamItemsArray(open_gl *OpenGL, gui_vstream_item** OutArray, uint32 **OutCount);

// mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV1_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
void OpenGLDrawMesh(open_gl *OpenGL, mesh_handle MeshHandle);
void OpenGLDrawMeshInstanced(open_gl *OpenGL, mesh_handle MeshHandle, uint32 Count);
void OpenGLDeleteMesh(open_gl *OpenGL, mesh_handle MeshHandle);

texture_handle OpenGLUploadTexture2D(open_gl *OpenGL, void *Bitmap, uint32 W, uint32 H, 
                                     pixel_format Format, bool32 LinearFiltering);
void OpenGLBindTexture2D(open_gl *OpenGL, uint32 Slot, texture_handle TextureHandle);
void OpenGLUpdateTexture2D(open_gl *OpenGL, texture_handle TextureHandle, void *Bitmap, uint32 W, uint32 H);
void OpenGLDeleteGPUTexture2D(open_gl *OpenGL, texture_handle TextureHandle);

void OpenGLCreateFramebuffer(open_gl_framebuffer *Framebuffer, uint32 W, uint32 H, bool32 DepthAttachment,
                             GLenum TargetFormat = GL_RGBA, GLenum SourceFormat = GL_RGBA,
                             GLenum PixelType = GL_UNSIGNED_BYTE);
void OpenGLResizeFramebuffer(open_gl_framebuffer *Framebuffer, uint32 W, uint32 H);
void OpenGLDeleteFramebuffer(open_gl_framebuffer *Framebuffer);

void OpenGLSSBOInitialize(n_buffered_ssbo *SSBO, memory_index FrameChunkSizeBytes, uint32 NumFrames = 3);
void OpenGLSSBODelete(n_buffered_ssbo *SSBO);
void OpenGLSSBOBindTo(n_buffered_ssbo *SSBO, GLuint BindingIndex);
ssbo_frame_chunk OpenGLSSBOBeginFrame(n_buffered_ssbo *SSBO);
void OpenGLSSBOEndFrame(n_buffered_ssbo *SSBO);

void OpenGLVertexStreamInitialize(persistent_vertex_stream *Stream, uint32 VertexCountPerFrame,
                                  vertex_descriptor VertexDescriptor, uint32 NumFrames = 3);
void OpenGLVertexStreamDelete(persistent_vertex_stream *Stream);
vstream_frame_chunk OpenGLVertexStreamBeginFrame(persistent_vertex_stream *Stream);
void OpenGLVertexStreamDraw(persistent_vertex_stream *Stream, umm OffsetBytes, uint32 VertexCount);
void OpenGLVertexStreamEndFrame(persistent_vertex_stream *Stream);

#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);
void Win32BeginFrame(renderer *Renderer);
void Win32EndFrame(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_OPENGL_H
