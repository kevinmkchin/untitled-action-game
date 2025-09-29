#include "shooter_renderer_opengl.h"
#include "shooter_shared.h"
#define GL3W_IMPLEMENTATION
#include "../ext/gl3w.h"

internal void OpenGLInit(open_gl *OpenGL)
{
    renderer *RendererFrontend = &(OpenGL->Header);

    debug_read_file_result TestVSRead = DEBUGPlatformReadEntireFile(0, "../data/glsl/test.vert");
    debug_read_file_result TestFSRead = DEBUGPlatformReadEntireFile(0, "../data/glsl/test.frag");
    char *TestVS = (char *)TestVSRead.Contents;
    char *TestFS = (char *)TestFSRead.Contents;
    // TODO(Kevin): This is just test code! Get rid of new! Constructor required for std::unordered_map bullshit
    OpenGL->Shaders[0] = new open_gl_shader();
    ++OpenGL->ShadersCount;
    GLCreateShaderProgram(*(OpenGL->Shaders[0]), TestVS, TestFS);
    DEBUGPlatformFreeFileMemory(0, TestVSRead.Contents);
    DEBUGPlatformFreeFileMemory(0, TestFSRead.Contents);
    RendererFrontend->WorldGeometryTest.ShaderID = 0;
    RendererFrontend->WorldGeometryTest.DepthTest = true;

    TestVSRead = DEBUGPlatformReadEntireFile(0, "../data/glsl/StaticLit.vert");
    TestFSRead = DEBUGPlatformReadEntireFile(0, "../data/glsl/StaticLit.frag");
    TestVS = (char *)TestVSRead.Contents;
    TestFS = (char *)TestFSRead.Contents;
    OpenGL->Shaders[1] = new open_gl_shader();
    ++OpenGL->ShadersCount;
    GLCreateShaderProgram(*(OpenGL->Shaders[1]), TestVS, TestFS);
    DEBUGPlatformFreeFileMemory(0, TestVSRead.Contents);
    DEBUGPlatformFreeFileMemory(0, TestFSRead.Contents);
    RendererFrontend->StaticLit.ShaderID = 1;
    RendererFrontend->StaticLit.DepthTest = true;

    RendererFrontend->StaticLitItemsMaxCount = 2000;
    OpenGLSSBOInitialize(&(OpenGL->PerInstanceDataSSBO), 
                         sizeof(open_gl_model_instance)*RendererFrontend->StaticLitItemsMaxCount, 3);

    OpenGL->FreeMeshHandlesCount = ArrayCount(OpenGL->FreeMeshHandles);
    for (int32 FreeHandle = 0; 
         FreeHandle < SafeCastToInt32(OpenGL->FreeMeshHandlesCount);
         ++FreeHandle)
    {
        OpenGL->FreeMeshHandles[OpenGL->FreeMeshHandlesCount - FreeHandle - 1] = 
            (mesh_handle)SafeCastToUInt32(FreeHandle);
    }

    OpenGL->FreeTextureHandlesCount = ArrayCount(OpenGL->FreeTextureHandles);
    for (int32 FreeHandle = 0; 
         FreeHandle < SafeCastToInt32(OpenGL->FreeTextureHandlesCount);
         ++FreeHandle)
    {
        OpenGL->FreeTextureHandles[OpenGL->FreeTextureHandlesCount - FreeHandle - 1] = 
            (texture_handle)SafeCastToUInt32(FreeHandle);
    }
}

internal void OpenGLBeginFrame(open_gl *OpenGL)
{
    OpenGL->PerInstanceDataFrameChunk = OpenGLSSBOBeginFrame(&(OpenGL->PerInstanceDataSSBO));
    OpenGL->PerInstanceDataCount = 0;
}

internal void OpenGLEndFrame(open_gl *OpenGL)
{
    renderer *RendererFrontend = &(OpenGL->Header);

    glViewport(0, 0, OpenGL->Header.BackBufferW, OpenGL->Header.BackBufferH);
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    // glEnable(GL_BLEND);
    // glEnable(GL_CULL_FACE);
    // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);

    // TODO(Kevin): Muzzle flash should made generic in list of "real-time dynamic point light sources"
    // which are the only point lights that meshes will check against (since they're dynamic). Probably,
    // not shadow casting unless I want that as a very pronounced effect e.g. Doom 3.

    mat4 ViewFromWorld = ViewMatrixLookAt(vec3(-100,60,0), vec3(-100,60,0) + vec3(1,-0.1f,0), GM_UP_VECTOR);
    float AspectRatio = float(OpenGL->Header.BackBufferW) / float(OpenGL->Header.BackBufferH);
    float FOVX = 90.f;
    float FOVYRad = HorizontalFOVToVerticalFOV(FOVX*GM_DEG2RAD, AspectRatio);
    mat4 ClipFromView = ProjectionMatrixPerspective(FOVYRad, AspectRatio, 4.f, 3200.f);

    // OpenGLBindTexture2D(OpenGL, 0, ColorTextureHandle)
    OpenGLBindTexture2D(OpenGL, 1, 0);

    OpenGLActivatePipeline(OpenGL, RendererFrontend->WorldGeometryTest);
    open_gl_shader *WorldGeomShader = OpenGL->Shaders[RendererFrontend->WorldGeometryTest.ShaderID];

    GLBindMatrix4fv(*WorldGeomShader, "viewMatrix", 1, ViewFromWorld.ptr());
    GLBindMatrix4fv(*WorldGeomShader, "projMatrix", 1, ClipFromView.ptr());    
    GLBind1i(*WorldGeomShader, "ColorTexture", 0);
    GLBind1i(*WorldGeomShader, "LightMap", 1);

    OpenGLDrawMesh(OpenGL, 1);
    OpenGLDrawMesh(OpenGL, 2);


    //
    // NOTE(Kevin): Model instancing
    //
    OpenGLActivatePipeline(OpenGL, RendererFrontend->StaticLit);
    open_gl_shader *StaticLitShader = OpenGL->Shaders[RendererFrontend->StaticLit.ShaderID];

    vec3 DirectionToSun = vec3(-1,1,1);
    // GLBind4f(Sha_ModelInstancedLit, "MuzzleFlash",
    //     RuntimeMap->Player.Weapon.MuzzleFlash.x, 
    //     RuntimeMap->Player.Weapon.MuzzleFlash.y, 
    //     RuntimeMap->Player.Weapon.MuzzleFlash.z, 
    //     RuntimeMap->Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(*StaticLitShader, "ProjFromView", 1, ClipFromView.ptr());
    GLBindMatrix4fv(*StaticLitShader, "ViewFromWorld", 1, ViewFromWorld.ptr());
    GLBind3f(*StaticLitShader, "DirectionToSun", DirectionToSun.x, DirectionToSun.y, DirectionToSun.z);

    if (OpenGL->PerInstanceDataCount > 0)
    {
        Assert(OpenGL->PerInstanceDataFrameChunk.WriteTo);

        OpenGLSSBOBindTo(&(OpenGL->PerInstanceDataSSBO), 0);
        uint32 BaseInstanceIndex = SafeTruncateUInt64(OpenGL->PerInstanceDataFrameChunk.BaseOffset / 
                                                      sizeof(open_gl_model_instance));

        open_gl_model_instance *PerInstanceData = (open_gl_model_instance *)OpenGL->PerInstanceDataFrameChunk.WriteTo;

        uint32 CurrentDescriptorCount = 0;
        uint32 CurrentOffsetFromBase = 0;
        render_descriptor CurrentDescriptor = PerInstanceData[0].Descriptor;

        for (uint32 PerInstanceDataIndex = 0;
             PerInstanceDataIndex <= OpenGL->PerInstanceDataCount; 
             ++PerInstanceDataIndex)
        {
            if (PerInstanceDataIndex == OpenGL->PerInstanceDataCount ||
                PerInstanceData[PerInstanceDataIndex].Descriptor != CurrentDescriptor)
            {
                GLBind1i(*StaticLitShader, "BaseInstanceIndex", 
                         (GLint)SafeCastToInt32(BaseInstanceIndex + CurrentOffsetFromBase));

                OpenGLBindTexture2D(OpenGL, 0, CurrentDescriptor.Albedo);
                OpenGLDrawMeshInstanced(OpenGL, CurrentDescriptor.Mesh, CurrentDescriptorCount);

                if (PerInstanceDataIndex == OpenGL->PerInstanceDataCount)
                {
                    break;
                }

                CurrentOffsetFromBase += CurrentDescriptorCount;
                CurrentDescriptor = PerInstanceData[PerInstanceDataIndex].Descriptor;
                CurrentDescriptorCount = 0;
            }

            ++CurrentDescriptorCount;
        }
    }

    OpenGLSSBOEndFrame(&(OpenGL->PerInstanceDataSSBO));
}

internal mesh_handle NextAvailableMeshHandle(open_gl *OpenGL)
{
    Assert(OpenGL->FreeMeshHandlesCount > 0);
    mesh_handle FreeHandle = OpenGL->FreeMeshHandles[--OpenGL->FreeMeshHandlesCount];
    return FreeHandle;
}

internal texture_handle NextAvailableTextureHandle(open_gl *OpenGL)
{
    Assert(OpenGL->FreeTextureHandlesCount > 0);
    texture_handle FreeHandle = OpenGL->FreeTextureHandles[--OpenGL->FreeTextureHandlesCount];
    return FreeHandle;
}

void OpenGLActivatePipeline(open_gl *OpenGL, rendering_pipeline Pipeline)
{
    Assert(Pipeline.ShaderID < OpenGL->ShadersCount);
    open_gl_shader *OpenglShader = (OpenGL->Shaders[Pipeline.ShaderID]);

    Assert(OpenglShader->ProgramId);
    glUseProgram(OpenglShader->ProgramId);

    if (Pipeline.DepthTest)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }

    // TODO(Kevin): blending, backface culling, etc.
    // glEnable(GL_BLEND);
    // glEnable(GL_CULL_FACE);
    // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
}

void OpenGLGetPerInstanceDataArray(open_gl *OpenGL, open_gl_model_instance **OutArray, uint32 **OutCount)
{
    *OutArray = (open_gl_model_instance *)OpenGL->PerInstanceDataFrameChunk.WriteTo;
    *OutCount = &OpenGL->PerInstanceDataCount;
}

mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV1_vertex *VB, uint32 VC, uint32 *IB, uint32 IC)
{
    GLsizei StrideInBytes = sizeof(PNUV1_vertex);
    GLsizeiptr VertexBufferSzInBytes = (GLsizeiptr)sizeof(PNUV1_vertex)*VC;
    GLsizeiptr IndexBufferSzInBytes = (GLsizeiptr)sizeof(uint32)*IC;

    open_gl_mesh Mesh = {};

    glGenVertexArrays(1, &Mesh.VAO);
    glBindVertexArray(Mesh.VAO);
    glGenBuffers(1, &Mesh.VBO);
    glBindBuffer(GL_ARRAY_BUFFER, Mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, VertexBufferSzInBytes, VB, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, StrideInBytes, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(PNUV1_vertex, IJK));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(PNUV1_vertex, UV0));

    glGenBuffers(1, &Mesh.IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Mesh.IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSzInBytes, IB, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Mesh.IndicesCount = IC;
    mesh_handle NewHandle = NextAvailableMeshHandle(OpenGL);
    OpenGL->HandleToMesh[NewHandle] = Mesh;

    return NewHandle;
}

mesh_handle OpenGLUploadMesh(open_gl *OpenGL, PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC)
{
    GLsizei StrideInBytes = sizeof(PNUV2_vertex);
    GLsizeiptr VertexBufferSzInBytes = (GLsizeiptr)sizeof(PNUV2_vertex)*VC;
    GLsizeiptr IndexBufferSzInBytes = (GLsizeiptr)sizeof(uint32)*IC;

    open_gl_mesh Mesh = {};

    glGenVertexArrays(1, &Mesh.VAO);
    glBindVertexArray(Mesh.VAO);
    glGenBuffers(1, &Mesh.VBO);
    glBindBuffer(GL_ARRAY_BUFFER, Mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, VertexBufferSzInBytes, VB, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, StrideInBytes, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(PNUV2_vertex, IJK));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(PNUV2_vertex, UV0));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(PNUV2_vertex, UV1));

    glGenBuffers(1, &Mesh.IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Mesh.IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSzInBytes, IB, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Mesh.IndicesCount = IC;
    mesh_handle NewHandle = NextAvailableMeshHandle(OpenGL);
    OpenGL->HandleToMesh[NewHandle] = Mesh;

    return NewHandle;
}

void OpenGLDrawMesh(open_gl *OpenGL, mesh_handle MeshHandle)
{
    open_gl_mesh Mesh = OpenGL->HandleToMesh[MeshHandle];

    Assert(Mesh.IndicesCount > 0);

    glBindVertexArray(Mesh.VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Mesh.IBO);
    glDrawElements(GL_TRIANGLES, Mesh.IndicesCount, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLDrawMeshInstanced(open_gl *OpenGL, mesh_handle MeshHandle, uint32 Count)
{
    open_gl_mesh Mesh = OpenGL->HandleToMesh[MeshHandle];

    Assert(Mesh.IndicesCount > 0);

    glBindVertexArray(Mesh.VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Mesh.IBO);
    glDrawElementsInstanced(GL_TRIANGLES, Mesh.IndicesCount, GL_UNSIGNED_INT, 0, Count);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLDeleteMesh(open_gl *OpenGL, mesh_handle MeshHandle)
{
    open_gl_mesh Mesh = OpenGL->HandleToMesh[MeshHandle];

    if (Mesh.IBO != 0)
    {
        glDeleteBuffers(1, &Mesh.IBO);
        Mesh.IBO = 0;
    }
    if (Mesh.VBO != 0)
    {
        glDeleteBuffers(1, &Mesh.VBO);
        Mesh.VBO = 0;
    }
    if (Mesh.VAO != 0)
    {
        glDeleteVertexArrays(1, &Mesh.VAO);
        Mesh.VAO = 0;
    }
    Mesh.IndicesCount = 0;

    OpenGL->FreeMeshHandles[OpenGL->FreeMeshHandlesCount++] = MeshHandle;
}

texture_handle OpenGLUploadTexture2D(open_gl *OpenGL, void *Bitmap, uint32 W, uint32 H, 
                                     pixel_format Format, bool32 LinearFiltering)
{
    open_gl_texture Texture;

    switch (Format)
    {
    case R_U8:
        Texture.TargetFormat = GL_RGBA;
        Texture.SourceFormat = GL_RED;
        Texture.PixelDataType = GL_UNSIGNED_BYTE;
        break;
    case R_Float:
        Texture.TargetFormat = GL_RGBA;
        Texture.SourceFormat = GL_RED;
        Texture.PixelDataType = GL_FLOAT;
        break;
    default:
    case RGBA_U8:
        Texture.TargetFormat = GL_RGBA;
        Texture.SourceFormat = GL_RGBA;
        Texture.PixelDataType = GL_UNSIGNED_BYTE;
        break;
    case RGBA_Float:
        Texture.TargetFormat = GL_RGBA;
        Texture.SourceFormat = GL_RGBA;
        Texture.PixelDataType = GL_FLOAT;
        break;
    case BGRA_U8:
        Texture.TargetFormat = GL_RGBA;
        Texture.SourceFormat = GL_BGRA;
        Texture.PixelDataType = GL_UNSIGNED_BYTE;
        break;
    case BGRA_Float:
        Texture.TargetFormat = GL_RGBA;
        Texture.SourceFormat = GL_BGRA;
        Texture.PixelDataType = GL_FLOAT;
        break;
    }

    glGenTextures(1, &Texture.Id);   // generate texture and grab texture id

    glBindTexture(GL_TEXTURE_2D, Texture.Id);
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
    glTexImage2D(
            GL_TEXTURE_2D,            // texture target type
            0,                        // level-of-detail number n = n-th mipmap reduction image
            Texture.TargetFormat,     // format of data to store (target): num of color components
            W,                        // texture width
            H,                        // texture height
            0,                        // must be 0 (legacy)
            Texture.SourceFormat,     // format of data being loaded (source)
            Texture.PixelDataType,    // data type of the texture data
            Bitmap);                  // data

    /*
        GL_NEAREST_MIPMAP_NEAREST
            - Sharp transitions between mip levels, "popping"
            - Pixelated appearance
            - Maximum performance
        GL_NEAREST_MIPMAP_LINEAR
            - Smooth transitions between mip levels, no visible "popping"
            - But still pixelated inside each mip level
        GL_LINEAR_MIPMAP_NEAREST
            - Bilinear filtering
            - Smoother appearance within levels with visible "popping" level transitions
        GL_LINEAR_MIPMAP_LINEAR
            - High-quality texture sampling
            - Trilinear filtering
    */

    GLenum MinFilterMode = GL_NEAREST_MIPMAP_NEAREST;
    GLenum MagFilterMode = GL_NEAREST;
    if (LinearFiltering)
    {
        MinFilterMode = GL_LINEAR_MIPMAP_LINEAR;
        MagFilterMode = GL_LINEAR;
    }

    if (MinFilterMode == GL_NEAREST_MIPMAP_LINEAR || MinFilterMode == GL_NEAREST_MIPMAP_NEAREST ||
        MinFilterMode == GL_LINEAR_MIPMAP_NEAREST || MinFilterMode == GL_LINEAR_MIPMAP_LINEAR)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MinFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, MagFilterMode);

    glBindTexture(GL_TEXTURE_2D, 0);

    texture_handle NewHandle = NextAvailableTextureHandle(OpenGL);
    OpenGL->HandleToTexture[NewHandle] = Texture;

    return NewHandle;
}

void OpenGLBindTexture2D(open_gl *OpenGL, uint32 Slot, texture_handle TextureHandle)
{
    open_gl_texture Texture = OpenGL->HandleToTexture[TextureHandle];
    
    GLenum TextureUnit = GL_TEXTURE0;
    switch (Slot)
    {
        case 0: TextureUnit = GL_TEXTURE0; break;
        case 1: TextureUnit = GL_TEXTURE1; break;
        case 2: TextureUnit = GL_TEXTURE2; break;
        case 3: TextureUnit = GL_TEXTURE3; break;
        case 4: TextureUnit = GL_TEXTURE4; break;
        case 5: TextureUnit = GL_TEXTURE5; break;
        case 6: TextureUnit = GL_TEXTURE6; break;
        case 7: TextureUnit = GL_TEXTURE7; break;
        case 8: TextureUnit = GL_TEXTURE8; break;
        case 9: TextureUnit = GL_TEXTURE9; break;
        case 10: TextureUnit = GL_TEXTURE10; break;
        case 11: TextureUnit = GL_TEXTURE11; break;
        case 12: TextureUnit = GL_TEXTURE12; break;
        case 13: TextureUnit = GL_TEXTURE13; break;
        case 14: TextureUnit = GL_TEXTURE14; break;
        case 15: TextureUnit = GL_TEXTURE15; break;
        default: Assert(0); // TODO(Kevin): Logging
    }

    glActiveTexture(TextureUnit);
    glBindTexture(GL_TEXTURE_2D, Texture.Id);
}

void OpenGLSSBOInitialize(n_buffered_ssbo *SSBO, umm FrameChunkSizeBytes, uint32 NumFrames)
{
    *SSBO = {};

    Assert(NumFrames <= 5);

    SSBO->NumFrames = NumFrames;
    SSBO->CurrentFrame = 0;
    SSBO->FrameChunkSize = FrameChunkSizeBytes;
    SSBO->TotalSize = SSBO->FrameChunkSize * SSBO->NumFrames;

    glGenBuffers(1, &(SSBO->BufferObject));
    Assert(SSBO->BufferObject);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO->BufferObject);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, SSBO->TotalSize, 0,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    SSBO->MappedPtr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, SSBO->TotalSize,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    Assert(SSBO->MappedPtr);
}

void OpenGLSSBODelete(n_buffered_ssbo *SSBO)
{
    if (SSBO->BufferObject)
    {
        glDeleteBuffers(1, &(SSBO->BufferObject));
        SSBO->BufferObject = 0;
        SSBO->MappedPtr = 0;
        // TODO(Kevin): probably glDeleteSync
    }
}

void OpenGLSSBOBindTo(n_buffered_ssbo *SSBO, GLuint BindingIndex)
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BindingIndex, SSBO->BufferObject);
}

ssbo_frame_chunk OpenGLSSBOBeginFrame(n_buffered_ssbo *SSBO)
{
    SSBO->CurrentFrame = (SSBO->CurrentFrame + 1) % SSBO->NumFrames;

    // IMPORTANT See note in OpenGLSSBOEndFrame
    // // When a fence becomes signaled, you can be certain that the GPU has completed all 
    // // OpenGL commands issued before the fence was created.
    // GLenum Status = GL_UNSIGNALED;
    // while (Status != GL_ALREADY_SIGNALED && Status != GL_CONDITION_SATISFIED && SSBO->FrameSyncObjects[SSBO->CurrentFrame] != nullptr)
    // {
    //     Status = glClientWaitSync(SSBO->FrameSyncObjects[SSBO->CurrentFrame], GL_SYNC_FLUSH_COMMANDS_BIT, 1);
    // }

    umm GlobalOffset = SSBO->CurrentFrame * SSBO->FrameChunkSize;

    ssbo_frame_chunk Result;
    Result.WriteTo = ((uint8 *)SSBO->MappedPtr) + GlobalOffset;
    Result.BaseOffset = GlobalOffset;
    return Result;
}

void OpenGLSSBOEndFrame(n_buffered_ssbo *SSBO)
{
    // IMPORTANT
    // NOTE(Kevin) 2025-03-31: Maybe this is a bug on my GPU, but calling glFenceSync
    //      is leaking memory. In Task Manager and Visual Studio can observe memory rising
    //      every frame this is called even if glDeleteSync is also called...
    //      hopefully this issue doesn't persist when I port to Vulkan.
    //      ACTUALLY when VSYNC is ON it does not leak memory. Which makes me think it's something
    //      to do with the CPU submitting jobs too quickly maybe. ChatGPT says this is expected on
    //      NVIDIA cards it is a backpressure problem. Hopefully the issue goes away when the CPU
    //      is taking longer to complete the frame and CPU GPU work is more balanced.
    // glDeleteSync(SSBO->FrameSyncObjects[SSBO->CurrentFrame]);
    // SSBO->FrameSyncObjects[SSBO->CurrentFrame] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


#ifdef SHOOTER_PLATFORM_WIN32

#define WGL_CONTEXT_MAJOR_VERSION_ARB           0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB           0x2092
#define WGL_CONTEXT_LAYER_PLANE_ARB             0x2093
#define WGL_CONTEXT_FLAGS_ARB                   0x2094
#define WGL_CONTEXT_PROFILE_MASK_ARB            0x9126

#define WGL_CONTEXT_DEBUG_BIT_ARB               0x0001
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB  0x0002

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB        0x00000001
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002

#define WGL_DRAW_TO_WINDOW_ARB                  0x2001
#define WGL_ACCELERATION_ARB                    0x2003
#define WGL_SUPPORT_OPENGL_ARB                  0x2010
#define WGL_DOUBLE_BUFFER_ARB                   0x2011
#define WGL_PIXEL_TYPE_ARB                      0x2013

#define WGL_TYPE_RGBA_ARB                       0x202B
#define WGL_FULL_ACCELERATION_ARB               0x2027

#define WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB        0x20A9

typedef HGLRC WINAPI wgl_create_context_attribs_arb(HDC hDC, HGLRC hShareContext, const int *attribList);

typedef BOOL WINAPI wgl_choose_pixel_format_arb(HDC hdc,
                                                const int *piAttribIList,
                                                const FLOAT *pfAttribFList,
                                                UINT nMaxFormats,
                                                int *piFormats,
                                                UINT *nNumFormats);

typedef BOOL WINAPI wgl_swap_interval_ext(int interval);

typedef const char * WINAPI wgl_get_extensions_string_ext(void);

global wgl_create_context_attribs_arb *wglCreateContextAttribsARB;
global wgl_choose_pixel_format_arb *wglChoosePixelFormatARB;
global wgl_swap_interval_ext *wglSwapIntervalEXT;
global wgl_get_extensions_string_ext *wglGetExtensionsStringEXT;

internal void
Win32SetPixelFormat(open_gl *OpenGL, HDC WindowDC)
{
    int SuggestedPixelFormatIndex = 0;
    GLuint ExtendedPick = 0;

    if (wglChoosePixelFormatARB)
    {
        int IntAttribList[] =
        {
            WGL_DRAW_TO_WINDOW_ARB, GL_TRUE, // 0
            WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB, // 1
            WGL_SUPPORT_OPENGL_ARB, GL_TRUE, // 2
            WGL_DOUBLE_BUFFER_ARB, GL_TRUE, // 3
            WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB, // 4
            WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB, GL_TRUE, // 5
            0,
        };
        
        if (!OpenGL->SupportsSRGBFramebuffer)
        {
            IntAttribList[10] = 0;
        }

        wglChoosePixelFormatARB(WindowDC, IntAttribList, 0, 1, &SuggestedPixelFormatIndex, &ExtendedPick);
    }
    
    if (!ExtendedPick)
    {
        PIXELFORMATDESCRIPTOR DesiredPixelFormat = {};
        DesiredPixelFormat.nSize = sizeof(DesiredPixelFormat);
        DesiredPixelFormat.nVersion = 1;
        DesiredPixelFormat.iPixelType = PFD_TYPE_RGBA;
        DesiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL|PFD_DRAW_TO_WINDOW|PFD_DOUBLEBUFFER;
        DesiredPixelFormat.cColorBits = 32;
        DesiredPixelFormat.cAlphaBits = 8;
        DesiredPixelFormat.cDepthBits = 24;
        DesiredPixelFormat.iLayerType = PFD_MAIN_PLANE;
        
        SuggestedPixelFormatIndex = ChoosePixelFormat(WindowDC, &DesiredPixelFormat);
    }

    PIXELFORMATDESCRIPTOR SuggestedPixelFormat;
    // NOTE(Kevin): Technically you do not need to call DescribePixelFormat here,
    // as SetPixelFormat doesn't actually need it to be filled out properly.
    DescribePixelFormat(WindowDC, SuggestedPixelFormatIndex, sizeof(SuggestedPixelFormat), &SuggestedPixelFormat);
    SetPixelFormat(WindowDC, SuggestedPixelFormatIndex, &SuggestedPixelFormat);
}

internal void Win32LoadWGLExtensions(open_gl *OpenGL)
{
    WNDCLASSA WindowClass = {};
    WindowClass.lpfnWndProc = DefWindowProcA;
    WindowClass.hInstance = GetModuleHandle(0);
    WindowClass.lpszClassName = "ShooterGameWGLLoader";

    if (RegisterClassA(&WindowClass))
    {
        HWND Window = CreateWindowExA(
            0,
            WindowClass.lpszClassName,
            "Shooter Game",
            0,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            0,
            0,
            WindowClass.hInstance,
            0);
        
        HDC WindowDC = GetDC(Window);

        Win32SetPixelFormat(OpenGL, WindowDC);

        HGLRC OpenGLRC = wglCreateContext(WindowDC);

        if (wglMakeCurrent(WindowDC, OpenGLRC))
        {
            wglChoosePixelFormatARB =
                (wgl_choose_pixel_format_arb *)wglGetProcAddress("wglChoosePixelFormatARB");
            wglCreateContextAttribsARB =
                (wgl_create_context_attribs_arb *)wglGetProcAddress("wglCreateContextAttribsARB");
            wglGetExtensionsStringEXT = 
                (wgl_get_extensions_string_ext *)wglGetProcAddress("wglGetExtensionsStringEXT");

            if (wglGetExtensionsStringEXT)
            {
                char *Extensions = (char *)wglGetExtensionsStringEXT();
                char *At = Extensions;
                while (*At)
                {
                    while (IsWhitespace(*At)) {++At;}
                    char *End = At;
                    while (*End && !IsWhitespace(*End)) {++End;}

                    umm Count = End - At;

                    if (0) {}
                    else if (StringsAreEqual(Count, At, "WGL_EXT_framebuffer_sRGB")) { OpenGL->SupportsSRGBFramebuffer = true; }
                    else if (StringsAreEqual(Count, At, "WGL_ARB_framebuffer_sRGB")) { OpenGL->SupportsSRGBFramebuffer = true; }

                    At = End;
                }
            }
            
            wglMakeCurrent(0, 0);
        }

        wglDeleteContext(OpenGLRC);

        ReleaseDC(Window, WindowDC);

        DestroyWindow(Window);
    }
}

internal void *
Win32RendererAlloc(umm Size)
{
    void *Result = VirtualAlloc(0, Size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    return Result;
}

internal void
PlatformOpenGLSetVSync(open_gl *OpenGL, bool32 VSyncEnabled)
{
    if (!wglSwapIntervalEXT)
    {
        wglSwapIntervalEXT = (wgl_swap_interval_ext *)wglGetProcAddress("wglSwapIntervalEXT");
    }

    if (wglSwapIntervalEXT)
    {
        wglSwapIntervalEXT(VSyncEnabled ? 1 : 0);
    }
}

global int Win32OpenGLAttribs[] =
{
    WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
    WGL_CONTEXT_MINOR_VERSION_ARB, 3,
    WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB
#if HANDMADE_INTERNAL
        |WGL_CONTEXT_DEBUG_BIT_ARB
#endif
    ,
#if 0
    WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
#else
    WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
#endif
    0,
};

internal open_gl *
Win32InitOpenGL(HDC WindowDC /* TODO(Kevin): platform limits?*/)
{
    open_gl *OpenGL = (open_gl *)Win32RendererAlloc(sizeof(open_gl));
    ConstructRendererServiceAt(&(OpenGL->Header));
    Win32LoadWGLExtensions(OpenGL);
    Win32SetPixelFormat(OpenGL, WindowDC);

    bool32 ModernContext = true;
    HGLRC OpenGLRC = 0;
    if (wglCreateContextAttribsARB)
    {
        OpenGLRC = wglCreateContextAttribsARB(WindowDC, 0, Win32OpenGLAttribs);
    }

    if (!OpenGLRC)
    {
        ModernContext = false;
        OpenGLRC = wglCreateContext(WindowDC);
    }

    Assert(ModernContext);
    // TODO(Kevin): Logging / Messagebox to say hardware does not support required opengl version

    if (wglMakeCurrent(WindowDC, OpenGLRC))
    {
        // OpenGL rendering context ready

#define Win32GetOpenGLFunction(Name) OpenGL->Name = (type_##Name *)wglGetProcAddress(#Name)
        // opengl_info Info = OpenGLGetInfo(OpenGL, ModernContext);
        // if(Info.OpenGL_ARB_framebuffer_object)
        // {
        //     Win32GetOpenGLFunction(glBindFramebuffer);
        //     Win32GetOpenGLFunction(glGenFramebuffers);
        //     Win32GetOpenGLFunction(glFramebufferTexture2D);
        //     Win32GetOpenGLFunction(glCheckFramebufferStatus);
        // }

        // NOTE(Kevin): gl3w load procedures (https://github.com/skaslev/gl3w)
        if (gl3w_init())
        {
            // TODO(Kevin): Logging
            Assert(0);
        }
        // TODO(Kevin): Logging
        // LogMessage("GL_VERSION %s", glGetString(GL_VERSION));
        
        PlatformOpenGLSetVSync(OpenGL, true);
        OpenGLInit(OpenGL);
    }

    return OpenGL;
}

renderer *
Win32LoadRenderer(HDC WindowDC)
{
    renderer *Renderer = (renderer *)Win32InitOpenGL(WindowDC);
    return Renderer;
}

void
Win32BeginFrame(renderer *Renderer)
{
    OpenGLBeginFrame((open_gl *)Renderer);
}

void
Win32EndFrame(renderer *Renderer)
{
    OpenGLEndFrame((open_gl *)Renderer);
    SwapBuffers(wglGetCurrentDC());
}

#endif // SHOOTER_PLATFORM_WIN32
