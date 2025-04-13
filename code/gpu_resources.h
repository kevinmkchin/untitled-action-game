#pragma once

#include "common.h"

struct GPUFrameBuffer
{
    u32 fbo;
    u32 colorTexId;
    u32 depthTexId;
    i32 width;
    i32 height;
};

void CreateGPUFrameBuffer(GPUFrameBuffer *buffer, GLenum TargetFormat = GL_RGBA, GLenum SourceFormat = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE);
void UpdateGPUFrameBufferSize(GPUFrameBuffer *buffer, i32 w, i32 h);
void DeleteGPUFrameBuffer(GPUFrameBuffer *buffer);


struct GPUMesh
{
    u32 idVAO = 0;
    u32 idVBO = 0;
    u32 vertexStride = 0;
    u32 vertexCount = 0;
};

// Allows up to three arbitrary float attributes in the vertex buffer
// It doesn't have to be position, texture, normal 
void CreateGPUMesh(GPUMesh *mesh,
                   u8 positionAttribSize = 3,
                   u8 textureAttribSize = 2,
                   u8 normalAttribSize = 3,
                   GLenum drawUsage = GL_STATIC_DRAW);
void RebindGPUMesh(GPUMesh *mesh, size_t sizeInBytes, float *data, GLenum drawUsage = GL_DYNAMIC_DRAW);
void RenderGPUMesh(u32 idVAO, u32 idVBO, u32 vertexCount, const struct GPUTexture *texture);
void DeleteGPUMesh(u32 idVAO, u32 idVBO);


struct GPUMeshIndexed
{
    u32 idVAO = 0;
    u32 idVBO = 0;
    u32 idIBO = 0;
    u32 indicesCount = 0;
};

void CreateGPUMeshIndexed(GPUMeshIndexed *mesh, 
                          float *vertices, 
                          u32 *indices, 
                          u32 verticesArrayCount, 
                          u32 indicesArrayCount, 
                          u8 positionAttribSize = 3, 
                          u8 textureAttribSize = 2, 
                          u8 normalAttribSize = 3, 
                          GLenum drawUsage = GL_DYNAMIC_DRAW);
void RebindGPUMeshIndexedData(GPUMeshIndexed *mesh, 
                              float *vertices, 
                              u32 *indices, 
                              u32 verticesArrayCount, 
                              u32 indicesArrayCount, 
                              GLenum drawUsage = GL_DYNAMIC_DRAW);
void RenderGPUMeshIndexed(GPUMeshIndexed mesh, GLenum rendermode = GL_TRIANGLES);
void DeleteGPUMeshIndexed(GPUMeshIndexed *mesh);


struct GPUTexture
{
    GLuint id = 0; // ID for the texture in GPU memory
    i32 width = 0;
    i32 height = 0;
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
    GLenum TargetFormat = GL_NONE; // internalformat
    GLenum SourceFormat = GL_NONE; // pixel data format
    GLenum PixelDataType = GL_NONE; // pixel data type
};

void CreateGPUTextureFromBitmap(GPUTexture        *texture,
                                void              *bitmap,
                                u32               bitmap_width,
                                u32               bitmap_height,
                                GLenum            target_format,
                                GLenum            source_format,
                                GLenum            min_filter_mode = GL_NEAREST,
                                GLenum            mag_filter_mode = GL_NEAREST,
                                GLenum            pixel_data_type = GL_UNSIGNED_BYTE);
void CreateGPUTextureFromDisk(GPUTexture *texture, const char* filePath, GLenum targetFormat = GL_RGBA);
void UpdateGPUTextureFromBitmap(GPUTexture *texture, unsigned char *bitmap, i32 w, i32 h);
void DeleteGPUTexture(GPUTexture *texture);


struct TripleBufferedSSBO
{
    void Init(size_t FrameChunkSizeBytes);
    void Destroy();

    void BeginFrame(); // wait for sync object
    // -> (pointer to write to, index, gpu offset)
    std::pair<void*, GLintptr> Alloc();
    void Bind(GLuint BindingPoint) const;
    void EndFrame(); // push new sync object

private:
    GLuint BufferObject = 0;
    void *MappedPtr = nullptr;

    static constexpr size_t NumFrames = 5;
    size_t FrameChunkSize = 0;
    size_t TotalSize = 0;
    u32 CurrentFrame = 0;
    GLsync FrameSyncObjects[NumFrames] = { nullptr, nullptr, nullptr };
};

