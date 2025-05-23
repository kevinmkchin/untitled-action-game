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


struct triple_buffered_ssbo
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

    static constexpr size_t NumFrames = 3;
    size_t FrameChunkSize = 0;
    size_t TotalSize = 0;
    u32 CurrentFrame = 0;
    GLsync FrameSyncObjects[NumFrames] = { nullptr, nullptr, nullptr };
};

struct persistent_vertex_stream
{
    struct vertex_desc
    {
        size_t VByteSize; // and stride
        GLenum VAttrib0_Format;
        u32 VAttrib0_Size = 0; // number of attrib0 values per vertex
        u32 VAttrib0_Offset;
        GLenum VAttrib1_Format;
        u32 VAttrib1_Size = 0;
        u32 VAttrib1_Offset;
        GLenum VAttrib2_Format;
        u32 VAttrib2_Size = 0;
        u32 VAttrib2_Offset;
        // GLenum VAttrib3_Format;
        // size_t VAttrib3_Size = 0;
        // size_t VAttrib3_Offset;
        // GLenum VAttrib4_Format;
        // size_t VAttrib4_Size = 0;
        // size_t VAttrib4_Offset;
    };

    /*
        Triple-buffered persistent buffer for vertex streaming
        https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming#Persistent_mapped_streaming
    */

    void Alloc(size_t VertexCountPerFrame, vertex_desc VertexDescriptor);
    void Draw(void *VertexData, u32 VertexCount);
    void Free();

private:
    GLuint VAO = 0;
    GLuint VBO = 0;
    char* MappedPtr = nullptr;
    size_t VertexSize = 0;

    static constexpr GLuint BindingIndex = 0;

    static constexpr size_t NumFrames = 3;
    size_t CurrentFrame = 0;
    size_t FrameSize = 0;
    size_t TotalSize = 0;
};


