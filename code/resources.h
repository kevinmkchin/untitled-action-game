#pragma once


struct BinaryFileHandle
{
    /** Handle for a file in memory */
    u32     size   = 0;        // size of file in memory
    void*   memory = nullptr;  // pointer to file in memory
};

struct BitmapHandle : BinaryFileHandle
{
    /** Handle for an UNSIGNED BYTE bitmap in memory */
    u32 width    = 0;   // image width
    u32 height   = 0;   // image height
    u8  bitDepth = 0;   // bit depth of bitmap in bytes (e.g. bit depth = 3 means there are 3 bytes in the bitmap per pixel)
};


/** Allocates memory, stores the binary file data in memory, makes binary_file_handle_t.memory
    point to it. Pass along a binary_file_handle_t to receive the pointer to the file data in
    memory and the size in bytes. */
void ReadFileBinary(BinaryFileHandle& mem_to_read_to, const char* file_path);
void FreeFileBinary(BinaryFileHandle& binary_file_to_free);
bool WriteFileBinary(const BinaryFileHandle& bin, const char* file_path);

/** Returns the string content of a file as an std::string */
std::string ReadFileString(const char* file_path);

/** Allocates memory, loads an image file as an UNSIGNED BYTE bitmap, makes bitmap_handle_t.memory
    point to it. Pass along a bitmap_handle_t to receive the pointer to the bitmap in memory and
    bitmap information. */
void ReadImage(BitmapHandle& image_handle, const char* image_file_path);
void FreeImage(BitmapHandle& image_handle);


struct GPUFrameBuffer
{
    u32 fbo;
    u32 colorTexId;
    u32 depthTexId;
    i32 width;
    i32 height;
};

struct GPUMesh
{
    u32 idVAO = 0;
    u32 idVBO = 0;
    u32 vertexStride = 0;
    u32 vertexCount = 0;
};

struct GPUMeshIndexed
{
    u32 idVAO = 0;
    u32 idVBO = 0;
    u32 idIBO = 0;
    u32 indicesCount = 0;
};

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

void CreateGPUFrameBuffer(GPUFrameBuffer *buffer, GLenum TargetFormat = GL_RGBA, GLenum SourceFormat = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE);
void UpdateGPUFrameBufferSize(GPUFrameBuffer *buffer, i32 w, i32 h);
void DeleteGPUFrameBuffer(GPUFrameBuffer *buffer);

// Allows up to three arbitrary float attributes in the vertex buffer
// It doesn't have to be position, texture, normal 
void CreateGPUMesh(GPUMesh *mesh,
                   u8 positionAttribSize = 3,
                   u8 textureAttribSize = 2,
                   u8 normalAttribSize = 3,
                   GLenum drawUsage = GL_STATIC_DRAW);
void RebindGPUMesh(GPUMesh *mesh, size_t sizeInBytes, float *data, GLenum drawUsage = GL_DYNAMIC_DRAW);
void RenderGPUMesh(u32 idVAO, u32 idVBO, u32 vertexCount, const GPUTexture *texture);
void DeleteGPUMesh(u32 idVAO, u32 idVBO);

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


// MIXER
Mix_Chunk *Mixer_LoadChunk(const char *filepath);

