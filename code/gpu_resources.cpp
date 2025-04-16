#include "gpu_resources.h"
#include "resources.h"

void CreateGPUFrameBuffer(GPUFrameBuffer *buffer, GLenum TargetFormat, GLenum SourceFormat, GLenum type)
{
    buffer->fbo = 0;

    glGenTextures(1, &buffer->colorTexId);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, TargetFormat, buffer->width, buffer->height, 0, SourceFormat, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenTextures(1, &buffer->depthTexId);
    glBindTexture(GL_TEXTURE_2D, buffer->depthTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, buffer->width, buffer->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &buffer->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, buffer->fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer->colorTexId, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, buffer->depthTexId, 0);

    ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void UpdateGPUFrameBufferSize(GPUFrameBuffer *buffer, i32 w, i32 h)
{
    if (buffer->width == w && buffer->height == h) return;

    buffer->width = w;
    buffer->height = h;
    glBindFramebuffer(GL_FRAMEBUFFER, buffer->fbo);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, buffer->width, buffer->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, buffer->depthTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, buffer->width, buffer->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        fprintf(stderr, "Failed to change size of Internal FrameBuffer Object.");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DeleteGPUFrameBuffer(GPUFrameBuffer *buffer)
{
    glDeleteTextures(1, &buffer->colorTexId);
    glDeleteTextures(1, &buffer->depthTexId);
    glDeleteFramebuffers(1, &buffer->fbo);
}


void CreateGPUMesh(GPUMesh *mesh,
                   u8 positionAttribSize,
                   u8 textureAttribSize,
                   u8 normalAttribSize,
                   GLenum drawUsage)
{
    glGenVertexArrays(1, &mesh->idVAO);
    glBindVertexArray(mesh->idVAO);
    glGenBuffers(1, &mesh->idVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->idVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, drawUsage);

    u8 stride = positionAttribSize + textureAttribSize + normalAttribSize;
    glVertexAttribPointer(0, positionAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride, nullptr);
    glEnableVertexAttribArray(0);
    if (textureAttribSize > 0)
    {
        glVertexAttribPointer(1, textureAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride,
                              (void*)(sizeof(float) * positionAttribSize));
        glEnableVertexAttribArray(1);
        if (normalAttribSize > 0)
        {
            glVertexAttribPointer(2, normalAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride,
                                  (void*)(sizeof(float) * ((GLsizeiptr) positionAttribSize + textureAttribSize)));
            glEnableVertexAttribArray(2);
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    mesh->vertexStride = stride;
}

void DeleteGPUMesh(u32 idVAO, u32 idVBO)
{
    glDeleteBuffers(1, &idVBO);
    glDeleteVertexArrays(1, &idVAO);
}

void RebindGPUMesh(GPUMesh *mesh, size_t sizeInBytes, float *data, GLenum drawUsage)
{
    glBindVertexArray(mesh->idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->idVBO);
    // https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming
    glBufferData(GL_ARRAY_BUFFER, sizeInBytes, nullptr, drawUsage); // orphan old, alloc new buf
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeInBytes, data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (mesh->vertexStride)
        mesh->vertexCount = (u32) sizeInBytes / mesh->vertexStride;
}

void RenderGPUMesh(u32 idVAO, u32 idVBO, u32 vertexCount, const GPUTexture *texture)
{
    glBindVertexArray(idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, idVBO);

    if (texture)
    {
        GLuint texId = texture->id;
        if (texture->id > 0)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texId);
        }
    }

    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


void CreateGPUMeshIndexed(GPUMeshIndexed *mesh, 
                          float *vertices, 
                          u32 *indices, 
                          u32 verticesArrayCount, 
                          u32 indicesArrayCount, 
                          u8 positionAttribSize, 
                          u8 textureAttribSize, 
                          u8 normalAttribSize, 
                          GLenum drawUsage)
{
    ASSERT(mesh->idVAO == 0);

    u8 stride = 0;
    if (textureAttribSize)
    {
        stride += positionAttribSize + textureAttribSize;
        if (normalAttribSize)
        {
            stride += normalAttribSize;
        }
    }

    mesh->indicesCount = indicesArrayCount;

    glGenVertexArrays(1, &mesh->idVAO);
    glBindVertexArray(mesh->idVAO);
    glGenBuffers(1, &mesh->idVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->idVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) 4 /*bytes cuz float*/ * verticesArrayCount, vertices, drawUsage);
    glVertexAttribPointer(0, positionAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride, nullptr);
    glEnableVertexAttribArray(0);
    if (textureAttribSize > 0)
    {
        glVertexAttribPointer(1, textureAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride,
                              (void*)(sizeof(float) * positionAttribSize));
        glEnableVertexAttribArray(1);
        if (normalAttribSize > 0)
        {
            glVertexAttribPointer(2, normalAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride,
                                  (void*)(sizeof(float) * ((GLsizeiptr) positionAttribSize + textureAttribSize)));
            glEnableVertexAttribArray(2);
        }
    }

    glGenBuffers(1, &mesh->idIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->idIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) 4 /*bytes cuz uint32*/ * indicesArrayCount, indices, drawUsage);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0); // Unbind the VAO;
}

void RebindGPUMeshIndexedData(GPUMeshIndexed *mesh, 
                              float *vertices, 
                              u32 *indices, 
                              u32 verticesArrayCount, 
                              u32 indicesArrayCount, 
                              GLenum drawUsage)
{
    if (mesh->idVBO == 0 || mesh->idIBO == 0)
        return;

    mesh->indicesCount = indicesArrayCount;
    glBindVertexArray(mesh->idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->idVBO);
    // https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) 4 * verticesArrayCount, nullptr, drawUsage); // orphan old, alloc new buf
    glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr) 4 * verticesArrayCount, vertices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->idIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) 4 * indicesArrayCount, nullptr, drawUsage); // orphan old, alloc new buf
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, (GLsizeiptr) 4 * indicesArrayCount, indices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void RenderGPUMeshIndexed(GPUMeshIndexed mesh, GLenum rendermode)
{
    if (mesh.indicesCount == 0) // Early out if index_count == 0, nothing to draw
    {
        printf("WARNING: Attempting to Render a mesh with 0 index count!\n");
        return;
    }

    // Bind VAO, bind VBO, draw elements(indexed draw)
    glBindVertexArray(mesh.idVAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.idIBO);
    glDrawElements(rendermode, mesh.indicesCount, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void DeleteGPUMeshIndexed(GPUMeshIndexed *mesh)
{
    if (mesh->idIBO != 0)
    {
        glDeleteBuffers(1, &mesh->idIBO);
        mesh->idIBO = 0;
    }
    if (mesh->idVBO != 0)
    {
        glDeleteBuffers(1, &mesh->idVBO);
        mesh->idVBO = 0;
    }
    if (mesh->idVAO != 0)
    {
        glDeleteVertexArrays(1, &mesh->idVAO);
        mesh->idVAO = 0;
    }

    mesh->indicesCount = 0;
}


void CreateGPUTextureFromBitmap(GPUTexture        *texture,
                                void              *bitmap,
                                u32               bitmap_width,
                                u32               bitmap_height,
                                GLenum            target_format,
                                GLenum            source_format,
                                GLenum            min_filter_mode,
                                GLenum            mag_filter_mode,
                                GLenum            pixel_data_type)
{
    if (bitmap == NULL)
    {
        LogError("CreateGPUTextureFromBitmap error: provided bitmap is null.");
        return;
    }

    ASSERT(texture->id == 0);

    texture->width = bitmap_width;
    texture->height = bitmap_height;
    texture->TargetFormat = target_format;
    texture->SourceFormat = source_format;
    texture->PixelDataType = pixel_data_type;

    glGenTextures(1, &texture->id);   // generate texture and grab texture id
    glBindTexture(GL_TEXTURE_2D, texture->id);
    glTexImage2D(
            GL_TEXTURE_2D,            // texture target type
            0,                        // level-of-detail number n = n-th mipmap reduction image
            target_format,            // format of data to store (target): num of color components
            bitmap_width,             // texture width
            bitmap_height,            // texture height
            0,                        // must be 0 (legacy)
            source_format,            // format of data being loaded (source)
            pixel_data_type,          // data type of the texture data
            bitmap);                  // data
    if (min_filter_mode == GL_NEAREST_MIPMAP_LINEAR || 
        min_filter_mode == GL_NEAREST_MIPMAP_NEAREST ||
        min_filter_mode == GL_LINEAR_MIPMAP_NEAREST ||
        min_filter_mode == GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter_mode); // filtering (e.g. GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter_mode);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void CreateGPUTextureFromDisk(GPUTexture *texture, const char* filePath, GLenum targetFormat)
{
    BitmapHandle textureBitmapHandle;
    ReadImage(textureBitmapHandle, filePath);
    if (textureBitmapHandle.memory == nullptr)
        return;

    CreateGPUTextureFromBitmap(texture,
                               (void*) textureBitmapHandle.memory,
                               textureBitmapHandle.width,textureBitmapHandle.height,
                               targetFormat,
                               (textureBitmapHandle.bitDepth == 3 ? GL_RGB : GL_RGBA));
    FreeImage(textureBitmapHandle); // texture data has been copied to GPU memory, so we can free image from memory
}

void UpdateGPUTextureFromBitmap(GPUTexture *texture, unsigned char *bitmap, i32 w, i32 h)
{
    ASSERT(texture->id != 0);

    texture->width = w;
    texture->height = h;

    glBindTexture(GL_TEXTURE_2D, texture->id);
    glTexImage2D(
        GL_TEXTURE_2D,            // texture target type
        0,                        // level-of-detail number n = n-th mipmap reduction image
        texture->TargetFormat,    // format of data to store (target): num of color components
        w,                        // texture width
        h,                        // texture height
        0,                        // must be 0 (legacy)
        texture->SourceFormat,    // format of data being loaded (source)
        texture->PixelDataType,   // data type of the texture data
        bitmap);                  // data
    glBindTexture(GL_TEXTURE_2D, 0);
}

void DeleteGPUTexture(GPUTexture *texture)
{
    if (texture->id == 0)
        return;

    glDeleteTextures(1, &texture->id);

    texture->id = 0;
    texture->width = 0;
    texture->height = 0;
    texture->TargetFormat = GL_NONE;
    texture->SourceFormat = GL_NONE;
    texture->PixelDataType = GL_NONE;
}


void triple_buffered_ssbo::Init(size_t FrameChunkSizeBytes)
{
    FrameChunkSize = FrameChunkSizeBytes;
    TotalSize = FrameChunkSize * NumFrames;

    glGenBuffers(1, &BufferObject);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferObject);

    // GLint alignment = 0;
    // glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment);
    // alignment_ = static_cast<size_t>(alignment);

    glBufferStorage(GL_SHADER_STORAGE_BUFFER, TotalSize, 0,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

    MappedPtr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, TotalSize,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

    if (!MappedPtr) 
    {
        throw std::runtime_error("Failed to map SSBO persistently");
    }
}

void triple_buffered_ssbo::Destroy()
{
    if (BufferObject)
    {
        glDeleteBuffers(1, &BufferObject);
        BufferObject = 0;
        MappedPtr = nullptr;
    }
}

void triple_buffered_ssbo::BeginFrame()
{
    CurrentFrame = (CurrentFrame + 1) % NumFrames;

    // See note in EndFrame
    // // When a fence becomes signaled, you can be certain that the GPU has completed all 
    // // OpenGL commands issued before the fence was created.
    // GLenum Status = GL_UNSIGNALED;
    // while (Status != GL_ALREADY_SIGNALED && Status != GL_CONDITION_SATISFIED && FrameSyncObjects[CurrentFrame] != nullptr)
    // {
    //     Status = glClientWaitSync(FrameSyncObjects[CurrentFrame], GL_SYNC_FLUSH_COMMANDS_BIT, 1);
    // }
}

std::pair<void *, GLintptr> triple_buffered_ssbo::Alloc()
{
    size_t GlobalOffset = CurrentFrame * FrameChunkSize;
    void *CPUPtr = static_cast<char*>(MappedPtr) + GlobalOffset;
    GLintptr GPUOffset = static_cast<GLintptr>(GlobalOffset);
    return { CPUPtr, GPUOffset };
}

void triple_buffered_ssbo::Bind(GLuint BindingPoint) const
{
    // This is pretty slow (3000 fps -> 2000 fps)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BindingPoint, BufferObject);
}

void triple_buffered_ssbo::EndFrame()
{
    // NOTE(Kevin) 2025-03-31: Maybe this is a bug on my GPU, but calling glFenceSync
    //      is leaking memory. In Task Manager and Visual Studio can observe memory rising
    //      every frame this is called even if glDeleteSync is also called...
    //      hopefully this issue doesn't persist when I port to Vulkan.
    //      ACTUALLY when VSYNC is ON it does not leak memory. Which makes me think it's something
    //      to do with the CPU submitting jobs too quickly maybe. ChatGPT says this is expected on
    //      NVIDIA cards it is a backpressure problem. Hopefully the issue goes away when the CPU
    //      is taking longer to complete the frame and CPU GPU work is more balanced.
    // glDeleteSync(FrameSyncObjects[CurrentFrame]);
    // FrameSyncObjects[CurrentFrame] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void persistent_vertex_stream::Alloc(
    size_t VertexCountPerFrame,
    vertex_desc VertexDescriptor)
{
    ASSERT(!(VAO || VBO));

    VertexSize = VertexDescriptor.VByteSize;
    FrameSize = VertexCountPerFrame * VertexSize;
    TotalSize = NumFrames * FrameSize;

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    const GLsizei StrideInBytes = (GLsizei)VertexSize;
    if (VertexDescriptor.VAttrib0_Size > 0)
    {
        ASSERT(VertexDescriptor.VAttrib0_Offset == 0);
        glEnableVertexAttribArray(0);
        glVertexAttribFormat(0, 
            VertexDescriptor.VAttrib0_Size, 
            VertexDescriptor.VAttrib0_Format, 
            GL_FALSE,
            VertexDescriptor.VAttrib0_Offset);
        glVertexAttribBinding(0, BindingIndex);
    }
    if (VertexDescriptor.VAttrib1_Size > 0)
    {
        glEnableVertexAttribArray(1);
        glVertexAttribFormat(1, 
            VertexDescriptor.VAttrib1_Size, 
            VertexDescriptor.VAttrib1_Format, 
            GL_FALSE,
            VertexDescriptor.VAttrib1_Offset);
        glVertexAttribBinding(1, BindingIndex);
    }
    if (VertexDescriptor.VAttrib2_Size > 0)
    {
        glEnableVertexAttribArray(2);
        glVertexAttribFormat(2, 
            VertexDescriptor.VAttrib2_Size, 
            VertexDescriptor.VAttrib2_Format, 
            GL_FALSE,
            VertexDescriptor.VAttrib2_Offset);
        glVertexAttribBinding(2, BindingIndex);
    }

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferStorage(GL_ARRAY_BUFFER, TotalSize, nullptr,
        GL_MAP_WRITE_BIT |
        GL_MAP_PERSISTENT_BIT |
        GL_MAP_COHERENT_BIT);
    MappedPtr = (char*) glMapBufferRange(GL_ARRAY_BUFFER, 0, TotalSize,
        GL_MAP_WRITE_BIT |
        GL_MAP_PERSISTENT_BIT |
        GL_MAP_COHERENT_BIT);
    // Tell OpenGL how to interpret the vertex struct (binding index = 0)
    // Bind buffer to binding index 0 (no offset yet)
    glBindVertexBuffer(BindingIndex, VBO, 0, StrideInBytes);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void persistent_vertex_stream::Draw(void *VertexData, u32 VertexCount)
{
    if (VertexCount == 0)
        return;
    ASSERT(VAO && VBO);

    size_t FrameIndex = CurrentFrame % NumFrames;
    size_t OffsetInBytes = FrameIndex * FrameSize;
    const GLsizei StrideInBytes = (GLsizei)VertexSize;

    memcpy(MappedPtr + OffsetInBytes, VertexData, VertexCount * StrideInBytes);

    glBindVertexArray(VAO);
    glBindVertexBuffer(BindingIndex, VBO, OffsetInBytes, StrideInBytes);
    glDrawArrays(GL_TRIANGLES, 0, VertexCount);
    glBindVertexArray(0);

    ++CurrentFrame;
    if (CurrentFrame >= FrameSize)
        CurrentFrame = 0;
}

void persistent_vertex_stream::Free()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}


