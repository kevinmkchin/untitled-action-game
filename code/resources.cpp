#include "resources.h"


asset_db_t Assets;


void FreeFileBinary(BinaryFileHandle& binary_file_to_free)
{
    if (binary_file_to_free.memory)
    {
        free(binary_file_to_free.memory);
        binary_file_to_free.memory = nullptr;
        binary_file_to_free.size = 0;
    }
}

void ReadFileBinary(BinaryFileHandle& mem_to_read_to, const char* file_path)
{
    if(mem_to_read_to.memory)
    {
        printf("WARNING: Binary File Handle already points to allocated memory. Freeing memory first...\n");
        FreeFileBinary(mem_to_read_to);
    }

    SDL_RWops* binary_file_rw = SDL_RWFromFile(file_path, "rb");
    if(binary_file_rw)
    {
        mem_to_read_to.size = (u32) SDL_RWsize(binary_file_rw); // total size in bytes
        mem_to_read_to.memory = malloc((size_t) mem_to_read_to.size);
        SDL_RWread(binary_file_rw, mem_to_read_to.memory, (size_t) mem_to_read_to.size, 1);
        SDL_RWclose(binary_file_rw);
    }
    else
    {
        printf("Failed to read %s! File doesn't exist.\n", file_path);
        return;
    }
}

bool WriteFileBinary(const BinaryFileHandle& bin, const char* file_path)
{
    if (bin.memory == NULL)
    {
        printf("WARNING: Binary File Handle does not point to any memory. Cancelled write to file operation.\n");
        return false;
    }

    SDL_RWops* bin_w = SDL_RWFromFile(file_path, "wb");
    if(bin_w)
    {
        SDL_RWwrite(bin_w, bin.memory, bin.size, 1);
        SDL_RWclose(bin_w);
        return true;
    }

    return false;
}

std::string ReadFileString(const char* file_path)
{
    std::string string_content;

    std::ifstream file_stream(file_path, std::ios::in);
    if (file_stream.is_open() == false)
    {
        printf("Failed to read %s! File doesn't exist.\n", file_path);
    }

    std::string line = "";
    while (file_stream.eof() == false)
    {
        std::getline(file_stream, line);
        string_content.append(line + "\n");
    }

    file_stream.close();

    return string_content;
}

void FreeImage(BitmapHandle& image_handle)
{
    FreeFileBinary(image_handle);
    image_handle.width = 0;
    image_handle.height = 0;
    image_handle.bitDepth = 0;
}

void ReadImage(BitmapHandle& image_handle, const char* image_file_path)
{
    if(image_handle.memory)
    {
        printf("WARNING: Binary File Handle already points to allocated memory. Freeing memory first...\n");
        FreeImage(image_handle);
    }

    stbi_set_flip_vertically_on_load(1);
    image_handle.memory = stbi_load(image_file_path, (int*)&image_handle.width, (int*)&image_handle.height, (int*)&image_handle.bitDepth, 0);
    if(image_handle.memory)
    {
        image_handle.size = image_handle.width * image_handle.height * image_handle.bitDepth;
    }
    else
    {
        printf("Failed to find image file at: %s\n", image_file_path);
        image_handle.width = 0;
        image_handle.height = 0;
        image_handle.bitDepth = 0;
        return;
    }
}


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
    glBufferData(GL_ARRAY_BUFFER, sizeInBytes, data, drawUsage);
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
        if (texture->id == 0)
            texId = Assets.DefaultMissingTexture.id;

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texId);
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
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) 4 * verticesArrayCount, vertices, drawUsage);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->idIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) 4 * indicesArrayCount, indices, drawUsage);
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



db_tex_t asset_db_t::GetTextureById(u32 persistId)
{
    auto textureiter = Textures.find(persistId);
    if (textureiter != Textures.end())
        return textureiter->second;
    
    db_tex_t missingTexture;
    missingTexture.persistId = persistId;
    missingTexture.gputex = DefaultMissingTexture;
    return missingTexture;
}

db_tex_t asset_db_t::LoadNewTexture(const char *path)
{
    db_tex_t tex;
    tex.persistId = ++TexturePersistIdCounter;

    BitmapHandle bitmapStorage;
    ReadImage(bitmapStorage, path);
    if (bitmapStorage.memory != NULL)
    {        
        CreateGPUTextureFromBitmap(&tex.gputex, (u8*)bitmapStorage.memory,
                        bitmapStorage.width, bitmapStorage.height,
                        GL_SRGB, (bitmapStorage.bitDepth == 3 ? GL_RGB : GL_RGBA),
                        GL_NEAREST_MIPMAP_LINEAR, GL_NEAREST, GL_UNSIGNED_BYTE);

        Assets.Textures.insert({tex.persistId, tex});
        LogMessage("Loaded %s with texture persist id %d", path, tex.persistId);
    }
    FreeImage(bitmapStorage);

    return tex;
}

void asset_db_t::CreateEntityBillboardAtlasForSupportRenderer(BitmapHandle *BillboardBitmaps)
{
    stbrp_rect *EntBilRects = NULL;
    for (int i = 0; i < 64; ++i)
    {
        BitmapHandle Bitmap = BillboardBitmaps[i];
        if (Bitmap.memory)
        {
            if (Bitmap.bitDepth != 4)
            {
                LogError("Entity billboard loading error: All entity billboard bitmaps must have RGBA components (32 bit depth).");
                continue;
            }
            stbrp_rect Rect;
            Rect.id = i; // entity_types_t
            Rect.w = Bitmap.width;
            Rect.h = Bitmap.height;
            arrput(EntBilRects, Rect);
        }
    }

    // Pack the rects
    i32 BillboardAtlasW = 512;
    i32 BillboardAtlasH = 512;
    stbrp_node *BillboardPackerNodes = NULL;
    arrsetlen(BillboardPackerNodes, BillboardAtlasW);
    stbrp_context BillboardPacker;
    stbrp_init_target(&BillboardPacker, BillboardAtlasW, BillboardAtlasH, 
        BillboardPackerNodes, (int)arrlenu(BillboardPackerNodes));
    stbrp_pack_rects(&BillboardPacker, EntBilRects, (int)arrlenu(EntBilRects));
    arrfree(BillboardPackerNodes);
    // EntBilRects is populated at this point

    // Take packed rect info; blit to BillboardAtlas and cache UV rect
    dynamic_array<u8> BillboardAtlasBuffer;
    BillboardAtlasBuffer.setlen(BillboardAtlasW * BillboardAtlasH * 4);
    memset(BillboardAtlasBuffer.data, 100, BillboardAtlasBuffer.lenu());
    for (int i = 0; i < arrlenu(EntBilRects); ++i)
    {
        stbrp_rect PackedRect = EntBilRects[i];
        if (!PackedRect.was_packed)
        {
            LogError("Entity billboard atlas packing error. Failed to pack a billboard.");
            continue;
        }

        BitmapHandle BillboardBitmap = BillboardBitmaps[PackedRect.id];

        BlitRect(BillboardAtlasBuffer.data, BillboardAtlasW, BillboardAtlasH,
            (u8*)BillboardBitmap.memory, PackedRect.w, PackedRect.h,
            PackedRect.x, PackedRect.y, sizeof(u8) * 4/*RGBA*/);

        vec4 *UVRect = &SupportRenderer.EntityBillboardRectMap[PackedRect.id];
        vec2 MinUV = vec2((float)(PackedRect.x) / (float)BillboardAtlasW, 
            (float)(PackedRect.y) / (float)BillboardAtlasH);
        vec2 MaxUV = vec2((float)(PackedRect.x + PackedRect.w) / (float)BillboardAtlasW, 
            (float)(PackedRect.y + PackedRect.h) / (float)BillboardAtlasH);
        UVRect->x = MinUV.x; // bottom left x
        UVRect->y = MinUV.y; // bottom left y
        UVRect->z = MaxUV.x; // top right x
        UVRect->w = MaxUV.y; // top right y

        float *BillboardWidth = &SupportRenderer.EntityBillboardWidthMap[PackedRect.id];
        *BillboardWidth = (float)PackedRect.w;
    }

    CreateGPUTextureFromBitmap(&SupportRenderer.EntityBillboardAtlas,
        BillboardAtlasBuffer.data, BillboardAtlasW, BillboardAtlasH,
        GL_RGBA, GL_RGBA);

    // Release all temporary buffers
    for (int i = 0; i < 64; ++i)
        if (BillboardBitmaps[i].memory)
            FreeImage(BillboardBitmaps[i]);
    arrfree(EntBilRects);
    BillboardAtlasBuffer.free();
}

void asset_db_t::LoadAllResources()
{
    LoadNewTexture(wd_path("default.png").c_str());
    LoadNewTexture(texture_path("t_bpav2.bmp").c_str());
    LoadNewTexture(texture_path("t_gf56464.bmp").c_str());
    LoadNewTexture(texture_path("t_hzdg.bmp").c_str());
    LoadNewTexture(texture_path("t_kgr2_p.bmp").c_str());
    LoadNewTexture(texture_path("t_mbrk2_1.bmp").c_str());
    LoadNewTexture(texture_path("t_vstnfcv.bmp").c_str());
    LoadNewTexture(texture_path("example_5.jpg").c_str());
    LoadNewTexture(texture_path("example_7.jpg").c_str());
    LoadNewTexture(texture_path("example_9.jpg").c_str());
    LoadNewTexture(texture_path("example_10.jpg").c_str());
    LoadNewTexture(texture_path("example_14.jpg").c_str());
    LoadNewTexture(texture_path("example_16.jpg").c_str());
    LoadNewTexture(texture_path("example_17.jpg").c_str());
    LoadNewTexture(texture_path("sld_gegfblock02b_64.jpg").c_str());
    LoadNewTexture(texture_path("example_19.jpg").c_str());
    LoadNewTexture(texture_path("example_20.jpg").c_str());

    DefaultEditorTexture = GetTextureById(1);
    CreateGPUTextureFromDisk(&DefaultMissingTexture, wd_path("missing_texture.png").c_str());

    // === Level entity billboards ===
    // Load all the billboard bitmaps
    BitmapHandle BillboardBitmaps[64];
    ReadImage(BillboardBitmaps[POINT_PLAYER_SPAWN], entity_icons_path("ent_bil_playerstart.png").c_str());
    ReadImage(BillboardBitmaps[POINT_LIGHT], entity_icons_path("ent_bil_pointlight.png").c_str());
    ReadImage(BillboardBitmaps[DIRECTIONAL_LIGHT_PROPERTIES], entity_icons_path("ent_bil_dlight.png").c_str());
    CreateEntityBillboardAtlasForSupportRenderer(BillboardBitmaps);
    

}