
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




void FreeModelGLTF(ModelGLTF model)
{
    ASSERT(arrlen(model.meshes) == arrlen(model.color));

    size_t meshcount = arrlenu(model.meshes);
    for (size_t i = 0; i < meshcount; ++i)
    {
        DeleteGPUMeshIndexed(&model.meshes[i]);
        DeleteGPUTexture(&model.color[i]);
    }

    arrfree(model.meshes);
    arrfree(model.color);
}

void RenderModelGLTF(ModelGLTF model)
{
    size_t meshcount = arrlenu(model.meshes);
    for (size_t i = 0; i < meshcount; ++i)
    {
        GPUMeshIndexed m = model.meshes[i];
        GPUTexture t = model.color[i];

        // TODO(Kevin): if t.id is 0 then bind MissingTexture 
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, t.id);

        RenderGPUMeshIndexed(m);
    }
}

static GPUMeshIndexed ASSIMPMeshToGPUMeshIndexed(aiMesh* meshNode)
{
    const u32 vertexStride = 8;

    std::vector<float> vb(meshNode->mNumVertices * vertexStride);
    std::vector<u32> ib(meshNode->mNumFaces * meshNode->mFaces[0].mNumIndices);
    if (meshNode->mTextureCoords[0])
    {
        for (size_t i = 0; i < meshNode->mNumVertices; ++i)
        {
            // mNormals and mVertices are both mNumVertices in size
            size_t v_start_index = i * vertexStride;
            vb[v_start_index] = meshNode->mVertices[i].x;
            vb[v_start_index + 1] = meshNode->mVertices[i].y;
            vb[v_start_index + 2] = meshNode->mVertices[i].z;
            vb[v_start_index + 3] = meshNode->mTextureCoords[0][i].x;
            vb[v_start_index + 4] = meshNode->mTextureCoords[0][i].y;
            vb[v_start_index + 5] = meshNode->mNormals[i].x;
            vb[v_start_index + 6] = meshNode->mNormals[i].y;
            vb[v_start_index + 7] = meshNode->mNormals[i].z;
        }
    }
    else
    {
        for (size_t i = 0; i < meshNode->mNumVertices; ++i)
        {
            size_t v_start_index = i * vertexStride;
            vb[v_start_index] = meshNode->mVertices[i].x;
            vb[v_start_index + 1] = meshNode->mVertices[i].y;
            vb[v_start_index + 2] = meshNode->mVertices[i].z;
            vb[v_start_index + 3] = 0.f;
            vb[v_start_index + 4] = 0.f;
            vb[v_start_index + 5] = meshNode->mNormals[i].x;
            vb[v_start_index + 6] = meshNode->mNormals[i].y;
            vb[v_start_index + 7] = meshNode->mNormals[i].z;
        }
    }

    for (size_t i = 0; i < meshNode->mNumFaces; ++i)
    {
        aiFace face = meshNode->mFaces[i];
        for (size_t j = 0; j < face.mNumIndices; ++j)
        {
            ib[i * face.mNumIndices + j] = face.mIndices[j];
        }
    }

    GPUMeshIndexed mesh;
    CreateGPUMeshIndexed(&mesh, &vb[0], &ib[0], (u32)vb.size(), (u32)ib.size());
    return mesh;
}

bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath)
{
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(filepath,
        aiProcess_Triangulate);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        LogError("ASSIMP failed to load file at '%s'.\nErr msg: %s", filepath, importer.GetErrorString());
        return false;
    }

    // == Load and create all the textures ==
    // For each material, I'm only going to load one texture per texture type.
    // I'm only going to support embedded textures (GLB).
    // Assume 8-bit per channel RGBA format for texture input format

    GPUTexture *matEmissiveTextures = NULL;

    for (u32 matIndex = 0; matIndex < scene->mNumMaterials; ++matIndex)
    {
        aiMaterial *mat = scene->mMaterials[matIndex];

        GPUTexture gputexEmissive;
        if (mat->GetTextureCount(aiTextureType_EMISSIVE))
        {
            aiString path;
            if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &path) == AI_SUCCESS)
            {
                ALWAYSASSERT(path.C_Str()[0] == '*'); // Assert texture is embedded into the binary

                int textureIndex = std::atoi(path.C_Str()+1); // Skip the '*' character
                
                ALWAYSASSERT(textureIndex >= 0 && textureIndex < (int)scene->mNumTextures);

                aiTexture *texture = scene->mTextures[textureIndex];

                bool compressed = texture->mHeight == 0;
                void *rawPixelData = (void*)texture->pcData;
                i32 width = texture->mWidth; // Width is stored in mWidth for uncompressed
                i32 height = texture->mHeight; // Height is stored in mHeight for uncompressed

                // Uncompress if compressed format (e.g. PNG / JPG)
                if (compressed)
                {
                    u8 *compressedImageData = (u8*)texture->pcData;
                    i32 channelsInFile;
                    rawPixelData = (void*)stbi_load_from_memory(compressedImageData, texture->mWidth, &width, &height, &channelsInFile, 0);
                    ALWAYSASSERT(channelsInFile == 4);
                }

                ALWAYSASSERT(rawPixelData);

                CreateGPUTextureFromBitmap(&gputexEmissive, rawPixelData, width, height, 
                    GL_RGBA, GL_RGBA, GL_NEAREST, GL_NEAREST);

                if (compressed)
                {
                    stbi_image_free(rawPixelData);
                }
            }
        }
        arrput(matEmissiveTextures, gputexEmissive);
    }

    ASSERT(scene->mNumMeshes > 0);
    ASSERT(model->meshes == NULL);

    arrsetcap(model->meshes, scene->mNumMeshes);
    arrsetcap(model->color, scene->mNumMeshes);

    for (u32 meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
    {
        aiMesh* meshNode = scene->mMeshes[meshIndex];
        GPUMeshIndexed gpumesh = ASSIMPMeshToGPUMeshIndexed(meshNode);
        u32 matIndex = meshNode->mMaterialIndex;
        GPUTexture colorTex = matEmissiveTextures[matIndex];

        arrput(model->meshes, gpumesh);
        arrput(model->color, colorTex);
    }

    arrfree(matEmissiveTextures);

    return true;
}
