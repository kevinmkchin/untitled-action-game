
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
                ASSERT(path.C_Str()[0] == '*'); // Assert texture is embedded into the binary

                int textureIndex = std::atoi(path.C_Str()+1); // Skip the '*' character
                
                ASSERT(textureIndex >= 0 && textureIndex < (int)scene->mNumTextures);

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
                    ASSERT(channelsInFile == 4);
                }

                ASSERT(rawPixelData);

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
