

void anim_model_t::SetVertexBoneDataToDefault(anim_vertex_t *Vertex)
{
    for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    {
        Vertex->BoneIDs[i] = -1;
        Vertex->BoneWeights[i] = 0.f;
    }
}

void anim_model_t::SetVertexBoneData(anim_vertex_t *Vertex, int BoneID, float Weight)
{
    for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    {
        if (Vertex->BoneIDs[i] < 0)
        {
            Vertex->BoneWeights[i] = Weight;
            Vertex->BoneIDs[i] = BoneID;
            break;
        }
    }
}

mat4 AssimpMatrixToColumnMajor(const aiMatrix4x4& from)
{
    mat4 to;
    //the a,b,c,d in assimp is the row ; the 1,2,3,4 is the column
    to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
    to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
    to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
    to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
    return to;
}

quat AssimpQuatToMyQuat(const aiQuaternion& pOrientation)
{
    return quat(pOrientation.w, pOrientation.x, pOrientation.y, pOrientation.z);
}

void anim_model_t::ExtractBoneWeightForVertices(dynamic_array<anim_vertex_t> Vertices, aiMesh* Mesh)
{
    for (u32 BoneIndex = 0; BoneIndex < Mesh->mNumBones; ++BoneIndex)
    {
        int BoneID = -1;
        std::string BoneName = Mesh->mBones[BoneIndex]->mName.C_Str();

        if (BoneInfoMap.find(BoneName) == BoneInfoMap.end())
        {
            bone_info_t BoneInfo;
            BoneInfo.Id = BoneCounter;
            BoneInfo.InverseBindPoseTransform = AssimpMatrixToColumnMajor(Mesh->mBones[BoneIndex]->mOffsetMatrix);
            BoneInfoMap[BoneName] = BoneInfo;
            BoneID = BoneCounter;
            ++BoneCounter;
        }
        else
        {
            // This bone also affects vertices outside of the scope of this mesh
            BoneID = BoneInfoMap[BoneName].Id;
        }
        ASSERT(BoneID != -1);

        // The influence weights of this bone, by vertex index
        aiVertexWeight *Weights = Mesh->mBones[BoneIndex]->mWeights;
        u32 NumWeights = Mesh->mBones[BoneIndex]->mNumWeights;

        for (u32 WeightIndex = 0; WeightIndex < NumWeights; ++WeightIndex)
        {
            int VertexIndex = Weights[WeightIndex].mVertexId;
            float Weight = Weights[WeightIndex].mWeight;
            ASSERT(VertexIndex <= Vertices.lenu());
            // This bone influences this vertex by this much
            SetVertexBoneData(&Vertices[VertexIndex], BoneID, Weight);
        }
    }
}

// Assimp mesh to GPU skeletal mesh
skeletal_mesh_t anim_model_t::ProcessMesh(aiMesh* Mesh)
{
    dynamic_array<anim_vertex_t> Vertices;
    dynamic_array<u32> Indices;
    Vertices.setlen(Mesh->mNumVertices);
    const u32 IndicesPerFace = Mesh->mFaces[0].mNumIndices;
    ASSERT(IndicesPerFace == 3); // faces are always triangles due to aiProcess_Triangulate
    Indices.setlen(Mesh->mNumFaces * IndicesPerFace); 

    // Extract vertices
    for (u32 i = 0; i < Mesh->mNumVertices; ++i)
    {
        anim_vertex_t& Vertex = Vertices[i];

        Vertex.Pos = vec3(Mesh->mVertices[i].x, Mesh->mVertices[i].y, Mesh->mVertices[i].z);

        if (Mesh->mTextureCoords[0])
            Vertex.Tex = vec2(Mesh->mTextureCoords[0][i].x, Mesh->mTextureCoords[0][i].y);
        else
            Vertex.Tex = vec2();

        Vertex.Norm = vec3(Mesh->mNormals[i].x, Mesh->mNormals[i].y, Mesh->mNormals[i].z);

        SetVertexBoneDataToDefault(&Vertex);
    }

    // Extract indices
    for (u32 i = 0; i < Mesh->mNumFaces; ++i)
    {
        const aiFace& Face = Mesh->mFaces[i];
        ASSERT(Face.mNumIndices == IndicesPerFace);
        for (u32 j = 0; j < Face.mNumIndices; ++j)
        {
            Indices[i * IndicesPerFace + j] = Face.mIndices[j];
        }
    }

    // Extract bone data
    ExtractBoneWeightForVertices(Vertices, Mesh);

    skeletal_mesh_t SkeletalMesh;
    CreateSkeletalMesh(&SkeletalMesh, Vertices, Indices);

    Vertices.free();
    Indices.free();

    return SkeletalMesh;
}


void CreateSkeletalMesh(skeletal_mesh_t *mesh,
    dynamic_array<anim_vertex_t> Vertices, 
    dynamic_array<u32> Indices)
{
    ASSERT(mesh->VAO == 0);

    GLsizei StrideInBytes = sizeof(anim_vertex_t);
    GLsizeiptr VertexBufferSzInBytes = (GLsizeiptr)sizeof(anim_vertex_t)*Vertices.lenu();
    GLsizeiptr IndexBufferSzInBytes = (GLsizeiptr)sizeof(u32)*Indices.lenu();

    mesh->IndicesCount = (u32)Indices.lenu();

    glGenVertexArrays(1, &mesh->VAO);
    glBindVertexArray(mesh->VAO);
    glGenBuffers(1, &mesh->VBO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->VBO);
    glBufferData(GL_ARRAY_BUFFER, VertexBufferSzInBytes, Vertices.data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, StrideInBytes, nullptr);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(anim_vertex_t, Tex));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(anim_vertex_t, Norm));

    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 4, GL_INT, StrideInBytes, (void*)offsetof(anim_vertex_t, BoneIDs));

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(anim_vertex_t, BoneWeights));

    glGenBuffers(1, &mesh->IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSzInBytes, Indices.data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


bool LoadAnimatedModel_GLTF2Bin(anim_model_t *Model, const char *FilePath)
{
    Assimp::Importer Importer;

    const aiScene *Scene = Importer.ReadFile(FilePath, aiProcess_Triangulate);

    if (!Scene || Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !Scene->mRootNode)
    {
        LogError("ASSIMP failed to load file at '%s'.\nErr msg: %s", FilePath, Importer.GetErrorString());
        return false;
    }


    dynamic_array<GPUTexture> matEmissiveTextures;

    for (u32 matIndex = 0; matIndex < Scene->mNumMaterials; ++matIndex)
    {
        aiMaterial *mat = Scene->mMaterials[matIndex];

        GPUTexture gputexEmissive;
        if (mat->GetTextureCount(aiTextureType_EMISSIVE))
        {
            aiString path;
            if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &path) == AI_SUCCESS)
            {
                ASSERT(path.C_Str()[0] == '*'); // Assert texture is embedded into the binary

                int textureIndex = std::atoi(path.C_Str()+1); // Skip the '*' character
                
                ASSERT(textureIndex >= 0 && textureIndex < (int)Scene->mNumTextures);

                aiTexture *texture = Scene->mTextures[textureIndex];

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
        matEmissiveTextures.put(gputexEmissive);
    }

    ASSERT(Scene->mNumMeshes > 0);
    ASSERT(Model->meshes == NULL);

    arrsetcap(Model->meshes, Scene->mNumMeshes);
    arrsetcap(Model->textures, Scene->mNumMeshes);

    for (u32 meshIndex = 0; meshIndex < Scene->mNumMeshes; ++meshIndex)
    {
        aiMesh* MeshNode = Scene->mMeshes[meshIndex];
        skeletal_mesh_t GPUSkeletalMesh = Model->ProcessMesh(MeshNode);
        u32 MaterialIndex = MeshNode->mMaterialIndex;
        GPUTexture ColorTex = matEmissiveTextures[MaterialIndex];

        arrput(Model->meshes, GPUSkeletalMesh);
        arrput(Model->textures, ColorTex);
    }

    matEmissiveTextures.free();


    animation_t *Animation = new animation_t();

    const bool TEMPFLAG_IsGLB = false;
    aiAnimation *AssimpAnim = Scene->mAnimations[0];
    // GLTF2.0 exports animation clip durations in MILLISECONDS!
    // Therefore Animation TicksPerSeconds must be set to 1000.
    Animation->DurationInTicks = (float)AssimpAnim->mDuration;
    Animation->TicksPerSecond = TEMPFLAG_IsGLB ? 1000.f : (float)AssimpAnim->mTicksPerSecond;
    Animation->ReadHierarchyData(Scene->mRootNode, Model);
    Animation->ReadBones(AssimpAnim, Model);

    Model->Animations.put(Animation);

    return true;
}

void anim_bone_t::Create(const aiNodeAnim *InChannel)
{
    Positions.setlen(InChannel->mNumPositionKeys);
    Rotations.setlen(InChannel->mNumRotationKeys);
    Scales.setlen(InChannel->mNumScalingKeys);

    for (u32 i = 0; i < InChannel->mNumPositionKeys; ++i)
    {
        aiVector3D Pos = InChannel->mPositionKeys[i].mValue;
        float Timestamp = (float)InChannel->mPositionKeys[i].mTime;
        
        keyframe_position_t *KeyData = &Positions[i];
        KeyData->Position = vec3(Pos.x, Pos.y, Pos.z);
        KeyData->Timestamp = Timestamp;
    }

    for (u32 i = 0; i < InChannel->mNumRotationKeys; ++i)
    {
        aiQuaternion Orientation = InChannel->mRotationKeys[i].mValue;
        float Timestamp = (float)InChannel->mRotationKeys[i].mTime;

        keyframe_rotation_t *KeyData = &Rotations[i];
        KeyData->Orientation = AssimpQuatToMyQuat(Orientation);
        KeyData->Timestamp = Timestamp;
    }

    for (u32 i = 0; i < InChannel->mNumScalingKeys; ++i)
    {
        aiVector3D Scale = InChannel->mScalingKeys[i].mValue;
        float Timestamp = (float)InChannel->mScalingKeys[i].mTime;
        
        keyframe_scale_t *KeyData = &Scales[i];
        KeyData->Scale = vec3(Scale.x, Scale.y, Scale.z);
        KeyData->Timestamp = Timestamp;
    }
}

void anim_bone_t::Delete()
{
    Positions.free();
    Rotations.free();
    Scales.free();
}

mat4 anim_bone_t::Update(float AnimationTime)
{
    mat4 Translation = InterpolatePosition(AnimationTime);
    mat4 Rotation = InterpolateRotation(AnimationTime);
    mat4 Scale = InterpolateScale(AnimationTime);
    mat4 CurrentLocalTransform = Translation * Rotation * Scale;
    return CurrentLocalTransform;
}

int anim_bone_t::GetPositionIndex(float AnimationTime)
{
    for (int i = 0; i < int(Positions.lenu()) - 1; ++i)
    {
        if (AnimationTime < Positions[i + 1].Timestamp)
            return i;
    }
    ASSERT(0);
    return 0;
}

int anim_bone_t::GetRotationIndex(float AnimationTime)
{
    for (int i = 0; i < int(Rotations.lenu()) - 1; ++i)
    {
        if (AnimationTime < Rotations[i + 1].Timestamp)
            return i;
    }
    ASSERT(0);
    return 0;
}

int anim_bone_t::GetScaleIndex(float AnimationTime)
{
    for (int i = 0; i < int(Scales.lenu()) - 1; ++i)
    {
        if (AnimationTime < Scales[i + 1].Timestamp)
            return i;
    }
    ASSERT(0);
    return 0;
}

float anim_bone_t::GetScaleFactor(float LastTimestamp, float NextTimestamp, float AnimationTime)
{
    float CurrentOffset = AnimationTime - LastTimestamp;
    float FramesDiff = NextTimestamp - LastTimestamp;
    return CurrentOffset / FramesDiff;
}

mat4 anim_bone_t::InterpolatePosition(float AnimationTime)
{
    if (Positions.lenu() == 1)
        return TranslationMatrix(Positions[0].Position);

    int P0Index = GetPositionIndex(AnimationTime);
    int P1Index = P0Index + 1;
    float ScaleFactor = GetScaleFactor(Positions[P0Index].Timestamp,
        Positions[P1Index].Timestamp, AnimationTime);
    
    vec3 FinalPosition = Lerp(Positions[P0Index].Position, Positions[P1Index].Position, ScaleFactor);
    return TranslationMatrix(FinalPosition);
}

mat4 anim_bone_t::InterpolateRotation(float AnimationTime)
{
    if (Rotations.lenu() == 1)
        return mat4(Normalize(Rotations[0].Orientation));

    int P0Index = GetRotationIndex(AnimationTime);
    int P1Index = P0Index + 1;
    float ScaleFactor = GetScaleFactor(Rotations[P0Index].Timestamp,
        Rotations[P1Index].Timestamp, AnimationTime);

    quat FinalRotation = Slerp(Rotations[P0Index].Orientation,
        Rotations[P1Index].Orientation, ScaleFactor);
    return mat4(Normalize(FinalRotation));
}

mat4 anim_bone_t::InterpolateScale(float AnimationTime)
{
    return mat4();

    if (Scales.lenu() == 1)
        return ScaleMatrix(Scales[0].Scale);

    int P0Index = GetScaleIndex(AnimationTime);
    int P1Index = P0Index + 1;
    float ScaleFactor = GetScaleFactor(Scales[P0Index].Timestamp,
        Scales[P1Index].Timestamp, AnimationTime);
    
    vec3 FinalScale = Lerp(Scales[P0Index].Scale, Scales[P1Index].Scale, ScaleFactor);
    return TranslationMatrix(FinalScale);
}

void animation_t::Destroy()
{
    // TODO iterate through bones flat list and delete

    // TODO destroy assimp_node_data_t
    // for (Dest.Name)

    // Dest.Name.clear();
    // Dest.ChildrenCount = 0;
}

void animation_t::ReadHierarchyData(const aiNode* Src, anim_model_t *Model)
{
    ASSERT(Src);

    const char *BoneName = Src->mName.data;

    if (Model->BoneInfoMap.find(BoneName) != Model->BoneInfoMap.end()) // is a bone
    {
        u8 ParentIndex = 0xFF;

        const char *ParentNodeName = Src->mParent->mName.data;
        bool ParentIsABone = Src->mParent && Model->BoneInfoMap.find(ParentNodeName) != Model->BoneInfoMap.end();
        if (ParentIsABone)
        {
            int ParentPaletteIndex = Model->BoneInfoMap[ParentNodeName].Id;
            anim_bone_t *ParentAnimBone = FindBone(ParentPaletteIndex);
            ASSERT(ParentAnimBone);
            ParentIndex = ParentAnimBone->SelfId;
        } 
        else
        {
            // otherwise, I am the root joint/bone therefore my parent index is 0xFF

            // This root bone may be nested deeper in the transform hierarchy
            // so we must find the transform from "skeleton or root joint space"
            // to actual "model space"
            // I can skip Src->mTransformation because the root bone's local transform for current pose
            // will be calculated from the animation clip
            mat4 AccumulateRootJointTransform = mat4();
            aiNode *Parent = Src->mParent;
            while (Parent)
            {
                AccumulateRootJointTransform *= AssimpMatrixToColumnMajor(Parent->mTransformation);
                Parent = Parent->mParent;
            }
            RootTransform = AccumulateRootJointTransform;
        }

        int PaletteIndex = Model->BoneInfoMap[BoneName].Id;
        u8 SelfId = (u8)Bones.size();
        // mat4 LocalTransformInBindPose = AssimpMatrixToColumnMajor(Src->mTransformation);
        // but i still need a sort of fallback matrix in case this bone is not animated

        anim_bone_t Bone;
        Bone.PaletteIndex = PaletteIndex;
        Bone.ParentId = ParentIndex;
        Bone.SelfId = SelfId;
        Bone.InverseBindPose = Model->BoneInfoMap[BoneName].InverseBindPoseTransform;
        // Bone.CurrentPoseTransform = _LocalTransformInBindPose;

        Bones.push_back(Bone);
    }

    for (u32 i = 0; i < Src->mNumChildren; ++i)
    {
        ReadHierarchyData(Src->mChildren[i], Model);
    }
}

void animation_t::ReadBones(const aiAnimation* AssimpAnim, anim_model_t *Model)
{
    // The number of bone animation channels. Each channel affects a single node.
    int NumChannels = AssimpAnim->mNumChannels;

    auto& boneInfoMap = Model->BoneInfoMap;//getting m_BoneInfoMap from Model class
    int& boneCount = Model->BoneCounter; //getting the m_BoneCounter from Model class

    // reading channels (bones engaged in an animation and their keyframes)
    for (int i = 0; i < NumChannels; ++i)
    {
        aiNodeAnim *NodeAnimChannel = AssimpAnim->mChannels[i];
        const char *BoneName = NodeAnimChannel->mNodeName.data;

        // check if missing this bone
        if (boneInfoMap.find(BoneName) == boneInfoMap.end())
        {
            ASSERT(0);
            // boneInfoMap[BoneName].Id = boneCount;
            // boneCount++;
        }

        int PaletteIndex = boneInfoMap[BoneName].Id;

        anim_bone_t *Bone = FindBone(PaletteIndex);
        ASSERT(Bone);
        Bone->Create(NodeAnimChannel);
    }
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
