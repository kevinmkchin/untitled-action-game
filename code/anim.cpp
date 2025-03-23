#include "anim.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


static mat4 AssimpMatrixToColumnMajor(const aiMatrix4x4& from)
{
    mat4 to;
    //the a,b,c,d in assimp is the row ; the 1,2,3,4 is the column
    to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
    to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
    to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
    to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
    return to;
}

static quat AssimpQuatToMyQuat(const aiQuaternion& pOrientation)
{
    return quat(pOrientation.w, pOrientation.x, pOrientation.y, pOrientation.z);
}

static void SetVertexBoneDataToDefault(skinned_vertex_t *Vertex)
{
    for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    {
        Vertex->BoneIDs[i] = -1;
        Vertex->BoneWeights[i] = 0.f;
    }
}

static void SetVertexBoneData(skinned_vertex_t *Vertex, int BoneIndex, float Weight)
{
    for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    {
        if (Vertex->BoneIDs[i] < 0)
        {
            Vertex->BoneWeights[i] = Weight;
            Vertex->BoneIDs[i] = BoneIndex;
            break;
        }
    }
}

static void ExtractBoneWeightForVertices(dynamic_array<skinned_vertex_t> Vertices, aiMesh* Mesh,
    const skeleton_t *Skeleton)
{
    for (u32 i = 0; i < Mesh->mNumBones; ++i)
    {
        const char *BoneName = Mesh->mBones[i]->mName.C_Str();
        auto Iter = Skeleton->JointNameToIndex.find(BoneName);
        ASSERT(Iter != Skeleton->JointNameToIndex.end());
        int JointIndex = Iter->second;

        // The influence weights of this bone, by vertex index
        aiVertexWeight *Weights = Mesh->mBones[i]->mWeights;
        u32 NumWeights = Mesh->mBones[i]->mNumWeights;

        for (u32 WeightIndex = 0; WeightIndex < NumWeights; ++WeightIndex)
        {
            int VertexIndex = Weights[WeightIndex].mVertexId;
            float Weight = Weights[WeightIndex].mWeight;
            ASSERT(VertexIndex <= Vertices.lenu());
            SetVertexBoneData(&Vertices[VertexIndex], JointIndex, Weight);
        }
    }
}

static void CreateSkinnedMesh(skinned_mesh_t *mesh,
    dynamic_array<skinned_vertex_t> Vertices, 
    dynamic_array<u32> Indices)
{
    ASSERT(mesh->VAO == 0);

    GLsizei StrideInBytes = sizeof(skinned_vertex_t);
    GLsizeiptr VertexBufferSzInBytes = (GLsizeiptr)sizeof(skinned_vertex_t)*Vertices.lenu();
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
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(skinned_vertex_t, Tex));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(skinned_vertex_t, Norm));

    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 4, GL_INT, StrideInBytes, (void*)offsetof(skinned_vertex_t, BoneIDs));

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)offsetof(skinned_vertex_t, BoneWeights));

    glGenBuffers(1, &mesh->IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSzInBytes, Indices.data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

static skinned_mesh_t ProcessSkinnedMesh(aiMesh* Mesh, const skeleton_t *Skeleton)
{
    dynamic_array<skinned_vertex_t> Vertices;
    dynamic_array<u32> Indices;
    Vertices.setlen(Mesh->mNumVertices);
    const u32 IndicesPerFace = Mesh->mFaces[0].mNumIndices;
    ASSERT(IndicesPerFace == 3); // faces are always triangles due to aiProcess_Triangulate
    Indices.setlen(Mesh->mNumFaces * IndicesPerFace); 

    // Extract vertices
    for (u32 i = 0; i < Mesh->mNumVertices; ++i)
    {
        skinned_vertex_t& Vertex = Vertices[i];

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
    ExtractBoneWeightForVertices(Vertices, Mesh, Skeleton);

    // Create the mesh resource on GPU
    skinned_mesh_t SkinnedMesh;
    CreateSkinnedMesh(&SkinnedMesh, Vertices, Indices);

    Vertices.free();
    Indices.free();

    return SkinnedMesh;
}


static void ReadSkeletonJoints(const aiNode *Node, 
    std::unordered_map<std::string, mat4>& BoneLookup,
    skeleton_t *Skeleton, dynamic_array<skeleton_joint_t>& OutSkeletonJoints)
{
    ASSERT(Node);
    ASSERT(Skeleton);

    const char *BoneName = Node->mName.data;
    if (BoneLookup.find(BoneName) != BoneLookup.end()) // is a bone
    {
        skeleton_joint_t Joint;

        Joint.InverseBindPoseTransform = BoneLookup[BoneName];
        Joint.ParentIndex = 0xFF;

        if (Node->mParent)
        {
            const char *ParentNodeName = Node->mParent->mName.data;
            if (BoneLookup.find(ParentNodeName) != BoneLookup.end())
            {
                int ParentIndex = Skeleton->JointNameToIndex[ParentNodeName]; // parent is a bone too
                Joint.ParentIndex = ParentIndex;
            }
        }

        OutSkeletonJoints.put(Joint);
        Skeleton->JointNameToIndex[BoneName] = (int)OutSkeletonJoints.lenu() - 1;
    }

    for (u32 i = 0; i < Node->mNumChildren; ++i)
        ReadSkeletonJoints(Node->mChildren[i], BoneLookup, Skeleton, OutSkeletonJoints);
}

static void ReadAnimationClip(const aiAnimation *AssimpAnim, animation_clip_t *OutClip)
{
    const bool TEMPFLAG_IsGLB = true;
    // GLTF2.0 exports animation clip durations in MILLISECONDS!
    // Therefore Animation TicksPerSeconds must be set to 1000.
    OutClip->DurationInTicks = (float)AssimpAnim->mDuration;
    OutClip->TicksPerSecond = TEMPFLAG_IsGLB ? 1000.f : (float)AssimpAnim->mTicksPerSecond;
    // The number of bone animation channels. Each channel affects a single node.
    u32 NumChannels = AssimpAnim->mNumChannels;
    OutClip->JointPoseSamplers = mem_indexer<joint_pose_sampler_t>(
        StaticGameMemory.Alloc(NumChannels*sizeof(joint_pose_sampler_t), 
        alignof(joint_pose_sampler_t)), NumChannels);
    ASSERT(NumChannels == (u32)OutClip->GetSkeleton()->Joints.count);
    for (size_t i = 0; i < OutClip->JointPoseSamplers.count; ++i)
        OutClip->JointPoseSamplers[i].Positions.data = NULL;

    // Reading channels (bones engaged in an animation and their keyframes)
    for (u32 i = 0; i < NumChannels; ++i)
    {
        aiNodeAnim *NodeAnimChannel = AssimpAnim->mChannels[i];
        const char *JointName = NodeAnimChannel->mNodeName.data;

        auto Iter = OutClip->GetSkeleton()->JointNameToIndex.find(JointName);
        ASSERT(Iter != OutClip->GetSkeleton()->JointNameToIndex.end());
        int JointIndex = Iter->second;

        joint_pose_sampler_t *JointPoseSampler = &OutClip->JointPoseSamplers[JointIndex];
        ASSERT(JointPoseSampler->Positions.data == NULL);
        JointPoseSampler->Create(NodeAnimChannel);
    }
}

bool LoadSkeleton_GLTF2Bin(const char *InFilePath, skeleton_t *OutSkeleton)
{
    Assimp::Importer Importer;
    const aiScene *Scene = Importer.ReadFile(InFilePath, aiProcess_Triangulate);
    if (!Scene || Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !Scene->mRootNode)
    {
        LogError("ASSIMP failed to load file at '%s'.\nErr msg: %s", InFilePath, Importer.GetErrorString());
        return false;
    }

    ASSERT(OutSkeleton->Joints.data == NULL);
    OutSkeleton->JointNameToIndex.clear();
    // Theres several ways I could get the set of bone nodes. If this file has meshes,
    // then I can just look at the aiMeshes, but if in the future I want to have a 
    // file that is purely just the armature, then I could try to use aiNode::mMetaData.

    std::unordered_map<std::string, mat4> BoneNameToInverseBindPose;

    for (u32 meshIndex = 0; meshIndex < Scene->mNumMeshes; ++meshIndex)
    {
        aiMesh* MeshNode = Scene->mMeshes[meshIndex];

        for (u32 BoneIndex = 0; BoneIndex < MeshNode->mNumBones; ++BoneIndex)
        {
            std::string BoneName = MeshNode->mBones[BoneIndex]->mName.C_Str();
            if (BoneNameToInverseBindPose.find(BoneName) == BoneNameToInverseBindPose.end())
                BoneNameToInverseBindPose[BoneName] = AssimpMatrixToColumnMajor(MeshNode->mBones[BoneIndex]->mOffsetMatrix);
        }
    }

    // Read skeleton data
    dynamic_array<skeleton_joint_t> SkeletonJoints;
    SkeletonJoints.setlen(64);
    SkeletonJoints.setlen(0);
    ReadSkeletonJoints(Scene->mRootNode, BoneNameToInverseBindPose, OutSkeleton, SkeletonJoints);
    void *MemDst = StaticGameMemory.Alloc(SkeletonJoints.lenu()*sizeof(skeleton_joint_t), alignof(skeleton_joint_t));
    memcpy(MemDst, SkeletonJoints.data, SkeletonJoints.lenu()*sizeof(skeleton_joint_t));
    OutSkeleton->Joints = mem_indexer<skeleton_joint_t>(MemDst, SkeletonJoints.lenu());
    SkeletonJoints.free();

    // Read animation clip data
    for (u32 i = 0; i < Scene->mNumAnimations; ++i)
    {
        aiAnimation *AssimpAnim = Scene->mAnimations[i];
        void *Address = StaticGameMemory.Alloc<animation_clip_t>();
        animation_clip_t *Clip = new(Address) animation_clip_t(OutSkeleton);
        ReadAnimationClip(AssimpAnim, Clip);
        OutSkeleton->Clips.put(Clip);
    }

    return true;
}


void LoadAdditionalAnimationsForSkeleton(struct skeleton_t *Skeleton, const char *InFilePath)
{
    Assimp::Importer Importer;
    const aiScene *Scene = Importer.ReadFile(InFilePath, aiProcess_Triangulate);
    if (!Scene || Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !Scene->mRootNode)
    {
        LogError("ASSIMP failed to load file at '%s'.\nErr msg: %s", InFilePath, Importer.GetErrorString());
        return;
    }

    ASSERT(Skeleton->Joints.data != NULL);
    // ensure this file has the same skeleton as the existing one
    for (u32 meshIndex = 0; meshIndex < Scene->mNumMeshes; ++meshIndex)
    {
        aiMesh* MeshNode = Scene->mMeshes[meshIndex];
        for (u32 BoneIndex = 0; BoneIndex < MeshNode->mNumBones; ++BoneIndex)
        {
            std::string BoneName = MeshNode->mBones[BoneIndex]->mName.C_Str();
            ASSERT(Skeleton->JointNameToIndex.find(BoneName) != Skeleton->JointNameToIndex.end());
        }
    }

    // Read animation clip data
    for (u32 i = 0; i < Scene->mNumAnimations; ++i)
    {
        aiAnimation *AssimpAnim = Scene->mAnimations[i];
        void *Address = StaticGameMemory.Alloc<animation_clip_t>();
        animation_clip_t *Clip = new(Address) animation_clip_t(Skeleton);
        ReadAnimationClip(AssimpAnim, Clip);
        Skeleton->Clips.put(Clip);
    }
}

bool LoadSkinnedModel_GLTF2Bin(const char* InFilePath, skinned_model_t *OutSkinnedModel)
{
    Assimp::Importer Importer;
    const aiScene *Scene = Importer.ReadFile(InFilePath, aiProcess_Triangulate);
    if (!Scene || Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !Scene->mRootNode)
    {
        LogError("ASSIMP failed to load file at '%s'.\nErr msg: %s", InFilePath, Importer.GetErrorString());
        return false;
    }

    ASSERT(OutSkinnedModel->GetSkeleton());

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
    ASSERT(OutSkinnedModel->Meshes.data == NULL);
    ASSERT(OutSkinnedModel->Textures.data == NULL);
    void *Meshes = StaticGameMemory.Alloc(Scene->mNumMeshes*sizeof(skinned_mesh_t), alignof(skinned_mesh_t));
    void *Textures = StaticGameMemory.Alloc(Scene->mNumMeshes*sizeof(GPUTexture), alignof(GPUTexture));
    OutSkinnedModel->Meshes = mem_indexer<skinned_mesh_t>(Meshes, Scene->mNumMeshes);
    OutSkinnedModel->Textures = mem_indexer<GPUTexture>(Textures, Scene->mNumMeshes);

    for (u32 MeshIndex = 0; MeshIndex < Scene->mNumMeshes; ++MeshIndex)
    {
        aiMesh* MeshNode = Scene->mMeshes[MeshIndex];
        skinned_mesh_t SkinnedMesh = ProcessSkinnedMesh(MeshNode, OutSkinnedModel->GetSkeleton());
        u32 MaterialIndex = MeshNode->mMaterialIndex;
        GPUTexture ColorTex = matEmissiveTextures[MaterialIndex];
        OutSkinnedModel->Meshes[MeshIndex] = SkinnedMesh;
        OutSkinnedModel->Textures[MeshIndex] = ColorTex;
    }

    matEmissiveTextures.free();

    return true;
}

void joint_pose_sampler_t::Create(const aiNodeAnim *InChannel)
{
    ASSERT(!Positions.data && !Rotations.data && !Scales.data);
    u32 NumPosKeys = InChannel->mNumPositionKeys;
    u32 NumRotKeys = InChannel->mNumRotationKeys;
    u32 NumScaKeys = InChannel->mNumScalingKeys;
    void *PosArrBlock = StaticGameMemory.Alloc(sizeof(keyframe_position_t)*NumPosKeys, 
        alignof(keyframe_position_t));
    void *RotArrBlock = StaticGameMemory.Alloc(sizeof(keyframe_rotation_t)*NumRotKeys,
        alignof(keyframe_rotation_t));
    void *ScaArrBlock = StaticGameMemory.Alloc(sizeof(keyframe_scale_t)*NumScaKeys,
        alignof(keyframe_scale_t));
    Positions = mem_indexer<keyframe_position_t>(PosArrBlock, NumPosKeys);
    Rotations = mem_indexer<keyframe_rotation_t>(RotArrBlock, NumRotKeys);
    Scales = mem_indexer<keyframe_scale_t>(ScaArrBlock, NumScaKeys);

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

void joint_pose_sampler_t::Delete()
{
    LogError("joint_pose_sampler_t::Delete NOT IMPLEMENTED");
}

mat4 joint_pose_sampler_t::SampleJointLocalPoseAt(float AnimationTime)
{
    mat4 Translation = InterpolatePosition(AnimationTime);
    mat4 Rotation = InterpolateRotation(AnimationTime);
    mat4 Scale = InterpolateScale(AnimationTime);
    mat4 CurrentLocalTransform = Translation * Rotation * Scale;
    return CurrentLocalTransform;
}

int joint_pose_sampler_t::GetPositionIndex(float AnimationTime)
{
    for (int i = 0; i < Positions.count - 1; ++i)
    {
        if (AnimationTime < Positions[i + 1].Timestamp)
            return i;
    }
    ASSERT(0);
    return 0;
}

int joint_pose_sampler_t::GetRotationIndex(float AnimationTime)
{
    for (int i = 0; i < Rotations.count - 1; ++i)
    {
        if (AnimationTime < Rotations[i + 1].Timestamp)
            return i;
    }
    ASSERT(0);
    return 0;
}

int joint_pose_sampler_t::GetScaleIndex(float AnimationTime)
{
    for (int i = 0; i < Scales.count - 1; ++i)
    {
        if (AnimationTime < Scales[i + 1].Timestamp)
            return i;
    }
    ASSERT(0);
    return 0;
}

float joint_pose_sampler_t::GetScaleFactor(float LastTimestamp, float NextTimestamp, float AnimationTime)
{
    float CurrentOffset = AnimationTime - LastTimestamp;
    float FramesDiff = NextTimestamp - LastTimestamp;
    return CurrentOffset / FramesDiff;
}

mat4 joint_pose_sampler_t::InterpolatePosition(float AnimationTime)
{
    if (Positions.count == 1)
        return TranslationMatrix(Positions[0].Position);

    int P0Index = GetPositionIndex(AnimationTime);
    int P1Index = P0Index + 1;
    float ScaleFactor = GetScaleFactor(Positions[P0Index].Timestamp,
        Positions[P1Index].Timestamp, AnimationTime);
    
    vec3 FinalPosition = Lerp(Positions[P0Index].Position, Positions[P1Index].Position, ScaleFactor);
    return TranslationMatrix(FinalPosition);
}

mat4 joint_pose_sampler_t::InterpolateRotation(float AnimationTime)
{
    if (Rotations.count == 1)
        return mat4(Normalize(Rotations[0].Orientation));

    int P0Index = GetRotationIndex(AnimationTime);
    int P1Index = P0Index + 1;
    float ScaleFactor = GetScaleFactor(Rotations[P0Index].Timestamp,
        Rotations[P1Index].Timestamp, AnimationTime);

    quat FinalRotation = Slerp(Rotations[P0Index].Orientation,
        Rotations[P1Index].Orientation, ScaleFactor);
    return mat4(Normalize(FinalRotation));
}

mat4 joint_pose_sampler_t::InterpolateScale(float AnimationTime)
{
    return mat4();

    if (Scales.count == 1)
        return ScaleMatrix(Scales[0].Scale);

    int P0Index = GetScaleIndex(AnimationTime);
    int P1Index = P0Index + 1;
    float ScaleFactor = GetScaleFactor(Scales[P0Index].Timestamp,
        Scales[P1Index].Timestamp, AnimationTime);
    
    vec3 FinalScale = Lerp(Scales[P0Index].Scale, Scales[P1Index].Scale, ScaleFactor);
    return TranslationMatrix(FinalScale);
}



void FreeModelGLTF(ModelGLTF model)
{
    ASSERT(0);

    ASSERT(model.meshes.count == model.color.count);

    for (int i = 0; i < model.meshes.count; ++i)
    {
        DeleteGPUMeshIndexed(&model.meshes[i]);
        DeleteGPUTexture(&model.color[i]);
    }
}

void RenderModelGLTF(ModelGLTF model)
{
    for (int i = 0; i < model.meshes.count; ++i)
    {
        GPUMeshIndexed m = model.meshes[i];
        GPUTexture t = model.color[i];

        glActiveTexture(GL_TEXTURE0);
        if (t.id)
            glBindTexture(GL_TEXTURE_2D, t.id);
        else
            glBindTexture(GL_TEXTURE_2D, Assets.DefaultMissingTexture.id);

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

    ASSERT(model->meshes.data == NULL);
    ASSERT(model->color.data == NULL);
    void *Meshes = StaticGameMemory.Alloc(scene->mNumMeshes*sizeof(GPUMeshIndexed), alignof(GPUMeshIndexed));
    void *Textures = StaticGameMemory.Alloc(scene->mNumMeshes*sizeof(GPUTexture), alignof(GPUTexture));
    model->meshes = mem_indexer<GPUMeshIndexed>(Meshes, scene->mNumMeshes);
    model->color = mem_indexer<GPUTexture>(Textures, scene->mNumMeshes);

    for (u32 meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
    {
        aiMesh* meshNode = scene->mMeshes[meshIndex];
        GPUMeshIndexed gpumesh = ASSIMPMeshToGPUMeshIndexed(meshNode);
        u32 matIndex = meshNode->mMaterialIndex;
        GPUTexture colorTex = matEmissiveTextures[matIndex];

        model->meshes[meshIndex] = gpumesh;
        model->color[meshIndex] = colorTex;
    }

    arrfree(matEmissiveTextures);

    return true;
}
