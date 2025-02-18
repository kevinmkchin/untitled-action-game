#pragma once


struct bone_info_t
{
    int Id = -1; // index in FinalBonesMatrices
    mat4 InverseBindPoseTransform; // transforms vertex from model space to bone/joint space
};

struct anim_vertex_t
{
    vec3 Pos;
    vec2 Tex;
    vec3 Norm;

#define MAX_BONE_INFLUENCE 4
    //bone indexes which will influence this vertex
    int BoneIDs[MAX_BONE_INFLUENCE];
    //weights from each bone
    float BoneWeights[MAX_BONE_INFLUENCE];
};

struct skeletal_mesh_t
{
    u32 VAO = 0;
    u32 VBO = 0;
    u32 IBO = 0;
    u32 IndicesCount = 0;
};

void CreateSkeletalMesh(skeletal_mesh_t *mesh, 
    dynamic_array<anim_vertex_t> Vertices, 
    dynamic_array<u32> Indices);

struct anim_model_t
{
    std::map<std::string, bone_info_t> BoneInfoMap;
    int BoneCounter = 0;

    // meshes
    // textures?
    struct skeletal_mesh_t *meshes   = NULL;
    struct GPUTexture      *textures = NULL;

    dynamic_array<struct animation_t*> Animations;

public:
    skeletal_mesh_t ProcessMesh(aiMesh* Mesh);
private:
    void SetVertexBoneDataToDefault(anim_vertex_t *Vertex);
    void SetVertexBoneData(anim_vertex_t *Vertex, int BoneID, float Weight);
    void ExtractBoneWeightForVertices(dynamic_array<anim_vertex_t> Vertices, aiMesh *Mesh);

};

bool LoadAnimatedModel_GLTF2Bin(anim_model_t *Model, const char *FilePath);

/*
For each frame we want to interpolate all bones in the hierarchy and get
their final transformations matrices which will be supplied to shader 
uniform FinalBonesMatrices.
*/

struct keyframe_position_t
{
    vec3 Position;
    float Timestamp;
};

struct keyframe_rotation_t
{
    quat Orientation;
    float Timestamp;
};

struct keyframe_scale_t
{
    // NOTE(Kevin): I should disallow non-uniform scaling and have a single float here
    vec3 Scale;
    float Timestamp;
};

struct anim_bone_t
{
    // Extract bone keyframes from aiNodeAnim
    void Create(const aiNodeAnim *InChannel);
    void Delete();

    // Interpolates b/w positions, rotations, scaling keys based on the current time of 
    // the animation and prepares the local transformation matrix by combining all keys 
    // tranformations */
    mat4 Update(float AnimationTime);
    
    // Gets the index of key Positions to interpolate from based on 
    // the current animation time
    int GetPositionIndex(float AnimationTime);
    int GetRotationIndex(float AnimationTime);
    int GetScaleIndex(float AnimationTime);

    // Instead of storing the local transform, each bone can index into its parent,
    // calculate the current pose matrix, and store that
    // Actually, this should be in the animator. The animation clip and anim_bone data
    // should be read-only
    mat4 CurrentPoseTransform; // takes joint space into current pose model space

    mat4 InverseBindPose;
    // std::string Name;
    int PaletteIndex;
    u8 ParentId; // array index not palette index
    u8 SelfId;

private:
    // Get normalized [0, 1] value for Lerp & Slerp
    float GetScaleFactor(float LastTimestamp, float NextTimestamp, float AnimationTime);

    // Figures out which position keys to interpolate b/w and performs
    // the interpolation and returns the translation matrix
    mat4 InterpolatePosition(float AnimationTime);
    mat4 InterpolateRotation(float AnimationTime);
    mat4 InterpolateScale(float AnimationTime);

    dynamic_array<keyframe_position_t> Positions;
    dynamic_array<keyframe_rotation_t> Rotations;
    dynamic_array<keyframe_scale_t>    Scales;
};

// struct assimp_node_data_t
// {
//     mat4 Transformation;
//     std::string Name;
//     int ChildrenCount;
//     //dynamic_array<assimp_node_data_t> Children;
//     std::vector<assimp_node_data_t> Children;
// };

// Holds a hierarchical record of anim bones read from aiAnimation
struct animation_t
{
    float DurationInTicks;
    float TicksPerSecond;
    std::vector<anim_bone_t> Bones;
    mat4 RootTransform;
    // assimp_node_data_t RootNode;
    // std::map<std::string, bone_info_t> m_BoneInfoMap;

    anim_bone_t *FindBone(const int PaletteIndex)
    {
        auto iter = std::find_if(Bones.begin(), Bones.end(),
            [&](const anim_bone_t& Bone)
            {
                return Bone.PaletteIndex == PaletteIndex;
            }
        );
        if (iter == Bones.end()) return nullptr;
        else return &(*iter);
    }

    void Destroy();

    void ReadHierarchyData(const aiNode *Src, anim_model_t *Model);
    void ReadBones(const aiAnimation* AssimpAnim, anim_model_t *Model);
};

struct skeletal_animator_t
{
    // Skinning matrix palette
    // This array is known as a matrix palette. The matrix palette is passed to the
    // rendering engine when rendering a skinned mesh. For each vertex, the
    // renderer looks up the appropriate joint’s skinning matrix in the palette
    // and uses it to transform the vertex from bind pose into current pose.
    dynamic_array<mat4> FinalBonesMatrices;

    animation_t* m_CurrentAnimation;
    float m_CurrentTime;
    float m_DeltaTime;

    void PlayAnimation(animation_t *pAnimation)
    {
        if (FinalBonesMatrices.data == nullptr)
        {
            FinalBonesMatrices.setlen(108);
            for (int i = 0; i < 108; i++)
                FinalBonesMatrices[i] = mat4();
        }

        m_CurrentAnimation = pAnimation;
        m_CurrentTime = 0.0f;
    }

    void UpdateAnimation(float dt)
    {
        m_DeltaTime = dt;
        if (m_CurrentAnimation)
        {
            m_CurrentTime += m_CurrentAnimation->TicksPerSecond * dt;
            m_CurrentTime = fmod(m_CurrentTime, m_CurrentAnimation->DurationInTicks);

            auto& Bones = m_CurrentAnimation->Bones;
            for (size_t i = 0; i < Bones.size(); ++i)
            {
                anim_bone_t& AnimBone = Bones[i];
                mat4 LocalTransformCurrentPose = AnimBone.Update(m_CurrentTime);
                mat4 ParentCurrentPoseTransform = mat4();
                if (AnimBone.ParentId < 0xFF)
                {
                    ASSERT(AnimBone.ParentId < i);
                    ParentCurrentPoseTransform = Bones[AnimBone.ParentId].CurrentPoseTransform;
                }
                else
                {
                    ParentCurrentPoseTransform = m_CurrentAnimation->RootTransform;
                }

                AnimBone.CurrentPoseTransform = ParentCurrentPoseTransform * LocalTransformCurrentPose;
                
                FinalBonesMatrices[AnimBone.PaletteIndex] = AnimBone.CurrentPoseTransform * AnimBone.InverseBindPose;
            }

            // CalculateBoneTransform(&m_CurrentAnimation->RootNode, mat4());
        }
    }
    
    // void CalculateBoneTransform(const assimp_node_data_t* node, mat4 ModelSpaceFromParentSpace)
    // {
    //     std::string nodeName = node->Name;
    //     mat4 ParentJointSpaceFromLocalJointSpace = node->Transformation; // bind pose
    
    //     anim_bone_t* Bone = m_CurrentAnimation->FindBone(nodeName);
    
    //     if (Bone)
    //     {
    //         Bone->Update(m_CurrentTime);
    //         // The Current Pose Transform
    //         ParentJointSpaceFromLocalJointSpace = Bone->LocalTransform; // current pose
    //     }
    
    //     // The complete Skinning Matrix
    //     mat4 GlobalModelFromJoint = ModelSpaceFromParentSpace * ParentJointSpaceFromLocalJointSpace;
    
    //     auto boneInfoMap = m_CurrentAnimation->m_BoneInfoMap;
    //     if (boneInfoMap.find(nodeName) != boneInfoMap.end())
    //     {
    //         int index = boneInfoMap[nodeName].Id;
    //         mat4 JointFromModel = boneInfoMap[nodeName].InverseBindPoseTransform;
    //         FinalBonesMatrices[index] = GlobalModelFromJoint * JointFromModel;
    //     }
    
    //     for (int i = 0; i < node->ChildrenCount; ++i)
    //         CalculateBoneTransform(&node->Children[i], GlobalModelFromJoint);
    // }

};


struct ModelGLTF
{
    struct GPUMeshIndexed *meshes   = NULL;
    struct GPUTexture     *color    = NULL;
};

void FreeModelGLTF(ModelGLTF model);
void RenderModelGLTF(ModelGLTF model);
bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath);
