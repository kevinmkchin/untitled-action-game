#pragma once


struct bone_info_t
{
    int Id = -1; // index in FinalBonesMatrices
    mat4 Offset; // transforms vertex from model space to bone space
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
    vec3 Scale;
    float Timestamp;
};

struct anim_bone_t
{
    // Extract bone keyframes from aiNodeAnim
    void Create(const std::string& InName, int InId, const aiNodeAnim *InChannel);
    void Delete();

    // Interpolates b/w positions, rotations, scaling keys based on the current time of 
    // the animation and prepares the local transformation matrix by combining all keys 
    // tranformations */
    void Update(float AnimationTime);
    
    // Gets the index of key Positions to interpolate from based on 
    // the current animation time
    int GetPositionIndex(float AnimationTime);
    int GetRotationIndex(float AnimationTime);
    int GetScaleIndex(float AnimationTime);

    mat4 LocalTransform;
    std::string Name;
    int Id;

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

struct assimp_node_data_t
{
    mat4 Transformation;
    std::string Name;
    int ChildrenCount;
    //dynamic_array<assimp_node_data_t> Children;
    std::vector<assimp_node_data_t> Children;
};

// Holds a hierarchical record of anim bones read from aiAnimation
struct animation_t
{
    float DurationInTicks;
    float TicksPerSecond;
    std::vector<anim_bone_t> Bones;
    assimp_node_data_t RootNode;
    std::map<std::string, bone_info_t> m_BoneInfoMap;

    anim_bone_t* FindBone(const std::string& name)
    {
        auto iter = std::find_if(Bones.begin(), Bones.end(),
            [&](const anim_bone_t& Bone)
            {
                return Bone.Name == name;
            }
        );
        if (iter == Bones.end()) return nullptr;
        else return &(*iter);
    }

    void Destroy();

    void ReadHierarchyData(const aiNode *InRoot) { ReadHierarchyData(RootNode, InRoot); }
    void ReadBones(const aiAnimation* AssimpAnim, anim_model_t *Model);
private:
    void ReadHierarchyData(assimp_node_data_t& Dest, const aiNode* Src);

};

struct skeletal_animator_t
{
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
            CalculateBoneTransform(&m_CurrentAnimation->RootNode, mat4());
        }
    }
    
    void CalculateBoneTransform(const assimp_node_data_t* node, mat4 parentTransform)
    {
        std::string nodeName = node->Name;
        mat4 nodeTransform = node->Transformation;
    
        anim_bone_t* Bone = m_CurrentAnimation->FindBone(nodeName);
    
        if (Bone)
        {
            Bone->Update(m_CurrentTime);
            nodeTransform = Bone->LocalTransform;
        }
    
        mat4 globalTransformation = parentTransform * nodeTransform;
    
        auto boneInfoMap = m_CurrentAnimation->m_BoneInfoMap;
        if (boneInfoMap.find(nodeName) != boneInfoMap.end())
        {
            int index = boneInfoMap[nodeName].Id;
            mat4 offset = boneInfoMap[nodeName].Offset;
            FinalBonesMatrices[index] = globalTransformation * offset;
        }
    
        for (int i = 0; i < node->ChildrenCount; ++i)
            CalculateBoneTransform(&node->Children[i], globalTransformation);
    }

};


struct ModelGLTF
{
    struct GPUMeshIndexed *meshes   = NULL;
    struct GPUTexture     *color    = NULL;
};

void FreeModelGLTF(ModelGLTF model);
void RenderModelGLTF(ModelGLTF model);
bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath);
