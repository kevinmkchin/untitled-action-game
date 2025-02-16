#pragma once


struct bone_info_t
{
    int Id; // index in FinalBonesMatrices
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
    u32 VAO;
    u32 VBO;
    u32 IBO;
    u32 IndicesCount = 0;
};

void CreateSkeletalMesh(skeletal_mesh_t *mesh, 
    dynamic_array<anim_vertex_t> Vertices, 
    dynamic_array<u32> Indices);

struct anim_model_t
{
    std::map<std::string, bone_info_t> BoneInfoMap;
    int BoneCounter = 0;

private:
    skeletal_mesh_t ProcessMesh(aiMesh* Mesh);
    void SetVertexBoneDataToDefault(anim_vertex_t *Vertex);
    void SetVertexBoneData(anim_vertex_t *Vertex, int BoneID, float Weight);
    void ExtractBoneWeightForVertices(dynamic_array<anim_vertex_t> Vertices, aiMesh *Mesh);

};

struct ModelGLTF
{
    struct GPUMeshIndexed *meshes   = NULL;
    struct GPUTexture     *color    = NULL;
    // animations and bones and shit
};

void FreeModelGLTF(ModelGLTF model);
void RenderModelGLTF(ModelGLTF model);
bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath);
