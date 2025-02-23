#pragma once


struct static_point_light_t
{
    vec3 Pos;
};

struct game_map_build_data_t
{
    ByteBuffer Output;

    int TotalFaceCount = 0;
    std::unordered_map<u32, std::vector<float>> VertexBuffers;
    std::vector<vec3> ColliderWorldPoints;
    std::vector<u32> ColliderSpans;

    dynamic_array<static_point_light_t> PointLights;

    vec3 PlayerStartPosition = vec3();
    vec3 PlayerStartRotation = vec3();

    vec3 DirectionToSun = vec3();
};

struct lm_face_t
{
    // texel data
    vec3  *pos = NULL;
    vec3  *norm = NULL;
    vec3  *tangent = NULL;
    float *light = NULL;
    float *light_direct = NULL;
    float *light_indirect = NULL;
    // dimensions
    i32 w = -1;
    i32 h = -1;
    MapEdit::Face *faceRef;
};

struct irradiance_record_placement_t
{
    vec3  Position;
    float Radius;
};

struct irradiance_record_data_t
{
    // vec3  Normal;
    float Irradiance;
    // vec3  Gradient;
};

constexpr float LightMapTexelSize = 4.f; // in world units
constexpr int MaxNumTexels = 1000000; // size to alloc per intermediate data array
constexpr int HemicubeFaceW = 100;
constexpr int HemicubeFaceH = HemicubeFaceW;
constexpr int HemicubeFaceWHalf = HemicubeFaceW/2;
constexpr int HemicubeFaceHHalf = HemicubeFaceH/2;
constexpr int HemicubeFaceArea = HemicubeFaceW*HemicubeFaceH;
constexpr int HemicubeFaceAreaHalf = HemicubeFaceW*HemicubeFaceHHalf;

struct lightmapper_t
{
    void BakeStaticLighting(game_map_build_data_t& BuildData);

private:
    void GenerateLightmapOcclusionTestTree();
    void PrepareFaceLightmapsAndTexelStorage();
    void PackLightmapsAndMapLocalUVToGlobalUV();
    void GenerateLevelVertices();
    void CreateMultiplierMap();
    bool CalcBounceLightForTexel(const lm_face_t& FaceLightmap, 
        u32 TexelOffset, const GLsizeiptr NumFloatsPerFace);

    void ThreadSafe_DoDirectLightingIntoLightMap(u32 patchIndexStart, u32 patchIndexEnd);

private:
    i32 lightMapAtlasW = 1024;//4096;
    i32 lightMapAtlasH = 1024;//4096;

private:
    game_map_build_data_t *BuildDataShared = nullptr;

    dynamic_array<lm_face_t> FaceLightmaps;
    dynamic_array<stbrp_rect> PackedLMRects;
    vec3 *all_lm_pos = NULL;
    vec3 *all_lm_norm = NULL;
    vec3 *all_lm_tangent = NULL;
    float *all_light_global = NULL;
    float *all_light_direct = NULL;
    float *all_light_indirect = NULL;

    face_batch_t SceneLightingModel;
    float MultiplierMapTop[HemicubeFaceArea];
    float MultiplierMapSide[HemicubeFaceAreaHalf];

    float DepthHarmonicMean;
};

extern lightmapper_t Lightmapper;

