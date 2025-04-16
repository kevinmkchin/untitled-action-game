#pragma once

#include "common.h"
#include "winged.h"
#include "mem.h"
#include <stb_rect_pack.h>
#include "facebatch.h"
#include "utility.h"

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

struct static_point_light_t
{
    vec3 Pos;
    float AttenuationLinear = 0.02f;
    float AttenuationQuadratic = 0.00019f;
};

struct game_map_build_data_t
{
    ByteBuffer Output;

    int TotalFaceCount = 0;
    std::unordered_map<u32, std::vector<float>> VertexBuffers;
    std::vector<vec3> ColliderWorldPoints;
    std::vector<u32> ColliderSpans;

    vec3 PlayerStartPosition = vec3();
    vec3 PlayerStartRotation = vec3();

    vec3 DirectionToSun = vec3();
    dynamic_array<static_point_light_t> PointLights;
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
    void GetLightmap(float **PtrToLightMapAtlas, i32 *AtlasWidth, i32 *AtlasHeight);
    void FreeLightmap();

private:
    void PrepareFaceLightmapsAndTexelStorage();
    void PackLightmapsAndMapLocalUVToGlobalUV();
    void GenerateLevelVertices();
    void GenerateLightmapOcclusionTestTree();
    void ThreadSafe_DoDirectLightingIntoLightMap(u32 patchIndexStart, u32 patchIndexEnd);
    void CreateMultiplierMap();
    // void CalcBounceLightForTexel(const lm_face_t& FaceLightmap, 
    //     u32 TexelOffset, const GLsizeiptr NumFloatsPerFace);

private:
    i32 lightMapAtlasW = 1024;//4096;
    i32 lightMapAtlasH = 1024;//4096;

private:
    game_map_build_data_t *BuildDataShared = nullptr;

    // output
    float *LIGHTMAPATLAS = NULL;

    // intermediary data
    dynamic_array<lm_face_t> FaceLightmaps;
    dynamic_array<stbrp_rect> PackedLMRects;
    vec3 *all_lm_pos = NULL;
    vec3 *all_lm_norm = NULL;
    vec3 *all_lm_tangent = NULL;
    float *all_light_global = NULL;
    float *all_light_direct = NULL;
    float *all_light_indirect = NULL;
    int UsedLightmapTexelCount = 0;

    face_batch_t SceneLightingModel;
    float MultiplierMapTop[HemicubeFaceArea];
    float MultiplierMapSide[HemicubeFaceAreaHalf];
};

struct lc_ambient_t
{
    float PosX = 0.f;
    float NegX = 0.f;
    float PosY = 0.f;
    float NegY = 0.f;
    float PosZ = 0.f;
    float NegZ = 0.f;
};

struct lc_light_indices_t
{
    // short Indices[4] = { -1, -1, -1, -1 };
    short Index0 = -1;
    short Index1 = -1;
    short Index2 = -1;
    short Index3 = -1;
};

constexpr short SUNLIGHTINDEX = SHRT_MAX;

struct lc_volume_t
{
    // Return the index of the nearest cube to position
    size_t IndexByPosition(vec3 WorldPosition);

    fixed_array<vec3> CubePositions;
    fixed_array<lc_ambient_t> AmbientCubes;
    fixed_array<lc_light_indices_t> SignificantLightIndices;

    static constexpr u64 lc_volume_t_serialize_start_marker = 0x6C63766F6C736572;
    void Serialize(ByteBuffer *Buf);
    void Deserialize(ByteBuffer *Buf, MemoryType VolumeStorageType = MemoryType::Level);


    vec3 Start;
    vec3 End;
    int CountX = -1; // length
    int CountY = -1; // height
    int CountZ = -1; // width
    ivec3 LightCubePlacementInterval = ivec3(48,64,48);
};

struct lc_volume_baker_t
{
    void BakeLightCubes(game_map_build_data_t& BuildData);

    lc_volume_t LightCubeVolume;

    // Since level geomtry will often by aligned to grid, let's offset the positions
    // of cubes by a certain amount so they aren't clipping geometry as often.
    vec3 CubePlacementOffsetToAvoidClipping = vec3(4.5f, 16.5f, 4.5f);

private:
    void PlaceLightCubes();

private:
    game_map_build_data_t *BuildDataShared = nullptr;

};
