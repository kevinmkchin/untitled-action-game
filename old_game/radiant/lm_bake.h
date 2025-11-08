#pragma once

enum RayType
{
    RAY_TYPE_DIRECTIONAL_LIGHT = 0,
    RAY_TYPE_POINT_LIGHT = 1,
    RAY_TYPE_HEMISPHERE_SAMPLE = 2,
    RAY_TYPE_COUNT
};

struct cu_pointlight_t
{
    float3 Position;
    float AttenuationLinear;
    float AttenuationQuadratic;
};

enum bake_lm_procedure
{
    BAKE_LIGHTMAP,
    BAKE_DIRECTLIGHTINFO
};

struct bake_lm_params_t
{
    bake_lm_procedure Procedure;

    // Procedure 1: Lightmap baking
    float *OutputLightmap;
    float3 *TexelWorldPositions;
    float3 *TexelWorldNormals;

    // Procedure 2: Cache light visibility information
    short *OutputDirectLightIndices;
    size_t OutputDirectLightIndicesPerSample;
    float *TempBufferForSignificanceComparisons;
    float3 *DirectLightCachePositions;

    // Light setup
    int DoSunLight;
    float3 DirectionToSun;
    float3 SkyboxColor;
    float SkyboxBrightness;
    unsigned int CountOfPointLights;
    cu_pointlight_t *PointLights;

    // Traversable handle to the geometry acceleration structure
    OptixTraversableHandle GASHandle;

    // Lightmap bake parameters
    int NumberOfSampleRaysPerTexel;
    int NumberOfBounces;
    int BakeDirectLighting;
};


struct RayGenData
{
    // No data needed
};

struct MissData
{
    // No data needed
};

struct HitGroupData
{
    // No data needed
};
