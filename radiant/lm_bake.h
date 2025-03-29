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

struct bake_lm_params_t
{
    float *OutputLightmap;

    int DoDirectionalLight;
    float3 DirectionToSun;

    float3 SkyboxColor;
    float SkyboxBrightness;

    int CountOfPointLights;
    cu_pointlight_t *PointLights;

    float3 *TexelWorldPositions;
    float3 *TexelWorldNormals;

    // Traversable handle to the geometry acceleration structure
    OptixTraversableHandle GASHandle;

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
