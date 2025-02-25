#pragma once

enum RayType
{
    RAY_TYPE_DIRECTIONAL_LIGHT = 0,
    RAY_TYPE_COUNT
};

struct Params
{
    float *OutputLightmap;

    int DoDirectionalLight;
    float3 DirectionToSun;

    float3 *TexelWorldPositions;
    float3 *TexelWorldNormals;

    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};
