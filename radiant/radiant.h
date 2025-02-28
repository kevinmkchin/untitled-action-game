// radiant.h

#pragma once

struct radiant_vec3_t
{
    float x, y, z;
};

struct radiant_pointlight_t
{
    radiant_vec3_t Position;
    float AttenuationLinear;
    float AttenuationQuadratic;
};

struct radiant_bake_info_t
{
    // The array of light values at each texel. This gets set when baking.
    float *OutputLightmap;
    size_t OutputLightmapSize;

    radiant_vec3_t *LightMapTexelPositions; // this must be OutputLightmapSize long
    radiant_vec3_t *LightMapTexelNormals;   // this must be OutputLightmapSize long

    radiant_vec3_t *WorldGeometryVertices;
    size_t WorldGeometryVerticesCount;

    radiant_pointlight_t *PointLights;
    size_t PointLightsCount;

    radiant_vec3_t DirectionToSun; // if set to { 0.f, 0.f, 0.f } then no sun

    int NumberOfSampleRaysPerTexel;
    int NumberOfLightBounces;
};


extern "C" __declspec(dllexport) void __cdecl RadiantBake(radiant_bake_info_t BakeInfo);
