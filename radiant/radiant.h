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
    // Procedure 1: Lightmap baking
    float *OutputLightmap; // The array of light values at each texel. This gets set when baking.
    size_t OutputLightmapSize;
    radiant_vec3_t *LightMapTexelPositions; // this must be OutputLightmapSize long
    radiant_vec3_t *LightMapTexelNormals;   // this must be OutputLightmapSize long

    // Procedure 2: Cache light visibility information
    bool CacheDirectLightIndices = false;
    short *OutputDirectLightIndices;
    size_t OutputDirectLightIndicesSize;
    size_t OutputDirectLightIndicesPerSample;
    radiant_vec3_t *DirectLightCachePositions;

    // World and lighting description
    radiant_vec3_t *WorldGeometryVertices;
    size_t WorldGeometryVerticesCount;
    radiant_pointlight_t *PointLights;
    size_t PointLightsCount;
    radiant_vec3_t DirectionToSun; // if set to { 0.f, 0.f, 0.f } then no sun
    radiant_vec3_t SkyboxColor = { 0.53f, 0.81f, 0.92f };
    float SkyboxBrightness = 0.4f;

    // Lightmap bake parameters
    int NumberOfSampleRaysPerTexel;
    int NumberOfLightBounces;
    bool BakeDirectLighting = false;
};

#ifdef RADIANT_EXPORTS
    #define RADIANT_API __declspec(dllexport)
#else
    #define RADIANT_API __declspec(dllimport)
#endif

extern "C" RADIANT_API void __cdecl RadiantBake(radiant_bake_info_t BakeInfo);
