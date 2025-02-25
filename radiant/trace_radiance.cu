
#include <optix.h>

#include "trace_radiance.h"

#include "cuda_helpers.h"
#include "vec_math.h"

extern "C" {
__constant__ Params params;
}


//static __forceinline__ __device__ void setPayload( float3 p )
//{
//    optixSetPayload_0( __float_as_uint( p.x ) );
//    optixSetPayload_1( __float_as_uint( p.y ) );
//    optixSetPayload_2( __float_as_uint( p.z ) );
//}

// Calculating surface normal of hit point:
// https://forums.developer.nvidia.com/t/built-in-triangles-normal-with-optix-7-2/215360


static __forceinline__ __device__ float CalculateDirectionalLight(float3 Position, float3 SurfaceNormal)
{
    float LightValue = 0.f;

    if (params.DoDirectionalLight)
    {
        float CosTheta = dot(SurfaceNormal, params.DirectionToSun);
        if (CosTheta > 0.f)
        {
            // Send
            // p0: nothing
            // Receive
            // p0: float light contribution
            unsigned int p0;
            optixTrace( // Trace the ray against our scene hierarchy
                params.handle,
                Position,
                params.DirectionToSun,
                0.01f,    // Min intersection distance
                100000.f, // Max intersection distance
                0.0f,     // rayTime -- used for motion blur
                OptixVisibilityMask(255), // Specify always visible
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_DIRECTIONAL_LIGHT, // SBT offset   -- See SBT discussion
                RAY_TYPE_COUNT,             // SBT stride   -- See SBT discussion
                RAY_TYPE_DIRECTIONAL_LIGHT, // missSBTIndex -- See SBT discussion
                p0);
            float result = __uint_as_float(p0);
            LightValue += result;
        }
    }

    int CountOfPointLights = params.CountOfPointLights;
    for (unsigned int i = 0; i < CountOfPointLights; ++i)
    {
        float3 PointLightWorldPos = params.PointLights[i].Position;
        float3 DirectionToPointLight = normalize(PointLightWorldPos - Position);
        float CosTheta = dot(SurfaceNormal, DirectionToPointLight);
        if (CosTheta > 0.f)
        {
            // Send
            // p0: unsigned int point light index
            // Receive
            // p0: float light contribution
            unsigned int p0 = i;
            optixTrace(
                params.handle,
                Position,
                DirectionToPointLight,
                0.01f,    // Min intersection distance
                100000.f, // Max intersection distance
                0.0f,     // rayTime -- used for motion blur
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_POINT_LIGHT,
                RAY_TYPE_COUNT,
                RAY_TYPE_POINT_LIGHT,
                p0);
            float LightContribution = __uint_as_float(p0);
            LightValue += LightContribution;
        }
    }

    return LightValue;
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 TexelPosition = params.TexelWorldPositions[idx.x];
    float3 TexelNormal = params.TexelWorldNormals[idx.x];

    float DirectLightValue = CalculateDirectionalLight(TexelPosition, TexelNormal);

    // TODO Indirect light values
    // Do Monte Carlo sampling from this texel to gather bounce lighting
    // Hmmm, instead of recursively casting rays, maybe it would be faster if
    // all the texels did monte carlo sampling once, update the light buffer
    // then do the subsequent bounces? by doing monte carlo sampling again.
    // we would need a way to look up or sample the lightmap upon a hit.
    // well, the lightmap is a flat array, so I can find the texel index
    // from the UV by doing V * ActualHeight * ActualWidth + U * ActualWidth
    // or maybe, 

    params.OutputLightmap[idx.x] = DirectLightValue;
}

extern "C" __global__ void __miss__PointLight()
{
    // If ray misses and goes off into space, then nothing in between the texel and the point light
    const uint3 idx = optixGetLaunchIndex();
    float3 TexelNormal = params.TexelWorldNormals[idx.x];
    unsigned int PointLightIdx = optixGetPayload_0();
    cu_pointlight_t PointLight = params.PointLights[PointLightIdx];
    float3 ToLight = PointLight.Position - optixGetWorldRayOrigin();
    float DistToLight = length(ToLight);

    float CosTheta = dot(TexelNormal, normalize(ToLight));
    float AttenLin = PointLight.AttenuationLinear;
    float AttenQuad = PointLight.AttenuationQuadratic;
    float Attenuation = 1.f / 
        (1.f + AttenLin * DistToLight + AttenQuad * DistToLight * DistToLight);
    float DirectIntensity = CosTheta * Attenuation;

    optixSetPayload_0(__float_as_uint(DirectIntensity));
}

extern "C" __global__ void __closesthit__PointLight()
{
    // If ray hits an object, then check if point light is before or after this object
    const uint3 idx = optixGetLaunchIndex();
    float3 TexelNormal = params.TexelWorldNormals[idx.x];
    unsigned int PointLightIdx = optixGetPayload_0();
    cu_pointlight_t PointLight = params.PointLights[PointLightIdx];
    float3 PointLightWorldPos = PointLight.Position;

    float3 RayOrigin = optixGetWorldRayOrigin();
    float3 RayDirection = optixGetWorldRayDirection();
    float HitT = optixGetRayTmax(); // Get the distance to the 
    float3 HitPosition = RayOrigin + HitT * RayDirection; // Compute the world-space hit position

    float3 ToLight = PointLightWorldPos - RayOrigin;
    float DistToLight = length(ToLight);
    float DistToHit = length(HitPosition - RayOrigin);

    if (DistToLight < DistToHit)
    {
        float CosTheta = dot(TexelNormal, normalize(ToLight));
        float AttenLin = PointLight.AttenuationLinear;
        float AttenQuad = PointLight.AttenuationQuadratic;
        float Attenuation = 1.f / 
            (1.f + AttenLin * DistToLight + AttenQuad * DistToLight * DistToLight);
        float DirectIntensity = CosTheta * Attenuation;

        optixSetPayload_0(__float_as_uint(DirectIntensity));
    }
    else
    {
        optixSetPayload_0(__float_as_uint(0.f));
    }
}

extern "C" __global__ void __miss__DirectionalLight()
{
    const uint3 idx = optixGetLaunchIndex();
    float3 TexelNormal = params.TexelWorldNormals[idx.x];
    float DirectIntensity = dot(TexelNormal, params.DirectionToSun);
    optixSetPayload_0(__float_as_uint(DirectIntensity));
}

extern "C" __global__ void __closesthit__DirectionalLight()
{
    optixSetPayload_0(__float_as_uint(0.f));
}
