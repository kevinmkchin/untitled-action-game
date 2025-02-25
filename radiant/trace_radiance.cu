
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


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin = params.TexelWorldPositions[idx.x];
    float3 ray_direction = params.DirectionToSun;

    float LightValue = 0.f;

    if (params.DoDirectionalLight)
    {
        float3 TexelNormal = params.TexelWorldNormals[idx.x];
        float CosTheta = dot(TexelNormal, params.DirectionToSun);
        if (CosTheta > 0.f)
        {
            unsigned int p0;
            optixTrace( // Trace the ray against our scene hierarchy
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,                // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask( 255 ), // Specify always visible
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_DIRECTIONAL_LIGHT, // SBT offset   -- See SBT discussion
                    RAY_TYPE_COUNT,             // SBT stride   -- See SBT discussion
                    RAY_TYPE_DIRECTIONAL_LIGHT, // missSBTIndex -- See SBT discussion
                    p0);
            float result = __uint_as_float( p0 );
            LightValue += result;
        }
    }
    
    params.OutputLightmap[idx.x] = LightValue;
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
    // setPayload( make_float3(1.f, 1.f, 1.f) );
}
