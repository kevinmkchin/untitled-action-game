
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


// static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
// {
//     const float3 U = params.cam_u;
//     const float3 V = params.cam_v;
//     const float3 W = params.cam_w;
//     const float2 d = 2.0f * make_float2(
//             static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
//             static_cast<float>( idx.y ) / static_cast<float>( dim.y )
//             ) - 1.0f;

//     origin    = params.cam_eye;
//     direction = normalize( d.x * U + d.y * V + W );
// }


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin = params.TexelWorldPositions[idx.x];
    float3 ray_direction = params.DirectionToSun;

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    //float3 ray_origin, ray_direction;
    // computeRay( idx, dim, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    unsigned int p0;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            RAY_TYPE_COUNT,      // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0);

    float result = __uint_as_float( p0 );
    params.OutputLightmap[idx.x] = result;

    // Record results in our output raster
    // params.image[idx.y * params.image_width + idx.x] = make_color( result );
}


extern "C" __global__ void __miss__ms()
{
    // MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    // setPayload(  miss_data->bg_color );

    const uint3 idx = optixGetLaunchIndex();

    float3 TexelNormal = params.TexelWorldNormals[idx.x];
    float CosTheta = dot(TexelNormal, optixGetWorldRayDirection());
    if (CosTheta < 0.f)
    {
        optixSetPayload_0(__float_as_uint(0.f));
    }
    else
    {
        optixSetPayload_0(__float_as_uint(1.f));
    }

}


extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(__float_as_uint(0.f));
    // setPayload( make_float3(1.f, 1.f, 1.f) );
}
