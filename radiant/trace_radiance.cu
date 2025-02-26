
#include <optix.h>

#include "trace_radiance.h"

#include "cuda_helpers.h"
#include "vec_math.h"

extern "C" {
__constant__ Params params;
}



// Orthonormal basis helper
struct Onb
{
    __forceinline__ __device__ Onb(const float3 &normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3 &p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

// A simple hash function
static __inline__ __device__ unsigned int hash(unsigned int a) 
{
    a = (a ^ 61u) ^ (a >> 16u);
    a *= 9u;
    a = a ^ (a >> 4u);
    a *= 0x27d4eb2du;
    a = a ^ (a >> 15u);
    return a;
}

// A simple linear congruential generator for pseudo-random numbers.
static __forceinline__ __device__ float rnd(unsigned int &seed) 
{
    seed = 1664525u * seed + 1013904223u;
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

// Cosine-weighted hemisphere sampling in local space.
static __forceinline__ __device__ float3 cosine_sample_hemisphere(unsigned int &seed) 
{
    float u1 = rnd(seed);
    float u2 = rnd(seed);
    float r = sqrtf(u1);
    float theta = 2.0f * M_PIf * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);
    return make_float3(x, y, z);
}

static __device__ float CalculateDirectionalLight(float3 Position, float3 SurfaceNormal)
{
    float LightValue = 0.f;

    if (params.DoDirectionalLight)
    {
        float CosTheta = dot(SurfaceNormal, params.DirectionToSun);
        if (CosTheta > 0.f)
        {
            // Send
            // p0: SurfaceNormal.x
            // p1: SurfaceNormal.y
            // p2: SurfaceNormal.z
            // Receive
            // p0: float light contribution
            unsigned int p0, p1, p2;
            p0 = __float_as_uint(SurfaceNormal.x);
            p1 = __float_as_uint(SurfaceNormal.y);
            p2 = __float_as_uint(SurfaceNormal.z);
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
                p0, p1, p2);
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
            // p1: SurfaceNormal.x
            // p2: SurfaceNormal.y
            // p3: SurfaceNormal.z
            // Receive
            // p0: float light contribution
            unsigned int p0, p1, p2, p3;
            p0 = i;
            p1 = __float_as_uint(SurfaceNormal.x);
            p2 = __float_as_uint(SurfaceNormal.y);
            p3 = __float_as_uint(SurfaceNormal.z);
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
                p0, p1, p2, p3);
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

    // Do Monte Carlo sampling from this texel to gather bounce lighting
    // Hmmm, instead of recursively casting rays, maybe it would be faster if
    // all the texels did monte carlo sampling once, update the light buffer
    // then do the subsequent bounces? by doing monte carlo sampling again.
    // we would need a way to look up or sample the lightmap upon a hit.
    // well, the lightmap is a flat array, so I can find the texel index
    // from the UV by doing V * ActualHeight * ActualWidth + U * ActualWidth
    // or maybe, 

    // Indirect light values
    unsigned int seed = hash(idx.x); // Each lightmap texel needs a unique seed
    float AccumulatedRadiance = 0.f;
    int NumSamples = 4096;
    // in scenes with tiny slivers of surfaces with direct lighting, doing only one bounce
    // is very high variance and introduces lots of noise. On the other hand, my indoor point
    // light test scene looks good even with low num samples and single bounce.
    int i = NumSamples;
    do
    {
        float3 N = TexelNormal;
        float3 w_in = cosine_sample_hemisphere(seed);
        Onb onb(N);
        onb.inverse_transform(w_in);
        // at this point, w_in is in world space
        float3 ray_direction = w_in;
        float3 ray_origin = TexelPosition;

        // Send
        // p0: nothing
        // Receive
        // p0: irradiance at that point
        unsigned int p0;
        optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,
            100000.f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_HEMISPHERE_SAMPLE,
            RAY_TYPE_COUNT,
            RAY_TYPE_HEMISPHERE_SAMPLE,
            p0);
        // miss returns 0
        // hit returns the irradiance at that point (direct + indirect)
        float IrradianceAtTheHitPoint = __uint_as_float(p0);
        // the Optix path tracer sample uses weighted luminance which is specifically 30% red, 59% green, 11% blue
        float3 Albedo = make_float3(0.33f);
        float AttenuatedRadianceFromThatDirection = IrradianceAtTheHitPoint * Albedo.x; // just one component for now
        AccumulatedRadiance += AttenuatedRadianceFromThatDirection;

    } while (--i);
    float IrradianceAtThisPoint = AccumulatedRadiance / float(NumSamples);

    params.OutputLightmap[idx.x] = DirectLightValue + IrradianceAtThisPoint;
    // params.OutputLightmap[idx.x] = IrradianceAtThisPoint;
}

extern "C" __global__ void __miss__HemisphereSample()
{
    optixSetPayload_0(__float_as_uint(0.f));
}

extern "C" __global__ void __closesthit__HemisphereSample()
{
    // if the hemisphere sampling ray hits a backface, then this point is inside a wall...

    float3 RayOrigin = optixGetWorldRayOrigin();
    float3 RayDirection = optixGetWorldRayDirection();
    float HitT = optixGetRayTmax();
    float3 HitPointPosition = RayOrigin + HitT * RayDirection; // Compute the world-space hit position

    float3 Vertices[3] = {}; // vertices of the hit triangle
    optixGetTriangleVertexData(params.handle, optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0.f, Vertices);
    float3 HitPointNormal = normalize(cross(Vertices[1] - Vertices[0], Vertices[2] - Vertices[0]));

    if (dot(HitPointNormal, RayDirection) >= 0.f) // backface...
    {
        optixSetPayload_0(__float_as_uint(0.f));
    }
    else
    {
        float DirectLightValueAtHitPoint = CalculateDirectionalLight(HitPointPosition, HitPointNormal);
        // no indir lighting for now
        optixSetPayload_0(__float_as_uint(DirectLightValueAtHitPoint));
    }

}

extern "C" __global__ void __miss__PointLight()
{
    // If ray misses and goes off into space, then nothing in between the texel and the point light

    float3 SurfaceNormal;
    SurfaceNormal.x = __uint_as_float(optixGetPayload_1());
    SurfaceNormal.y = __uint_as_float(optixGetPayload_2());
    SurfaceNormal.z = __uint_as_float(optixGetPayload_3());

    unsigned int PointLightIdx = optixGetPayload_0();
    cu_pointlight_t PointLight = params.PointLights[PointLightIdx];
    float3 ToLight = PointLight.Position - optixGetWorldRayOrigin();
    float DistToLight = length(ToLight);

    float CosTheta = dot(SurfaceNormal, normalize(ToLight));
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

    float3 SurfaceNormal;
    SurfaceNormal.x = __uint_as_float(optixGetPayload_1());
    SurfaceNormal.y = __uint_as_float(optixGetPayload_2());
    SurfaceNormal.z = __uint_as_float(optixGetPayload_3());

    unsigned int PointLightIdx = optixGetPayload_0();
    cu_pointlight_t PointLight = params.PointLights[PointLightIdx];
    float3 PointLightWorldPos = PointLight.Position;

    float3 RayOrigin = optixGetWorldRayOrigin();
    float3 RayDirection = optixGetWorldRayDirection();
    float HitT = optixGetRayTmax();
    float3 HitPosition = RayOrigin + HitT * RayDirection; // Compute the world-space hit position

    float3 ToLight = PointLightWorldPos - RayOrigin;
    float DistToLight = length(ToLight);
    float DistToHit = length(HitPosition - RayOrigin);

    if (DistToLight < DistToHit)
    {
        float CosTheta = dot(SurfaceNormal, normalize(ToLight));
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
    float3 SurfaceNormal;
    SurfaceNormal.x = __uint_as_float(optixGetPayload_0());
    SurfaceNormal.y = __uint_as_float(optixGetPayload_1());
    SurfaceNormal.z = __uint_as_float(optixGetPayload_2());

    float DirectIntensity = dot(SurfaceNormal, params.DirectionToSun);
    optixSetPayload_0(__float_as_uint(DirectIntensity));
}

extern "C" __global__ void __closesthit__DirectionalLight()
{
    optixSetPayload_0(__float_as_uint(0.f));
}
