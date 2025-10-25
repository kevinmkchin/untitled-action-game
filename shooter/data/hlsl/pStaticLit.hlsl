#pragma pack_matrix(column_major)

cbuffer ConstantBuffer0 : register(b0)
{
    float4x4 ProjFromWorld;
    float3 DirectionToSun;
};

Texture2D Tex0 : register(t8);
Texture2D Tex1 : register(t9);

SamplerState Sampler0 : register(s0);
SamplerState Sampler1 : register(s1);

struct model_instance_data
{
    float4x4 WorldFromModel;
    float4 PointLightsPos[4];
    float4 AmbientCube[6];
    int DoSunLight;
    int PointLightsCount;
    float PointLightsAttLin[4];
    float PointLightsAttQuad[4];
    float _padding_[6];
};

StructuredBuffer<model_instance_data> InstanceData : register(t0);

struct vs_out
{
    float4 P : SV_Position;
    float3 WorldP : POSITION;
    float3 WorldN : NORMAL;
    float2 UV0 : TEXCOORD0;
    float2 UV1 : TEXCOORD1;
    nointerpolation uint InstanceID : SV_InstanceID;
};

// world normal -> directional ambient contribution
float3 AmbientLight(float3 WorldNormal, uint InstanceIndex)
{
    /*

        Ambient Cube

    */

    float3 NSquared = WorldNormal * WorldNormal;
    int3 IsNegative = int3(WorldNormal.x < 0.0, WorldNormal.y < 0.0, WorldNormal.z < 0.0);
    float3 LinearColor = NSquared.x * InstanceData[InstanceIndex].AmbientCube[IsNegative.x].xyz +
                         NSquared.y * InstanceData[InstanceIndex].AmbientCube[IsNegative.y + 2].xyz +
                         NSquared.z * InstanceData[InstanceIndex].AmbientCube[IsNegative.z + 4].xyz;
    return LinearColor;
}

// world pos normal -> direct light contribution
float DiffuseLight(float3 WorldPos, float3 WorldNormal, uint InstanceIndex)
{
    /*

        Half-Lambertian Diffuse

        Softens diffuse contribution from light sources to prevent backsides of characters
        and props from looking too flat. Purely non-physical but pleasing to the eye.

    */

    float DirectLight = 0.f;

    if (InstanceData[InstanceIndex].DoSunLight != 0)
    {
        // TODO(Kevin): Intensity multiplier should be configured in map editor
        float SunIntensity = 1.f;
        float Lambertian = dot(DirectionToSun, WorldNormal);
        float ModifiedLambertian = (Lambertian * 0.5f + 0.5f);
        Lambertian = ModifiedLambertian * ModifiedLambertian;
        if (Lambertian > 0.f)
        {
            DirectLight += Lambertian * SunIntensity;
        }
    }

    for (int i = 0; i < InstanceData[InstanceIndex].PointLightsCount; ++i)
    {
        float3 PointLightPos = InstanceData[InstanceIndex].PointLightsPos[i].xyz;
        float3 ToPointLight = PointLightPos - WorldPos;
        float Lambertian = dot(normalize(ToPointLight), WorldNormal);
        float ModifiedLambertian = (Lambertian * 0.5f + 0.5f);
        Lambertian = ModifiedLambertian * ModifiedLambertian;
        if (Lambertian > 0.f)
        {
            float DistToLight = length(ToPointLight);
            float AttLin = InstanceData[InstanceIndex].PointLightsAttLin[i];
            float AttQuad = InstanceData[InstanceIndex].PointLightsAttQuad[i];
            float Attenuation = 1.f / (1.f + AttLin * DistToLight + AttQuad * DistToLight * DistToLight);
            DirectLight += Lambertian * Attenuation;
        }
    }

    return DirectLight;
}

// albedo texture color -> shaded color
float3 ShadeDynamicModel(float3 Color, float3 WorldPos, float3 WorldNormal, uint InstanceIndex)
{
    WorldNormal = normalize(WorldNormal);
    float3 AmbientTerm = AmbientLight(WorldNormal, InstanceIndex);
    float DiffuseTerm = DiffuseLight(WorldPos, WorldNormal, InstanceIndex);
    // TODO(Kevin): heuristically boost ambient term contribution as a function of direct lights (diffuse term)
    //              so direct light doesn't swamp the ambient and cause harsh appearance
    float3 Light = AmbientTerm + float3(DiffuseTerm, DiffuseTerm, DiffuseTerm);
    return Color * Light;
}

float4 main(vs_out Input) : SV_Target
{
    float4 AlbedoTexSample = Tex0.Sample(Sampler0, Input.UV0);
    float3 Albedo = AlbedoTexSample.xyz;
    float FinalAlpha = AlbedoTexSample.a;
    float LightMap = Tex1.Sample(Sampler1, Input.UV1).r;
    float3 FinalColor = ShadeDynamicModel(Albedo, Input.WorldP, Input.WorldN, Input.InstanceID);
    FinalColor = mul(FinalColor, LightMap);
    FinalColor = pow(FinalColor, 1.0/2.6); // gamma correction
    return float4(FinalColor, FinalAlpha);
}
