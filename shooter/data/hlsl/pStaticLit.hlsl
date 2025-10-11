#pragma pack_matrix(column_major)

Texture2D Tex0 : register(t0); // SRV slot 0
Texture2D Tex1 : register(t1); // SRV slot 1

SamplerState Sampler0 : register(s0); // Sampler slot 0
SamplerState Sampler1 : register(s1); // Sampler slot 1

struct vs_out
{
    float4 P : SV_Position;
    float3 N : NORMAL;
    float2 UV0 : TEXCOORD0;
    float2 UV1 : TEXCOORD1;
};

float4 main(vs_out Input) : SV_Target
{
    float4 Albedo = Tex0.Sample(Sampler0, Input.UV0);
    float4 LightMap = Tex1.Sample(Sampler1, Input.UV1);
    return mul(Albedo, LightMap.r);
}
