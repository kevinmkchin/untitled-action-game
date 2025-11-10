#pragma pack_matrix(column_major)

cbuffer ConstantBuffer0 : register(b0)
{
    float4x4 ProjFromWorld;
    float2 ViewportSize;
    float FadeOutZ;
};

struct vs_out
{
    float4 P : SV_Position;
    float4 Color : COLOR0;
};

float4 main(vs_out Input) : SV_Target
{
    // float FallOffZ = FadeOutZ;
    // float t = (Input.P.z - FallOffZ) / (1.0 - FallOffZ);
    // Input.Color.a = min(Input.Color.a - t, 1);
    return Input.Color;
}
