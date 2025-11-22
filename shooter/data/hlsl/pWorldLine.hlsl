#pragma pack_matrix(column_major)

cbuffer ConstantBuffer0 : register(b0)
{
    float4x4 ViewFromWorld;
    float4x4 ProjFromView;
    float2 ViewportSize;
    float FadeOutZ;
};

struct vs_out
{
    float4 P : SV_Position;
    float4 Color : COLOR0;
    float DistanceFromEye : EYE_DISTANCE;
};

float4 main(vs_out Input) : SV_Target
{
    // float FallOffZ = FadeOutZ;
    // float t = (Input.P.z - FallOffZ) / (1.0 - FallOffZ);
    // Input.Color.a = min(Input.Color.a - t, 1);
#ifndef THICK_LINES
    float FadeMult = clamp(exp(-pow((Input.DistanceFromEye * 0.0001), 5)), 0.0, 1.0);
    Input.Color.a *= FadeMult;
#endif
    return Input.Color;
}
