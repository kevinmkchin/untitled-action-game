#pragma pack_matrix(column_major)

cbuffer ConstantBuffer0 : register(b0)
{
    float4x4 ProjFromWorld;
    float2 ViewportSize;
    float FadeOutZ;
};

struct vs_in
{
    // Per-vertex
    float2 Corner : CORNER_T_SIGN;
    // Per-instance
    float3 WorldP0 : POSITION0;
    float3 WorldP1 : POSITION1;
    float Thickness : THICKNESS;
    float4 Color : COLOR;
};

struct vs_out
{
    float4 P : SV_Position;
    float4 Color : COLOR0;
};

vs_out main(vs_in Input)
{
    float CornerT = Input.Corner.x;
    float CornerSign = Input.Corner.y;

    float4 Clip0 = mul(ProjFromWorld, float4(Input.WorldP0, 1.0));
    float4 Clip1 = mul(ProjFromWorld, float4(Input.WorldP1, 1.0));

    float2 Ndc0 = Clip0.xy / Clip0.w;
    float2 Ndc1 = Clip1.xy / Clip1.w;

    float2 Dir = normalize(Ndc1 - Ndc0);
    float2 Orthogonal = float2(-Dir.y, Dir.x);

    float2 PixelToNDC = float2(2.0 / ViewportSize.x, 2.0 / ViewportSize.y);
    float HalfWidthPx = Input.Thickness * 0.5;
    float2 OffsetNDC = Orthogonal * HalfWidthPx * PixelToNDC;

    float4 ClipBase = lerp(Clip0, Clip1, CornerT);
    float2 NdcBase = ClipBase.xy / ClipBase.w;

    float2 NdcFinal = NdcBase + CornerSign * OffsetNDC;
    float4 ClipFinal = ClipBase;
    ClipFinal.xy = NdcFinal * ClipBase.w;

    vs_out Output;
    Output.P = ClipFinal;
    Output.Color = Input.Color;
    return Output;
}
