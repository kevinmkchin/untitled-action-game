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

// NOTE(Kevin): This NDC pixel offset is really bad when W values of the two line end points are far apart. 
//              What is calculated to be horizontal NDC corresponding to 2 pixels ends up being huge when
//              applied in clip space. This line thickening is really shoddy!
#ifdef THICK_LINES

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

vs_out main(vs_in Input)
{
#if 0
    float CornerT = Input.Corner.x;
    float CornerSign = Input.Corner.y;

    float4 View0 = mul(ViewFromWorld, float4(Input.WorldP0, 1.0));
    float4 View1 = mul(ViewFromWorld, float4(Input.WorldP1, 1.0));

    float2 ViewXY0 = View0.xy;
    float2 ViewXY1 = View1.xy;

    float2 ViewXYDir = normalize(ViewXY1 - ViewXY0);
    float2 Orthogonal = float2(-ViewXYDir.y, ViewXYDir.x);

    Orthogonal = Orthogonal * Input.Thickness * 0.15;

    float4 ViewBase = lerp(View0, View1, CornerT);
    ViewBase += CornerSign * float4(Orthogonal, 0, 0);
    ViewBase.z += 0.1;

    vs_out Output;
    Output.P = mul(ProjFromView, ViewBase);
    Output.Color = Input.Color;
    return Output;
#else
    float CornerT = Input.Corner.x;
    float CornerSign = Input.Corner.y;

    float4x4 ProjFromWorld = mul(ProjFromView, ViewFromWorld);

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
    ClipFinal.z -= 0.01; // Bring the line ever so slightly forward so it is not occluded

    vs_out Output;
    Output.P = ClipFinal;
    Output.Color = Input.Color;
    return Output;
#endif
}

#else

struct vs_in
{
    float3 WorldP : POSITION;
    float4 Color : COLOR;
};

vs_out main(vs_in Input)
{
    float4 ViewP = mul(ViewFromWorld, float4(Input.WorldP, 1.0f));

    vs_out Output;
    Output.P = mul(ProjFromView, ViewP);
    Output.Color = Input.Color;
    Output.DistanceFromEye = -ViewP.z;
    return Output;
}

#endif
