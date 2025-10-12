#pragma pack_matrix(column_major)

cbuffer ConstantBuffer0 : register(b0)
{
    float4x4 ClipFromScreen2D;
};

struct vs_in
{
    float2 P : POSITION;
    float2 UV0 : TEXCOORD0;
    float4 Color : COLOR;
};

struct vs_out
{
    float4 P : SV_Position;
    float2 UV0 : TEXCOORD0;
    float4 ColorMultiplier : COLOR;
};

vs_out main(vs_in Input, uint InstanceID : SV_InstanceID)
{
    vs_out Output;
    Output.P = mul(ClipFromScreen2D, float4(Input.P, 0.0f, 1.0f));
    Output.UV0 = Input.UV0;
    Output.ColorMultiplier = Input.Color;
    return Output;
}
