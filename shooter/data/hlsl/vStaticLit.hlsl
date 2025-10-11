#pragma pack_matrix(column_major)

cbuffer CameraCB : register(b0)
{
    float4x4 ProjView;
};

struct vs_in
{
    float3 P : POSITION;
    float3 N : NORMAL;
    float2 UV0 : TEXCOORD0;
    float2 UV1 : TEXCOORD1;
};

struct vs_out
{
    float4 P : SV_Position;
    float3 N : NORMAL;
    float2 UV0 : TEXCOORD0;
    float2 UV1 : TEXCOORD1;
};

vs_out main(vs_in Input)
{
    vs_out Output;
    Output.P = mul(ProjView, float4(Input.P, 1.0)); // Note: D3D NDC z is [0,1]
    Output.UV0 = Input.UV0; // I could flip UVs in the shader
    return Output;
}
