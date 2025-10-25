#pragma pack_matrix(column_major)

// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules
cbuffer ConstantBuffer0 : register(b0)
{
    float4x4 ProjFromWorld;
    float3 DirectionToSun;
};

cbuffer ConstantBuffer1 : register(b1)
{
    uint BaseInstance;
};

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
    float3 WorldP : POSITION;
    float3 WorldN : NORMAL;
    float2 UV0 : TEXCOORD0;
    float2 UV1 : TEXCOORD1;
    nointerpolation uint InstanceID : SV_InstanceID;
};

float3x3 Inverse3x3(float3x3 M)
{
    float3x3 Cof;

    Cof[0][0] =  M[1][1]*M[2][2] - M[1][2]*M[2][1];
    Cof[0][1] = -M[1][0]*M[2][2] + M[1][2]*M[2][0];
    Cof[0][2] =  M[1][0]*M[2][1] - M[1][1]*M[2][0];

    Cof[1][0] = -M[0][1]*M[2][2] + M[0][2]*M[2][1];
    Cof[1][1] =  M[0][0]*M[2][2] - M[0][2]*M[2][0];
    Cof[1][2] = -M[0][0]*M[2][1] + M[0][1]*M[2][0];

    Cof[2][0] =  M[0][1]*M[1][2] - M[0][2]*M[1][1];
    Cof[2][1] = -M[0][0]*M[1][2] + M[0][2]*M[1][0];
    Cof[2][2] =  M[0][0]*M[1][1] - M[0][1]*M[1][0];

    float Det = dot(M[0], Cof[0]);
    return transpose(Cof) / Det;
}

vs_out main(vs_in Input, uint InstanceID : SV_InstanceID)
{
    float4x4 WorldFromModel = InstanceData[BaseInstance + InstanceID].WorldFromModel;
    float4 WorldPosition = mul(WorldFromModel, float4(Input.P, 1.0));

    vs_out Output;
    Output.P = mul(ProjFromWorld, WorldPosition); // Note: D3D NDC z is [0,1]
    Output.WorldP = WorldPosition.xyz;
    Output.WorldN = mul(transpose(Inverse3x3((float3x3)WorldFromModel)), Input.N);
    Output.UV0 = Input.UV0; // I could flip UVs in the shader
    Output.UV1 = Input.UV1;
    Output.InstanceID = BaseInstance + InstanceID;
    return Output;
}
