#pragma pack_matrix(column_major)

Texture2D Tex0 : register(t8);

SamplerState Sampler0 : register(s0);

struct vs_out
{
    float4 P : SV_Position;
    float2 UV0 : TEXCOORD0;
    float4 ColorMultiplier : COLOR;
};

float4 main(vs_out Input) : SV_Target
{
    float4 FinalColor = Tex0.Sample(Sampler0, Input.UV0);
    FinalColor = FinalColor * Input.ColorMultiplier;
    return FinalColor;
}
