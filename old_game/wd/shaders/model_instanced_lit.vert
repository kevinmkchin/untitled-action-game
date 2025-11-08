#version 460

layout (location = 0) in vec3 Model_Position;
layout (location = 1) in vec2 Model_UV;
layout (location = 2) in vec3 Model_Normal;

struct model_instance_data_t
{
    mat4 WorldFromModel;
    vec4 PointLightsPos[4];
    float AmbientCube[6];
    int DoSunLight;
    int PointLightsCount;
    float PointLightsAttLin[4];
    float PointLightsAttQuad[4];
    float _padding_[4];
};

layout(std430, binding = 0) buffer InstanceDataSSBO 
{
    model_instance_data_t Instances[];
};

// the base instance is used (calculated on CPU by SSBO offset / sizeof(type)) so that
// any number of arbitrary types can be allocated on the 
uniform int BaseInstanceIndex;
uniform mat4 ProjFromView;
uniform mat4 ViewFromWorld;

out flat int InstanceIndex;
out vec2 TexCoords;
out vec3 WorldPos;
out vec3 WorldNormal;

void main()
{
    InstanceIndex = gl_InstanceID;
    mat4 WorldFromModel = Instances[BaseInstanceIndex + gl_InstanceID].WorldFromModel;
    vec4 World_Position = WorldFromModel * vec4(Model_Position, 1.0);
    WorldPos = World_Position.xyz;
    WorldNormal = transpose(inverse(mat3(WorldFromModel))) * Model_Normal;
    TexCoords = Model_UV;
    gl_Position = ProjFromView * ViewFromWorld * World_Position;
}
