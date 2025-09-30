#version 430

// NOTE(Kevin): All worldspace, lit, mesh-based entities are instanced via this shader!

layout (location = 0) in vec3 Model_Position;
layout (location = 1) in vec3 Model_Normal;
layout (location = 2) in vec2 Model_UV0;
layout (location = 3) in vec2 Model_UV1;

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

uniform int BaseInstanceIndex;
uniform mat4 ProjFromView;
uniform mat4 ViewFromWorld;

out flat int InstanceIndex;
out vec2 UV0;
out vec2 UV1;
out vec3 WorldPos;
out vec3 WorldNormal;

void main()
{
    InstanceIndex = gl_InstanceID;
    mat4 WorldFromModel = Instances[BaseInstanceIndex + gl_InstanceID].WorldFromModel;
    vec4 World_Position = WorldFromModel * vec4(Model_Position, 1.0);
    WorldPos = World_Position.xyz;
    WorldNormal = transpose(inverse(mat3(WorldFromModel))) * Model_Normal;
    UV0 = Model_UV0;
    UV1 = Model_UV1;
    gl_Position = ProjFromView * ViewFromWorld * World_Position;
}
