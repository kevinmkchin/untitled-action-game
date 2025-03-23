#version 330

layout (location = 0) in vec3 Model_Position;
layout (location = 1) in vec2 Model_UV;
layout (location = 2) in vec3 Model_Normal;

uniform mat4 ProjFromView;
uniform mat4 ViewFromWorld;
uniform mat4 WorldFromModel;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 WorldNormal;

void main()
{
    vec4 World_Position = WorldFromModel * vec4(Model_Position, 1.0);
    WorldPos = World_Position.xyz;
    WorldNormal = transpose(inverse(mat3(WorldFromModel))) * Model_Normal;
    TexCoords = Model_UV;
    gl_Position = ProjFromView * ViewFromWorld * World_Position;
}
