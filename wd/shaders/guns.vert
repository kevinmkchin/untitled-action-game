#version 330

layout (location = 0) in vec3 Model_Position;
layout (location = 1) in vec2 Model_UV;
layout (location = 2) in vec3 Model_Normal;

uniform mat4 WorldFromView;
uniform mat4 ViewFromModel;
uniform mat4 ProjFromView;

out vec3 Light;

void main()
{
    vec4 VertInClipSpace = ProjFromView * ViewFromModel * vec4(Model_Position, 1.0);
    gl_Position = VertInClipSpace;

    const vec4 World_SunDir = vec4(0.5, -0.7, -0.8, 0.0);
    vec3 World_Normal = transpose(inverse(mat3(WorldFromView * ViewFromModel))) * Model_Normal;
    float AmbientIntensity = 0.6;
    float DiffuseIntensity = max(0.0, dot(normalize(World_Normal), normalize(-vec3(World_SunDir))));
    vec3 LightAMB = vec3(1.0) * AmbientIntensity;
    vec3 LightDIFF = vec3(0.4) * DiffuseIntensity;
    Light = min(LightAMB + LightDIFF, vec3(1.0));
    // Light = vec3(1.0);
}
