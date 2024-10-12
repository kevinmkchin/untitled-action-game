#version 330

layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec2 vertex_uv1;
layout (location = 3) in vec2 vertex_uv2;

uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec2 uv1;
out vec2 uv2;

void main()
{
    uv1 = vertex_uv1;
    uv2 = vertex_uv2;

    vec4 vertInClipSpace = projMatrix * viewMatrix * vec4(vertex_pos, 1.0);
    gl_Position = vertInClipSpace;
}
