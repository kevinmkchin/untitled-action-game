#version 330

layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec2 vertex_uv;
layout (location = 2) in vec3 vertex_normal;

uniform mat4 viewMatrix;
uniform mat4 projMatrix;

out vec2 uv;

void main()
{
    uv = vertex_uv;

    vec4 vertInClipSpace = projMatrix * viewMatrix * vec4(vertex_pos, 1.0);
    gl_Position = vertInClipSpace;
}
