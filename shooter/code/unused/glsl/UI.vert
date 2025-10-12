#version 330 core

layout (location = 0) in vec2 Pos;
layout (location = 1) in vec2 UV;
layout (location = 2) in vec4 Color;

out vec2 FragUV;
out vec4 FragColorMultiplier;

uniform mat4 ClipFromScreen2D;

void main()
{
    gl_Position = ClipFromScreen2D * vec4(Pos, 0.0, 1.0);
    FragUV = UV;
    FragColorMultiplier = Color;
}
