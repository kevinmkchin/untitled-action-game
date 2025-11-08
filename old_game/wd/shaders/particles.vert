#version 450

layout (location = 0) in vec3 Particle_Position;
layout (location = 1) in vec4 Particle_Color;
layout (location = 2) in vec2 Particle_UV;

uniform mat4 ClipFromWorld;

out vec4 Color;
out vec2 UV;

void main()
{
    Color = Particle_Color;
    UV = Particle_UV;

    gl_Position = ClipFromWorld * vec4(Particle_Position, 1.0);
}
