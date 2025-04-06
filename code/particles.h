#pragma once

struct particle
{
    vec3 P;
    vec3 dP;
    vec4 Color;
    vec4 dColor;
};

void UpdateParticles();

extern fixed_array<particle> Particles;
