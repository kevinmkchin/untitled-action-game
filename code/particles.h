#pragma once

#include "utility.h"

struct particle_vertex
{
    vec3 WorldPos;
    vec4 Color;
    vec2 UV;
};

struct particle
{
    vec4 dColor;
    vec4 Color;
    vec3 P;
    vec3 dP;
    vec3 ddP;
    float HalfWidth;
    float dHalfWidth;
    float Life;
    bool JustEmitted;
    u32 _pad_; // Enum of anim_sprite
};

// blood should spurt out at very fast velocity and then dampen quickly to almost horizontally stationary

// sometimes I want to do a burst of particles in one frame
// othertimes I want the emitter to stay around and continue emitting particles
struct particle_emitter
{
    vec3 WorldP;
    vec3 PSpread;
    vec3 dP;
    vec3 dPSpread;
    vec3 ddP;
    vec4 Color;
    vec4 ColorSpread;
    vec4 dColor;
    float HalfWidth;
    float HalfWidthSpread;
    float dHalfWidth;
    float Timer = 0.f;
    float ParticleLifeTimer = 2.f;
    // holds info about what particles to create
    // emitter shape (plane, cone, etc.)
    // rate of creation (particles / second) increase probability of 
    //      emitting new particles as time goes on
};

struct particle_buffer
{
    // circular buffer

    void Alloc(u32 Count, u32 MaxEmitters, MemoryType MemType)
    {
        Free();
        Particles = fixed_array<particle>(Count, MemType);
        Particles.setlen(Count);
        Emitters = fixed_array<particle_emitter>(MaxEmitters, MemType);
    }

    void Free()
    {
        Particles.free();
        Emitters.free();
    }

    fixed_array<particle> Particles;
    u32 NextParticleIndex = 0;

    fixed_array<particle_emitter> Emitters;
};

void UpdateParticles(particle_buffer &ParticleState, random_series &EmitterRNG);
void AssembleParticleQuads(
    particle_buffer &Collection, 
    vec3 QuadDirection,
    fixed_array<particle_vertex> &QuadAssemblyBuf);


