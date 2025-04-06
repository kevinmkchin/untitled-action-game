#include "particles.h"

// external
fixed_array<particle> Particles;

u32 NextParticleIndex;

void UpdateParticles()
{
    for (u32 ParticleSpawnIndex = 0;
         ParticleSpawnIndex < 1;
         ++ParticleSpawnIndex)
    {
        particle *Particle = Particles.data + NextParticleIndex++;
        if (NextParticleIndex >= Particles.lenu())
            NextParticleIndex = 0;
        Particle->P = vec3();
        Particle->dP = vec3(0.f, 32.f, RNG.NextFloat(-32.f,32.f));
        Particle->Color = vec4(0,0,0,1.4f);
        Particle->dColor = vec4(0,0,0,-1.35f);
    }

    for (u32 i = 0; i < Particles.lenu(); ++i)
    {
        particle *Particle = Particles.data + i;

        // Simulate particles
        Particle->P += DeltaTime * Particle->dP;
        Particle->Color += DeltaTime * Particle->dColor;
        vec4 Color;
        Color.x = GM_clamp(Particle->Color.x, 0.f, 1.f);
        Color.y = GM_clamp(Particle->Color.y, 0.f, 1.f);
        Color.z = GM_clamp(Particle->Color.z, 0.f, 1.f);
        Color.w = GM_clamp(Particle->Color.w, 0.f, 1.f);

        // Render particles
        SupportRenderer.DrawSolidRect(Particle->P, vec3(1,0,0), 3.f, Color);

    }
}
