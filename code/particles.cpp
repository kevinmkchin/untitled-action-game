#include "particles.h"


static void CreateParticleFromEmitter(
    particle_buffer &ParticleState, 
    particle_emitter &Emitter)
{
    random_series &EmitterRNG = GameState.ParticlesRNG;

    particle &Particle = ParticleState.Particles[ParticleState.NextParticleIndex++];
    if (ParticleState.NextParticleIndex >= ParticleState.Particles.lenu())
        ParticleState.NextParticleIndex = 0;

    Particle.P = Emitter.WorldP + vec3(
        EmitterRNG.frand() * Emitter.PSpread.x,
        EmitterRNG.frand() * Emitter.PSpread.y,
        EmitterRNG.frand() * Emitter.PSpread.z);
    Particle.dP = Emitter.dP + vec3(
        EmitterRNG.frand() * Emitter.dPSpread.x,
        EmitterRNG.frand() * Emitter.dPSpread.y,
        EmitterRNG.frand() * Emitter.dPSpread.z);
    Particle.Color = Emitter.Color + vec4(
        EmitterRNG.frand() * Emitter.ColorSpread.x,
        EmitterRNG.frand() * Emitter.ColorSpread.y,
        EmitterRNG.frand() * Emitter.ColorSpread.z,
        EmitterRNG.frand() * Emitter.ColorSpread.w);
    Particle.dColor = Emitter.dColor;
    Particle.Life = Emitter.ParticleLifeTimer;
}

void UpdateParticles(particle_buffer &ParticleState)
{
    // Update Emitters
    for (u32 EmitterIndex = 0; 
         EmitterIndex < ParticleState.Emitters.lenu();
         ++EmitterIndex)
    {
        particle_emitter &Emitter = ParticleState.Emitters[EmitterIndex];

        // if (ShouldEmitNewParticle(Emitter))
        // {
            CreateParticleFromEmitter(ParticleState, Emitter);
        // }

        Emitter.Timer -= DeltaTime;
        if (Emitter.Timer <= 0.f)
        {
            ParticleState.Emitters.delswap(EmitterIndex--);
        }
    }

    // Update Particles
    for (u32 i = 0; i < ParticleState.Particles.lenu(); ++i)
    {
        particle *Particle = ParticleState.Particles.data + i;

        if (Particle->Life > 0.f)
        {
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
        Particle->Life -= DeltaTime;

    }
}
