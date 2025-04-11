#include "particles.h"


static void CreateParticleFromEmitter(
    particle_buffer &ParticleState, 
    particle_emitter &Emitter,
    random_series &EmitterRNG)
{
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
    Particle.ddP = Emitter.ddP;
    Particle.Color = Emitter.Color + vec4(
        EmitterRNG.frand() * Emitter.ColorSpread.x,
        EmitterRNG.frand() * Emitter.ColorSpread.y,
        EmitterRNG.frand() * Emitter.ColorSpread.z,
        EmitterRNG.frand() * Emitter.ColorSpread.w);
    Particle.dColor = Emitter.dColor;
    Particle.HalfWidth = Emitter.HalfWidth +
        EmitterRNG.frand() * Emitter.HalfWidthSpread;
    Particle.dHalfWidth = Emitter.dHalfWidth;
    Particle.Life = Emitter.ParticleLifeTimer;
    Particle.JustEmitted = true;
}

void UpdateParticles(particle_buffer &ParticleState, random_series &EmitterRNG)
{
    // Update Emitters
    for (u32 EmitterIndex = 0; 
         EmitterIndex < ParticleState.Emitters.lenu();
         ++EmitterIndex)
    {
        particle_emitter &Emitter = ParticleState.Emitters[EmitterIndex];

        // if (ShouldEmitNewParticle(Emitter))
        // {
            CreateParticleFromEmitter(ParticleState, Emitter, EmitterRNG);
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

        if (Particle->JustEmitted)
        {
            Particle->JustEmitted = false;
            continue;
        }

        Particle->Life -= DeltaTime;
        if (Particle->Life > 0.f)
        {
            // Simulate particles
            Particle->P += DeltaTime * Particle->dP;
            Particle->dP += DeltaTime * Particle->ddP;
            Particle->Color += DeltaTime * Particle->dColor;
            Particle->HalfWidth += DeltaTime * Particle->dHalfWidth;
        }
    }
}

static inline void AssembleQuadForParticle(
    fixed_array<particle_vertex> &QuadAssemblyBuf,
    particle *Particle, vec3 Up, vec3 Right)
{
    particle_vertex BL;
    particle_vertex BR;
    particle_vertex TL;
    particle_vertex TR;

    BL.WorldPos = Particle->P - Up - Right;
    BR.WorldPos = Particle->P - Up + Right;
    TL.WorldPos = Particle->P + Up - Right;
    TR.WorldPos = Particle->P + Up + Right;

    vec4 Color;
    Color.x = GM_clamp(Particle->Color.x, 0.f, 1.f);
    Color.y = GM_clamp(Particle->Color.y, 0.f, 1.f);
    Color.z = GM_clamp(Particle->Color.z, 0.f, 1.f);
    Color.w = GM_clamp(Particle->Color.w, 0.f, 1.f);
    BL.Color = Color;
    BR.Color = Color;
    TL.Color = Color;
    TR.Color = Color;

    BL.UV = vec2(0,0);
    BR.UV = vec2(1,0);
    TL.UV = vec2(0,1);
    TR.UV = vec2(1,1);

    QuadAssemblyBuf.put(BL);
    QuadAssemblyBuf.put(BR);
    QuadAssemblyBuf.put(TL);
    QuadAssemblyBuf.put(TL);
    QuadAssemblyBuf.put(BR);
    QuadAssemblyBuf.put(TR);
}

void AssembleParticleQuads(
    particle_buffer &Collection,
    vec3 QuadDirection,
    fixed_array<particle_vertex> &QuadAssemblyBuf)
{
    vec3 Right = CalculateTangent(QuadDirection);
    vec3 Up = Normalize(Cross(QuadDirection, Right));

    for (u32 i = 0; i < Collection.Particles.lenu(); ++i)
    {
        particle *Particle = Collection.Particles.data + i;

        if (Particle->Life > 0.f)
        {
            if (QuadAssemblyBuf.lenu() + 6 > QuadAssemblyBuf.cap())
                break;

            float ParticleHalfSize = Particle->HalfWidth;
            AssembleQuadForParticle(QuadAssemblyBuf, Particle,
                Up * ParticleHalfSize, Right * ParticleHalfSize);
            // vec4 ClipCoord = ClipFromWorldMatrix * vec4(Particle->P, 1.f);
            // vec3 ScreenClip = ClipCoord.xyz / ClipCoord.w;
        }
    }
}


