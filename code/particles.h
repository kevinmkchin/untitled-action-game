#pragma once


struct particle_vertex
{
    vec3 WorldPos;
    vec4 Color;
    vec2 UV;
};

struct particle_vertex_stream
{
    /*
        Triple-buffered persistent buffer for vertex streaming
        https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming#Persistent_mapped_streaming
    */

    void Alloc(size_t VertexCountPerFrame)
    {
        ASSERT(!(VAO || VBO));

        FrameSize = VertexCountPerFrame * sizeof(particle_vertex);
        TotalSize = NumFrames * FrameSize;

        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferStorage(GL_ARRAY_BUFFER, TotalSize, nullptr,
            GL_MAP_WRITE_BIT |
            GL_MAP_PERSISTENT_BIT |
            GL_MAP_COHERENT_BIT);
        MappedPtr = (char*) glMapBufferRange(GL_ARRAY_BUFFER, 0, TotalSize,
            GL_MAP_WRITE_BIT |
            GL_MAP_PERSISTENT_BIT |
            GL_MAP_COHERENT_BIT);

        constexpr GLsizei StrideInBytes = sizeof(particle_vertex);
        glEnableVertexAttribArray(0);
        glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexAttribBinding(0, BindingIndex);
        glEnableVertexAttribArray(1);
        glVertexAttribFormat(1, 4, GL_FLOAT, GL_FALSE, offsetof(particle_vertex, Color));
        glVertexAttribBinding(1, BindingIndex);
        glEnableVertexAttribArray(2);
        glVertexAttribFormat(2, 2, GL_FLOAT, GL_FALSE, offsetof(particle_vertex, UV));
        glVertexAttribBinding(2, BindingIndex);

        // Tell OpenGL how to interpret the vertex struct (binding index = 0)
        // Bind buffer to binding index 0 (no offset yet)
        glBindVertexBuffer(BindingIndex, VBO, 0, StrideInBytes);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void Draw(particle_vertex *VertexData, u32 VertexCount)
    {
        if (VertexCount == 0)
            return;
        ASSERT(VAO && VBO);

        size_t FrameIndex = CurrentFrame % NumFrames;
        size_t OffsetInBytes = FrameIndex * FrameSize;
        constexpr GLsizei StrideInBytes = sizeof(particle_vertex);

        memcpy(MappedPtr + OffsetInBytes, VertexData, VertexCount * StrideInBytes);

        glBindVertexArray(VAO);
        glBindVertexBuffer(BindingIndex, VBO, OffsetInBytes, StrideInBytes);
        glDrawArrays(GL_TRIANGLES, 0, VertexCount);
        glBindVertexArray(0);

        ++CurrentFrame;
        if (CurrentFrame >= FrameSize)
            CurrentFrame = 0;
    }

    void Free()
    {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }

private:
    GLuint VAO = 0;
    GLuint VBO = 0;
    char* MappedPtr = nullptr;

    static constexpr GLuint BindingIndex = 0;

    static constexpr size_t NumFrames = 3;
    size_t CurrentFrame = 0;
    size_t FrameSize = 0;
    size_t TotalSize = 0;
};

struct particle
{
    vec4 dColor;
    vec4 Color;
    vec3 P;
    vec3 dP;
    vec3 ddP;
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


