#pragma once

constexpr u32 EnemyFlag_Dead     = 0x0001;
constexpr u32 EnemyFlag_Active   = 0x0002;
// constexpr u32  = 0x0004;
// constexpr u32  = 0x0008;
// constexpr u32  = 0x0010;
// constexpr u32  = 0x0020;
// constexpr u32  = 0x0040;
// constexpr u32  = 0x0080;
// constexpr u32  = 0x0100;
// constexpr u32  = 0x0200;
// constexpr u32  = 0x1000;
// constexpr u32  = 0x2000;
// constexpr u32  = 0x4000;
// constexpr u32  = 0x8000;

constexpr u32 BAD_UINDEX = 0xFFFFFFFF;

struct enemy_t
{
    u32 Index = BAD_UINDEX;
    u32 Flags = 0x0;

    vec3 Position;
    quat Orientation;

    float Health;
    float DeadTimer;
    bool RemainAfterDead;

    animator_t *Animator;

    // Jolt Physics
    JPH::Character *RigidBody;

    // Detour pathfinding
    dynamic_array<float> SmoothPath;
    int SmoothPathCount;
    int SmoothPathIter;
    float TimeSinceLastPathFind = 0.f;

    // each instance should have their own animator_t
};

void NonPhysicsTickAllEnemies(game_state *GameState);
void PrePhysicsTickAllEnemies(game_state *GameState);
void PostPhysicsTickAllEnemies(game_state *GameState);
void RenderEnemies(game_state *GameState, const mat4 &ProjFromView, const mat4 &ViewFromWorld);
void DebugDrawEnemyColliders();

void HurtEnemy(game_state *GameState, u32 EnemyIndex, float Damage);
void KillEnemy(game_state *GameState, u32 EnemyIndex);

struct global_enemy_state_t
{
    static constexpr int MaxEnemies = 64;
    static constexpr int MaxCharacterBodies = 32;

    fixed_array<enemy_t> Enemies;
    fixed_array<JPH::Character *> CharacterBodies;
    // TODO(Kevin): separate array per collider type?

    void Init(); // Call once at start of game
    void Destroy(); // Call once at end of game

    void RemoveAll(); // Call before changing levels etc.

    void SpawnEnemy(game_state *GameState);
    void RemoveEnemy(u32 EnemyIndex);

    JPH::Character *NextAvailableCharacterBody();
    void RemoveCharacterBodyFromSimulation(JPH::Character *CharacterBody);

private:
    static constexpr float AttackerHeightStanding = 1.7f;
    static constexpr float AttackerCapsuleRadiusStanding = 0.3f;
    static constexpr float AttackerCapsuleHalfHeightStanding = (AttackerHeightStanding 
        - AttackerCapsuleRadiusStanding * 2.f) * 0.5f;
};

extern global_enemy_state_t EnemySystem;


