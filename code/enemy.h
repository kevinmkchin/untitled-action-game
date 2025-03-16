#pragma once

struct enemy_t
{
    void Init();
    void Destroy();

    vec3 Position;
    quat Orientation;

    JPH::Character *RigidBody;

    dynamic_array<float> SmoothPath;
    int SmoothPathCount;
    int SmoothPathIter;
    float TimeSinceLastPathFind = 0.f;

private:
    void AddToPhysicsSystem();
    void RemoveFromPhysicsSystem();
};

void PrePhysicsTickAllEnemies();
void PostPhysicsTickAllEnemies();
void DebugDrawEnemyColliders();

extern dynamic_array<enemy_t> Enemies;

