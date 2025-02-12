#pragma once

struct enemy_t
{
    void Init();
    void Destroy();

    vec3 Position;
    quat Orientation;

    dynamic_array<float> SmoothPath;
    int SmoothPathCount;
    int SmoothPathIter;
    float TimeSinceLastPathFind = 0.f;
};

void UpdateAllEnemies();

extern dynamic_array<enemy_t> Enemies;

