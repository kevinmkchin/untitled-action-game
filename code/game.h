#pragma once

#include "common.h"
#include "mem.h"
#include "particles.h"
#include "player.h"
#include "cam.h"
#include "instanced.h"
#include "lightmap.h"

struct game_state
{
    random_series ParticlesRNG;
    particle_buffer BloodParticles;
    persistent_vertex_stream BloodParticlesVB;
    fixed_array<particle_vertex> PQuadBuf;

    player_t Player;

    bool GameLoopCanRun = true;
    bool LevelLoaded = false;
    JPH::BodyID LevelColliderBodyId;

    fixed_array<model_instance_data_t> StaticInstances;
    fixed_array<model_instance_data_t> DynamicInstances;

    fixed_array<animator_t> AnimatorPool;

    // Runtime map info
    vec3 PlayerStartPosition;
    vec3 PlayerStartRotation;
    lc_volume_t *LightCacheVolume = nullptr;
    vec3 DirectionToSun;
    fixed_array<static_point_light_t> PointLights;
    std::vector<face_batch_t> GameLevelFaceBatches;

    // Testing
    int KillEnemyCounter = 0;

};



void InitializeGame();
void DestroyGame();
void LoadLevel(const char *MapPath);
void UnloadPreviousLevel();

void DoGameLoop();

// private
void CreateAndRegisterLevelCollider();

/** NonPhysicsTick runs once per frame.
    Input handling should be done here.
    LateNonPhysicsTick runs every frame but always after physics.
*/
void NonPhysicsTick();
void LateNonPhysicsTick();

/** Pre/PostPhysicsTick can run once, zero, or several times per frame
    depending on FixedDeltaTime. PrePhysicsTick should be used when applying
    force, torques, or other physics-related functions.
*/
void PrePhysicsTick();
void PostPhysicsTick();

void UpdateGameGUI();
void RenderGameLayer();
