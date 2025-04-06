#pragma once

struct game_state
{
    random_series ParticlesRNG;
    particle_buffer BloodParticles;

    // Runtime map info
    vec3 PlayerStartPosition;
    vec3 PlayerStartRotation;
    lc_volume_t *LightCacheVolume = nullptr;
    vec3 DirectionToSun;
    fixed_array<static_point_light_t> PointLights;
};
extern game_state GameState;

extern std::vector<face_batch_t> GameLevelFaceBatches;
extern bool GameLoopCanRun;
extern fixed_array<model_instance_data_t> GlobalStaticInstances;
extern fixed_array<model_instance_data_t> GlobalDynamicInstances;



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
