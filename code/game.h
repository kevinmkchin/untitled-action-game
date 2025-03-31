#pragma once


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

extern std::vector<face_batch_t> GameLevelFaceBatches;
extern bool GameLoopCanRun;
extern map_info_t RuntimeMapInfo;
extern fixed_array<model_instance_data_t> GlobalStaticInstances;
extern fixed_array<model_instance_data_t> GlobalDynamicInstances;
