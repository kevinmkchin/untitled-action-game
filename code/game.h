#pragma once


void InitializeGame();
void DestroyGame();
void LoadLevel(const char *MapPath);
void UnloadPreviousLevel();

void DoGameLoop();

// private
void CreateAndRegisterLevelCollider();
void CreateAndRegisterPlayerPhysicsController();

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
