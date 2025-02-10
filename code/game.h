#pragma once


void InitializeGame();
void DestroyGame();
void LoadLevel(const char *MapPath);
void UnloadPreviousLevel();

void DoGameLoop();

// private
void CreateAndRegisterLevelCollider();
bool CreateRecastNavMesh();
void DestroyRecastNavMesh();
void DetourTesting();
void CreateAndRegisterPlayerPhysicsController();
void PrePhysicsTick();
void PostPhysicsTick();
void UpdateGameGUI();
void RenderGameLayer();

extern std::vector<face_batch_t> GameLevelFaceBatches;
extern bool GameLoopCanRun;
