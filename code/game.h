#pragma once


void OpenGame();
void CloseGame();
void LoadLevel();

void DoGameLoop();

// private
void UnloadPreviousLevel();
void CreateAndRegisterLevelCollider();
void CreateAndRegisterPlayerPhysicsController();
void PrePhysicsTick();
void PostPhysicsTick();
void UpdateGameGUI();
void RenderGameLayer();

extern std::vector<face_batch_t> GameLevelFaceBatches;
extern bool GameLoopCanRun;