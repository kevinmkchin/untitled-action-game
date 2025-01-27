#pragma once


void OpenGame();
void CloseGame();
void LoadLevel();

void DoGameLoop();

// private
void PrePhysicsTick();
void PostPhysicsTick();
void RenderGameLayer();

extern std::vector<vec3> GameLevelColliderPoints;
extern std::vector<FlatPolygonCollider> GameLevelColliders;
extern std::vector<face_batch_t> GameLevelFaceBatches;
