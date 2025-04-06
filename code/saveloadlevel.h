#pragma once


bool BuildGameMap(const char *path);
bool LoadGameMap(struct game_state *MapInfo, const char *path);

extern std::vector<vec3> LoadingLevelColliderPoints;
extern std::vector<u32> LoadingLevelColliderSpans;
