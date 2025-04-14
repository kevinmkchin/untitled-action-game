#pragma once

#include "common.h"
#include "leveleditor.h"


bool BuildGameMap(level_editor_t *EditorState, const char *path);
bool LoadGameMap(struct game_state *MapInfo, const char *path);

extern std::vector<vec3> LoadingLevelColliderPoints;
extern std::vector<u32> LoadingLevelColliderSpans;
