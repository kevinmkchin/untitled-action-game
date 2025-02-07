#pragma once

struct map_load_result_t
{
    bool Success = false;

    vec3 PlayerStartPosition;
    vec3 PlayerStartRotation;
};

bool BuildGameMap(const char *path);
map_load_result_t LoadGameMap(const char *path);

extern std::vector<vec3> LoadingLevelColliderPoints;
extern std::vector<u32> LoadingLevelColliderSpans;
