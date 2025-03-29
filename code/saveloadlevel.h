#pragma once

struct map_info_t
{
    vec3 PlayerStartPosition;
    vec3 PlayerStartRotation;

    lc_volume_t *LightCacheVolume = nullptr;

    vec3 DirectionToSun;
    fixed_array<static_point_light_t> PointLights;
};

bool BuildGameMap(const char *path);
bool LoadGameMap(map_info_t *MapInfo, const char *path);

extern std::vector<vec3> LoadingLevelColliderPoints;
extern std::vector<u32> LoadingLevelColliderSpans;
