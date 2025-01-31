#pragma once



enum entity_types_t
{
    POINT_PLAYER_SPAWN
};

struct level_entity_t
{
    entity_types_t Type;

    vec3 Position;
    vec3 Rotation;
};

