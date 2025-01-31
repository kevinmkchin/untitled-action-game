#pragma once





// struct player_spawn_ent_t
// {
//     vec3 Pos;
//     float Yaw;
// };



enum entity_types_t
{
    POINT_PLAYER_SPAWN
};

struct billboard_t
{
    GPUTexture Tex;
    float Sz;
};

struct level_entity_t
{
    entity_types_t Type;

    vec3 Position;
    vec3 Rotation;
};

