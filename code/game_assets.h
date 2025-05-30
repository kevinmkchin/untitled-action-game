#pragma once

#include "common.h"

#include "anim.h"
#include "resources.h"
#include "gpu_resources.h"
// Game Assets Database

Mix_Chunk *Mixer_LoadChunk(const char *filepath);

struct db_tex_t
{
    u32 persistId = 0;
    GPUTexture gputex;
};

// perhaps an enum GODFLESH_MODELS_SKINNED as well

enum GODFLESH_MODELS_TEXTURED : u16
{
    MT_INVALID,

    MT_WPN_TYPE1,
    MT_WPN_ROCKETLAUNCHER,
    MT_PRJ_NAIL,
    MT_PRJ_ROCKET,

    MT_ATTACKER_CORPSE,

    MT_GENERIC_GIB_0,
    MT_GENERIC_GIB_1,
    MT_GENERIC_GIB_2,

    MT_COUNT
    // perhaps modders can add more models with u16 values > MT_COUNT
};

struct asset_db_t
{
    // TODO(Kevin): probably just replace this with a fixed_array...
    std::map<u32, db_tex_t> Textures;

    db_tex_t LoadNewTexture(const char *path);
    db_tex_t GetTextureById(u32 persistId);

    void LoadAllResources();

public:
    db_tex_t DefaultEditorTexture;
    GPUTexture DefaultMissingTexture; // Conceptually, missing texture is not a persisted resource

    Mix_Chunk *Sfx_Shoot0;
    Mix_Chunk *Sfx_Ricochet[3];
    Mix_Chunk *Sfx_ShootRocket;
    Mix_Chunk *Sfx_ExplodeRocket;


    skeleton_t *Skeleton_Humanoid = nullptr;
    skinned_model_t *Model_Attacker = nullptr;

    fixed_array<ModelGLTF> ModelsTextured;

private:
    u32 TexturePersistIdCounter;
};

extern asset_db_t Assets;

