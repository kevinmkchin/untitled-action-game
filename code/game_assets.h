#pragma once

// Game Assets Database

struct db_tex_t
{
    u32 persistId = 0;
    GPUTexture gputex;
};

// perhaps an enum GODFLESH_MODELS_SKINNED as well

enum GODFLESH_MODELS_TEXTURED : u16
{
    MT_WPN_TYPE1,
    MT_PRJ_NAIL,

    MT_ATTACKER_CORPSE,

    MT_GENERIC_GIB_0,
    MT_GENERIC_GIB_1,
    MT_GENERIC_GIB_2,

    MT_COUNT
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


    skeleton_t *Skeleton_Humanoid = nullptr;
    skinned_model_t *Model_Attacker = nullptr;

    fixed_array<ModelGLTF> ModelsTextured;

private:
    u32 TexturePersistIdCounter;

private:
    void CreateEntityBillboardAtlasForSupportRenderer(BitmapHandle *BillboardBitmaps);
};

extern asset_db_t Assets;

