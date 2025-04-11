#include "game_assets.h"

// external
asset_db_t Assets;

Mix_Chunk *Mixer_LoadChunk(const char *filepath)
{
    Mix_Chunk *chunk = Mix_LoadWAV(filepath);
    if (chunk == NULL)
        printf("Failed to load sound effect! SDL_mixer error: %s\n", SDL_GetError());
    return chunk;
}

db_tex_t asset_db_t::GetTextureById(u32 persistId)
{
    auto textureiter = Textures.find(persistId);
    if (textureiter != Textures.end())
        return textureiter->second;
    
    db_tex_t missingTexture;
    missingTexture.persistId = persistId;
    missingTexture.gputex = DefaultMissingTexture;
    return missingTexture;
}

db_tex_t asset_db_t::LoadNewTexture(const char *path)
{
    db_tex_t tex;
    tex.persistId = ++TexturePersistIdCounter;

    BitmapHandle bitmapStorage;
    ReadImage(bitmapStorage, path);
    if (bitmapStorage.memory != NULL)
    {        
        CreateGPUTextureFromBitmap(&tex.gputex, (u8*)bitmapStorage.memory,
                        bitmapStorage.width, bitmapStorage.height,
                        GL_SRGB, (bitmapStorage.bitDepth == 3 ? GL_RGB : GL_RGBA),
                        GL_NEAREST_MIPMAP_LINEAR, GL_NEAREST, GL_UNSIGNED_BYTE);

        Assets.Textures.insert({tex.persistId, tex});
        LogMessage("Loaded %s with texture persist id %d", path, tex.persistId);
    }
    FreeImage(bitmapStorage);

    return tex;
}

void asset_db_t::CreateEntityBillboardAtlasForSupportRenderer(BitmapHandle *BillboardBitmaps)
{
    stbrp_rect *EntBilRects = NULL;
    for (int i = 0; i < 64; ++i)
    {
        BitmapHandle Bitmap = BillboardBitmaps[i];
        if (Bitmap.memory)
        {
            if (Bitmap.bitDepth != 4)
            {
                LogError("Entity billboard loading error: All entity billboard bitmaps must have RGBA components (32 bit depth).");
                continue;
            }
            stbrp_rect Rect;
            Rect.id = i; // entity_types_t
            Rect.w = Bitmap.width;
            Rect.h = Bitmap.height;
            arrput(EntBilRects, Rect);
        }
    }

    // Pack the rects
    i32 BillboardAtlasW = 512;
    i32 BillboardAtlasH = 512;
    stbrp_node *BillboardPackerNodes = NULL;
    arrsetlen(BillboardPackerNodes, BillboardAtlasW);
    stbrp_context BillboardPacker;
    stbrp_init_target(&BillboardPacker, BillboardAtlasW, BillboardAtlasH, 
        BillboardPackerNodes, (int)arrlenu(BillboardPackerNodes));
    stbrp_pack_rects(&BillboardPacker, EntBilRects, (int)arrlenu(EntBilRects));
    arrfree(BillboardPackerNodes);
    // EntBilRects is populated at this point

    // Take packed rect info; blit to BillboardAtlas and cache UV rect
    dynamic_array<u8> BillboardAtlasBuffer;
    BillboardAtlasBuffer.setlen(BillboardAtlasW * BillboardAtlasH * 4);
    memset(BillboardAtlasBuffer.data, 100, BillboardAtlasBuffer.lenu());
    for (int i = 0; i < arrlenu(EntBilRects); ++i)
    {
        stbrp_rect PackedRect = EntBilRects[i];
        if (!PackedRect.was_packed)
        {
            LogError("Entity billboard atlas packing error. Failed to pack a billboard.");
            continue;
        }

        BitmapHandle BillboardBitmap = BillboardBitmaps[PackedRect.id];

        BlitRect(BillboardAtlasBuffer.data, BillboardAtlasW, BillboardAtlasH,
            (u8*)BillboardBitmap.memory, PackedRect.w, PackedRect.h,
            PackedRect.x, PackedRect.y, sizeof(u8) * 4/*RGBA*/);

        vec4 *UVRect = &SupportRenderer.EntityBillboardRectMap[PackedRect.id];
        vec2 MinUV = vec2((float)(PackedRect.x) / (float)BillboardAtlasW, 
            (float)(PackedRect.y) / (float)BillboardAtlasH);
        vec2 MaxUV = vec2((float)(PackedRect.x + PackedRect.w) / (float)BillboardAtlasW, 
            (float)(PackedRect.y + PackedRect.h) / (float)BillboardAtlasH);
        UVRect->x = MinUV.x; // bottom left x
        UVRect->y = MinUV.y; // bottom left y
        UVRect->z = MaxUV.x; // top right x
        UVRect->w = MaxUV.y; // top right y

        float *BillboardWidth = &SupportRenderer.EntityBillboardWidthMap[PackedRect.id];
        *BillboardWidth = (float)PackedRect.w;
    }

    CreateGPUTextureFromBitmap(&SupportRenderer.EntityBillboardAtlas,
        BillboardAtlasBuffer.data, BillboardAtlasW, BillboardAtlasH,
        GL_RGBA, GL_RGBA);

    // Release all temporary buffers
    for (int i = 0; i < 64; ++i)
        if (BillboardBitmaps[i].memory)
            FreeImage(BillboardBitmaps[i]);
    arrfree(EntBilRects);
    BillboardAtlasBuffer.free();
}

void asset_db_t::LoadAllResources()
{
    LoadNewTexture(wd_path("default.png").c_str());
    // LoadNewTexture(texture_path("t_bpav2.bmp").c_str());
    // LoadNewTexture(texture_path("t_gf56464.bmp").c_str());
    // LoadNewTexture(texture_path("t_hzdg.bmp").c_str());
    // LoadNewTexture(texture_path("t_kgr2_p.bmp").c_str());
    // LoadNewTexture(texture_path("t_mbrk2_1.bmp").c_str());
    // LoadNewTexture(texture_path("t_vstnfcv.bmp").c_str());
    // LoadNewTexture(texture_path("example_5.jpg").c_str());
    // LoadNewTexture(texture_path("example_7.jpg").c_str());
    // LoadNewTexture(texture_path("example_9.jpg").c_str());
    // LoadNewTexture(texture_path("example_10.jpg").c_str());
    // LoadNewTexture(texture_path("example_14.jpg").c_str());
    // LoadNewTexture(texture_path("example_16.jpg").c_str());
    // LoadNewTexture(texture_path("example_17.jpg").c_str());
    // LoadNewTexture(texture_path("sld_gegfblock02b_64.jpg").c_str());
    // LoadNewTexture(texture_path("example_19.jpg").c_str());
    // LoadNewTexture(texture_path("example_20.jpg").c_str());
    LoadNewTexture(texture_path("tex_concretepanel.png").c_str());
    LoadNewTexture(texture_path("tex_metalgrate.png").c_str());
    LoadNewTexture(texture_path("tex_cables_0.png").c_str());
    LoadNewTexture(texture_path("tex_cables_1.png").c_str());
    LoadNewTexture(texture_path("tex_cables_glow.png").c_str());
    LoadNewTexture(texture_path("tex_cables_pipe.png").c_str());

    DefaultEditorTexture = GetTextureById(1);
    CreateGPUTextureFromDisk(&DefaultMissingTexture, wd_path("missing_texture.png").c_str());

    // === Skinned models ===
    Skeleton_Humanoid = new(StaticGameMemory.Alloc<skeleton_t>()) skeleton_t();
    ASSERT(LoadSkeleton_GLTF2Bin(model_path("attacker.glb").c_str(), Skeleton_Humanoid));
    Model_Attacker = new(StaticGameMemory.Alloc<skinned_model_t>()) skinned_model_t(Skeleton_Humanoid);
    ASSERT(LoadSkinnedModel_GLTF2Bin(model_path("attacker.glb").c_str(), Model_Attacker));

    // == Textured models
    ModelsTextured = fixed_array<ModelGLTF>(MT_COUNT, MemoryType::Game);
    ModelsTextured.setlen(MT_COUNT);
    for (u16 i = 0; i < MT_COUNT; ++i)
        ModelsTextured[i].MT_ID = i;
    ASSERT(LoadModelGLTF2Bin(&ModelsTextured[MT_WPN_TYPE1], model_path("wpn_type1.glb").c_str()));
    ASSERT(LoadModelGLTF2Bin(&ModelsTextured[MT_PRJ_NAIL], model_path("prj_nail.glb").c_str()));
    ASSERT(LoadModelGLTF2Bin(&ModelsTextured[MT_PRJ_ROCKET], model_path("gib_generic_0.glb").c_str()));
    ASSERT(LoadModelGLTF2Bin(&ModelsTextured[MT_ATTACKER_CORPSE], model_path("attacker_corpse.glb").c_str()));
    ASSERT(LoadModelGLTF2Bin(&ModelsTextured[MT_GENERIC_GIB_0], model_path("gib_generic_0.glb").c_str()));
    ASSERT(LoadModelGLTF2Bin(&ModelsTextured[MT_GENERIC_GIB_1], model_path("gib_generic_1.glb").c_str()));

    // === SFX ===
    // SDL mixer does not support pitch adjustment so I must resample the sound manually
    // and have multiple pitch-shifted versions of the sound
    Sfx_Shoot0 = Mixer_LoadChunk(sfx_path("snd_quakesupernailgun.ogg").c_str());
    Sfx_Ricochet[0] = Mixer_LoadChunk(sfx_path("snd_ricochet_0.ogg").c_str());
    Sfx_Ricochet[1] = Mixer_LoadChunk(sfx_path("snd_ricochet_1.ogg").c_str());
    Sfx_Ricochet[2] = Mixer_LoadChunk(sfx_path("snd_ricochet_2.ogg").c_str());

    // === Level entity billboards ===
    // Load all the billboard bitmaps
    BitmapHandle BillboardBitmaps[64];
    ReadImage(BillboardBitmaps[POINT_PLAYER_SPAWN], entity_icons_path("ent_bil_playerstart.png").c_str());
    ReadImage(BillboardBitmaps[POINT_LIGHT], entity_icons_path("ent_bil_pointlight.png").c_str());
    ReadImage(BillboardBitmaps[DIRECTIONAL_LIGHT_PROPERTIES], entity_icons_path("ent_bil_dlight.png").c_str());
    CreateEntityBillboardAtlasForSupportRenderer(BillboardBitmaps);
    

}

// TODO(Kevin): FreeAllResources
