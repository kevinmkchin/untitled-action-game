#include "saveloadlevel.h"
#include "levelentities.h"
#include "winged.h"
#include "lightmap.h"
#include "game.h"


static void BuildOutLevelEntities(
    level_editor_t *EditorState,
    game_map_build_data_t *BuildData)
{
    for (size_t i = 0; i < EditorState->LevelEntities.lenu(); ++i)
    {
        const level_entity_t& Ent = EditorState->LevelEntities[i];
        switch (Ent.Type)
        {
            case POINT_LIGHT: {
                static_point_light_t PointLight;
                PointLight.Pos = Ent.Position;
                BuildData->PointLights.put(PointLight);
            } break;
            case POINT_PLAYER_SPAWN: {
                BuildData->PlayerStartPosition = Ent.Position;
                BuildData->PlayerStartRotation = Ent.Rotation;
            } break;
            case DIRECTIONAL_LIGHT_PROPERTIES: {
                // If DirectionToSun is 0,0,0 then no sun in the level
                BuildData->DirectionToSun = Normalize(Ent.Rotation);
            } break;
        }
    }
}

static void DeallocateGameMapBuildData(game_map_build_data_t *BuildData)
{
    BuildData->VertexBuffers.clear();
    BuildData->ColliderWorldPoints.clear();
    BuildData->ColliderSpans.clear();
    BuildData->PointLights.free();
}


constexpr u64 lightdata_serialize_start_marker = 0x6C69676874646174;
static void SerializeMapLightData(game_map_build_data_t *BuildData)
{
    ByteBufferWrite(&BuildData->Output, u64, lightdata_serialize_start_marker);

    ByteBufferWrite(&BuildData->Output, vec3, BuildData->DirectionToSun);
    ByteBufferWrite(&BuildData->Output, u32, (u32)BuildData->PointLights.lenu());
    ByteBufferWriteBulk(&BuildData->Output, BuildData->PointLights.data, sizeof(static_point_light_t) * BuildData->PointLights.lenu());
}

static void DeserializeMapLightData(ByteBuffer *Buf, game_state *MapInfo)
{
    u64 SerializeStartMarker;
    ByteBufferRead(Buf, u64, &SerializeStartMarker);
    ASSERT(SerializeStartMarker == lightdata_serialize_start_marker);

    ByteBufferRead(Buf, vec3, &MapInfo->DirectionToSun);
    MapInfo->DirectionToSun = Normalize(MapInfo->DirectionToSun);
    u32 PointLightsCount;
    ByteBufferRead(Buf, u32, &PointLightsCount);
    MapInfo->PointLights = fixed_array<static_point_light_t>(PointLightsCount, MemoryType::Level);
    MapInfo->PointLights.setlen(PointLightsCount);
    ByteBufferReadBulk(Buf, MapInfo->PointLights.data, sizeof(static_point_light_t) * PointLightsCount);
}

bool BuildGameMap(level_editor_t *EditorState, const char *path)
{
    u64 TimeAtStartOfBuildGameMap = SDL_GetTicks();

    LogMessage("Building game map data to %s", path);
    LogMessage("Light baking will take some time...");

    game_map_build_data_t BuildData;
    BuildData.Output = ByteBufferNew();
    BuildData.TotalFaceCount = MapEdit::LevelEditorFaces.count;
    const int totalfacecount = BuildData.TotalFaceCount;
    std::unordered_map<u32, std::vector<float>>& VertexBuffers = BuildData.VertexBuffers;
    std::vector<vec3>& ColliderWorldPoints = BuildData.ColliderWorldPoints;
    std::vector<u32>& ColliderSpans = BuildData.ColliderSpans;

    // need to save a bunch of FaceBatches (which are basically just meshes)
    // organize by texture, etc. every face using same texture goes into same FaceBatch
    // collider data -> just save the point clouds for mesh colliders and add to Octree while loading into game
    // need to do something smarter for texture data (want to use database so that textures are just enums)

    // Polygon colliders data
    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::LevelEditorFaces[i];

        std::vector<MapEdit::Vert*> faceVerts = face->GetVertices();

        u32 ColliderSpan = 0;
        for (MapEdit::Vert *v : faceVerts)
        {
            vec3 worldpos = v->pos;
            ColliderWorldPoints.push_back(worldpos);
            ++ColliderSpan;
        }
        ColliderSpans.push_back(ColliderSpan);
    }

    // Process entities into BuildData
    BuildOutLevelEntities(EditorState, &BuildData);

    ByteBufferWrite(&BuildData.Output, vec3, BuildData.PlayerStartPosition);
    ByteBufferWrite(&BuildData.Output, vec3, BuildData.PlayerStartRotation);

    // Lightmap
    lightmapper_t *Lightmapper = new lightmapper_t();
    Lightmapper->BakeStaticLighting(BuildData);
    float *LightmapAtlas;
    i32 LightmapAtlasW;
    i32 LightmapAtlasH;
    Lightmapper->GetLightmap(&LightmapAtlas, &LightmapAtlasW, &LightmapAtlasH);
    ByteBufferWrite(&BuildData.Output, i32, LightmapAtlasW);
    ByteBufferWrite(&BuildData.Output, i32, LightmapAtlasH);
    ByteBufferWriteBulk(&BuildData.Output, LightmapAtlas, LightmapAtlasW*LightmapAtlasH*sizeof(float));
    Lightmapper->FreeLightmap();
    delete Lightmapper;

    // Light cache volume
    lc_volume_baker_t *LightCacheVolumeBaker = new lc_volume_baker_t();
    LightCacheVolumeBaker->BakeLightCubes(BuildData);
    LightCacheVolumeBaker->LightCubeVolume.Serialize(&BuildData.Output);
    delete LightCacheVolumeBaker;

    SerializeMapLightData(&BuildData);

    // colliders
    size_t numColliderPoints = ColliderWorldPoints.size();
    size_t numColliderSpans = ColliderSpans.size();
    ByteBufferWrite(&BuildData.Output, size_t, numColliderPoints);
    ByteBufferWrite(&BuildData.Output, size_t, numColliderSpans);
    ByteBufferWriteBulk(&BuildData.Output, ColliderWorldPoints.data(), sizeof(vec3)*numColliderPoints);
    ByteBufferWriteBulk(&BuildData.Output, ColliderSpans.data(), sizeof(u32)*numColliderSpans);

    // vertex buffers
    size_t numVertexBufs = VertexBuffers.size();
    ByteBufferWrite(&BuildData.Output, size_t, numVertexBufs);
    for (auto& vbpair : VertexBuffers)
    {
        u32 texturePersistId = vbpair.first;
        const std::vector<float>& vb = vbpair.second;

        ByteBufferWrite(&BuildData.Output, u32, texturePersistId);
        ByteBufferWrite(&BuildData.Output, size_t, vb.size());
        ByteBufferWriteBulk(&BuildData.Output, (void*)vb.data(), sizeof(float)*vb.size());
    }

    bool writtenToFile = ByteBufferWriteToFile(&BuildData.Output, path) == 1;
    ByteBufferFree(&BuildData.Output);

    u64 TimeAtEndOfBuildGameMap = SDL_GetTicks();
    float TimeElapsedToBuildGameMapInSeconds = (TimeAtEndOfBuildGameMap - TimeAtStartOfBuildGameMap)/1000.f;
    LogMessage("Took %fs to build.", TimeElapsedToBuildGameMapInSeconds);

    DeallocateGameMapBuildData(&BuildData);

    return writtenToFile;
}

bool LoadGameMap(game_state *MapInfo, const char *path)
{
    std::unordered_map<u32, void*> DeserElemIdToElem;

    ByteBuffer mapbuf;
    if (ByteBufferReadFromFile(&mapbuf, path) == 0)
        return false;

    ByteBufferRead(&mapbuf, vec3, &MapInfo->PlayerStartPosition);
    ByteBufferRead(&mapbuf, vec3, &MapInfo->PlayerStartRotation);

    // Lightmap
    i32 lmw, lmh;
    ByteBufferRead(&mapbuf, i32, &lmw);
    ByteBufferRead(&mapbuf, i32, &lmh);
    float *lightMapData = NULL; 
    arrsetcap(lightMapData, lmw*lmh);
    ByteBufferReadBulk(&mapbuf, lightMapData, lmw*lmh*sizeof(float));
    // These light map textures use bilinear texture filtering to allow light values 
    // to be interpolated between the given texture coordinate's neighboring texels.
    GPUTexture LevelLightmapTexture;
    CreateGPUTextureFromBitmap(&LevelLightmapTexture, lightMapData, lmw, lmh,
                               GL_R32F, GL_RED, GL_LINEAR, GL_LINEAR, GL_FLOAT);
    // CreateGPUTextureFromBitmap(&Temporary_TestTex0, lightMapData, lmw, lmh,
    //                            GL_R32F, GL_RED, GL_NEAREST, GL_NEAREST, GL_FLOAT);
    arrfree(lightMapData);

    // Light cache volume
    MapInfo->LightCacheVolume = new_InLevelMemory(lc_volume_t)();
    MapInfo->LightCacheVolume->Deserialize(&mapbuf, MemoryType::Level);

    DeserializeMapLightData(&mapbuf, MapInfo);

    // colliders
    size_t numColliderPoints, numColliderSpans;
    ByteBufferRead(&mapbuf, size_t, &numColliderPoints);
    ByteBufferRead(&mapbuf, size_t, &numColliderSpans);

    MapInfo->LoadingLevelColliderPoints = fixed_array<vec3>((u32)numColliderPoints, MemoryType::Level);
    MapInfo->LoadingLevelColliderPoints.setlen((u32)numColliderPoints);
    MapInfo->LoadingLevelColliderSpans = fixed_array<u32>((u32)numColliderSpans, MemoryType::Level);
    MapInfo->LoadingLevelColliderSpans.setlen((u32)numColliderSpans);
    ByteBufferReadBulk(&mapbuf, MapInfo->LoadingLevelColliderPoints.data, sizeof(vec3)*numColliderPoints);
    ByteBufferReadBulk(&mapbuf, MapInfo->LoadingLevelColliderSpans.data, sizeof(u32)*numColliderSpans);

    // vertex buffers
    size_t numVertexBufs;
    ByteBufferRead(&mapbuf, size_t, &numVertexBufs);
    for (int i = 0; i < numVertexBufs; ++i)
    {
        u32 texturePersistId;
        ByteBufferRead(&mapbuf, u32, &texturePersistId);

        size_t VertexCount;
        ByteBufferRead(&mapbuf, size_t, &VertexCount);

        std::vector<float> vb;
        vb.resize(VertexCount);
        ByteBufferReadBulk(&mapbuf, vb.data(), sizeof(float)*VertexCount);
        ASSERT(vb.size() == VertexCount);
        
        face_batch_t FaceBatch;
        FaceBatch.ColorTexture = Assets.GetTextureById(texturePersistId).gputex;
        FaceBatch.LightMapTexture = LevelLightmapTexture;
        CreateFaceBatch(&FaceBatch);
        RebindFaceBatch(&FaceBatch, sizeof(float)*VertexCount, vb.data());
        MapInfo->GameLevelFaceBatches.push_back(FaceBatch);
    }

    ByteBufferFree(&mapbuf);
    return true;
}
