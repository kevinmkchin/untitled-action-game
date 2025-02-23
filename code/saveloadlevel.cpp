
std::vector<vec3> LoadingLevelColliderPoints;
std::vector<u32> LoadingLevelColliderSpans;

static void BuildOutLevelEntities(game_map_build_data_t *BuildData)
{
    for (size_t i = 0; i < LevelEditor.LevelEntities.lenu(); ++i)
    {
        const level_entity_t& Ent = LevelEditor.LevelEntities[i];
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
                BuildData->DirectionToSun = Ent.Rotation;
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

bool BuildGameMap(const char *path)
{
    // TODO(Kevin): remove this timing shit
    u32 TimeAtStartOfBuildGameMap = SDL_GetTicks();

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
        MapEdit::Face *face = MapEdit::LevelEditorFaces.At(i);

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
    BuildOutLevelEntities(&BuildData);

    ByteBufferWrite(&BuildData.Output, vec3, BuildData.PlayerStartPosition);
    ByteBufferWrite(&BuildData.Output, vec3, BuildData.PlayerStartRotation);

    Lightmapper.BakeStaticLighting(BuildData);

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

    // TODO(Kevin): remove this timing shit
    u32 TimeAtEndOfBuildGameMap = SDL_GetTicks();
    float TimeElapsedToBuildGameMapInSeconds = (TimeAtEndOfBuildGameMap - TimeAtStartOfBuildGameMap)/1000.f;
    LogMessage("Took %fs to build.", TimeElapsedToBuildGameMapInSeconds);

    DeallocateGameMapBuildData(&BuildData);

    return writtenToFile;
}

map_load_result_t LoadGameMap(const char *path)
{
    map_load_result_t Result;

    std::unordered_map<u32, void*> DeserElemIdToElem;

    // deserialize
    ByteBuffer mapbuf;
    if (ByteBufferReadFromFile(&mapbuf, path) == 0)
    {
        Result.Success = false;
        return Result;
    }

    ByteBufferRead(&mapbuf, vec3, &Result.PlayerStartPosition);
    ByteBufferRead(&mapbuf, vec3, &Result.PlayerStartRotation);

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


    // colliders
    size_t numColliderPoints, numColliderSpans;
    ByteBufferRead(&mapbuf, size_t, &numColliderPoints);
    ByteBufferRead(&mapbuf, size_t, &numColliderSpans);

    LoadingLevelColliderPoints.clear();
    LoadingLevelColliderPoints.resize(numColliderPoints); 
    LoadingLevelColliderSpans.clear();
    LoadingLevelColliderSpans.resize(numColliderSpans);

    ByteBufferReadBulk(&mapbuf, LoadingLevelColliderPoints.data(), sizeof(vec3)*numColliderPoints);
    ByteBufferReadBulk(&mapbuf, LoadingLevelColliderSpans.data(), sizeof(u32)*numColliderSpans);


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
        GameLevelFaceBatches.push_back(FaceBatch);
    }

    ByteBufferFree(&mapbuf);

    Result.Success = true;

    return Result;
}
