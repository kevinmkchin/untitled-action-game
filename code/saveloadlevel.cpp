
bool BuildGameMap(const char *path)
{
    // TODO(Kevin): remove this timing shit
    u32 TimeAtStartOfBuildGameMap = SDL_GetTicks();


    game_map_build_data_t BuildData;
    BuildData.Output = ByteBufferNew();
    BuildData.TotalFaceCount = MapEdit::LevelEditorFaces.count;
    const int totalfacecount = BuildData.TotalFaceCount;
    std::unordered_map<u32, std::vector<float>>& VertexBuffers = BuildData.VertexBuffers;
    std::vector<vec3>& ColliderWorldPoints = BuildData.ColliderWorldPoints;
    std::vector<u32>& ColliderSpans = BuildData.ColliderSpans;


    // need to save a bunch of FaceBatches (which are basically just meshes)
    // sort by texture, etc. every face using same texture goes into same FaceBatch
    // collider data -> just save the point clouds for mesh colliders and add to Octree while loading into game
    // need to do something smarter for texture data (want to use database so that textures are just enums)

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

    BakeStaticLighting(BuildData);


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

    return writtenToFile;
}

bool LoadGameMap(const char *path)
{
    std::unordered_map<u32, void*> DeserElemIdToElem;

    // deserialize
    ByteBuffer mapbuf;
    if (ByteBufferReadFromFile(&mapbuf, path) == 0)
        return false;

    i32 lmw, lmh;
    ByteBufferRead(&mapbuf, i32, &lmw);
    ByteBufferRead(&mapbuf, i32, &lmh);
    float *lightMapData = NULL; 
    arrsetcap(lightMapData, lmw*lmh);
    ByteBufferReadBulk(&mapbuf, lightMapData, lmw*lmh*sizeof(float));
    // These light map textures use bilinear texture filtering to allow light values 
    // to be interpolated between the given texture coordinate's neighboring texels. 
    CreateGPUTextureFromBitmap(&Temporary_LightMapVisualizeTex, lightMapData, lmw, lmh,
                               GL_R32F, GL_RED, GL_LINEAR, GL_LINEAR, GL_FLOAT);
    CreateGPUTextureFromBitmap(&Temporary_TestTex0, lightMapData, lmw, lmh,
                               GL_R32F, GL_RED, GL_NEAREST, GL_NEAREST, GL_FLOAT);
    arrfree(lightMapData);

    // colliders
    size_t numColliderPoints, numColliderSpans;
    ByteBufferRead(&mapbuf, size_t, &numColliderPoints);
    ByteBufferRead(&mapbuf, size_t, &numColliderSpans);

    ASSERT(GameLevelColliders.size() == 0); // just crash for now
    GameLevelColliderPoints.resize(numColliderPoints);
    GameLevelColliders.resize(numColliderSpans);

    std::vector<u32> ColliderSpans(numColliderSpans); 
    ByteBufferReadBulk(&mapbuf, GameLevelColliderPoints.data(), sizeof(vec3)*numColliderPoints);
    ByteBufferReadBulk(&mapbuf, ColliderSpans.data(), sizeof(u32)*numColliderSpans);
    
    int GameLevelColliderPointsIterator = 0;
    for (u32 colliderIndex = 0; colliderIndex < numColliderSpans; ++colliderIndex)
    {
        u32 span = ColliderSpans[colliderIndex];
        FlatPolygonCollider& collider = GameLevelColliders[colliderIndex];
        collider.pointCloudPtr = &GameLevelColliderPoints[GameLevelColliderPointsIterator];
        collider.pointCount = span;
        GameLevelColliderPointsIterator += span;
    }
    ASSERT(GameLevelCollisionTree.root.get() == NULL); // i don't care. crash for now.
    Bounds GameLevelWorldBound = Bounds(vec3(-0.17f, -0.17f, -0.17f), vec3(8000, 8000, 8000));
    GameLevelCollisionTree = Octree(GameLevelWorldBound, 7, 24); // NOTE(Kevin): maybe I can be smarter about how I'm deciding the values for max tree depth and max count per octant
    for (FlatPolygonCollider& collider : GameLevelColliders)
    {
        GameLevelCollisionTree.Insert(&collider);
    }

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
        FaceBatch.LightMapTexture = Temporary_LightMapVisualizeTex;
        CreateFaceBatch(&FaceBatch);
        RebindFaceBatch(&FaceBatch, u32(sizeof(float)*VertexCount), vb.data());
        GameLevelFaceBatches.push_back(FaceBatch);
    }

    ByteBufferFree(&mapbuf);

    return true;
}
