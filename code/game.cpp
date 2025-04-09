

// extern
game_state g_GameState;


enum ske_humanoid_clips : u32
{
    SKE_HUMANOID_DEATH = 0,
    SKE_HUMANOID_RUN = 1,
    SKE_HUMANOID_CLIPCOUNT = 2
};


void InitializeGame()
{
    InstanceDrawing_AcquireGPUResources();

    g_GameState.AnimatorPool = fixed_array<animator_t>(64, MemoryType::Game);
    g_GameState.AnimatorPool.setlen(64);
    for (size_t i = 0; i < g_GameState.AnimatorPool.length; ++i)
        g_GameState.AnimatorPool[i] = animator_t();

    Physics.Initialize();
    Physics.VirtualCharacterContactListener.GameState = &g_GameState;

    SetupProjectilesDataAndAllocateMemory();
    g_GameState.BloodParticles.Alloc(128, 32, MemoryType::Game);
    g_GameState.BloodParticlesVB.Alloc(128*6);
    g_GameState.PQuadBuf = fixed_array<particle_vertex>(256*6, MemoryType::Game);

    EnemySystem.Init();

    g_GameState.Player.Init();

#if INTERNAL_BUILD
    RecastDebugDrawer.Init();
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    JoltDebugDrawer = new_InGameMemory(jph_debug_draw_gl3_t)();
#endif // JPH_DEBUG_RENDERER
}

void DestroyGame()
{
    EnemySystem.Destroy();
    Physics.Destroy();

    g_GameState.BloodParticlesVB.Free();
    InstancedDrawing_ReleaseGPUResources();

#if INTERNAL_BUILD
    RecastDebugDrawer.Destroy();
#endif // INTERNAL_BUILD
}

void LoadLevel(const char *MapPath)
{
    StaticLevelMemory.ArenaOffset = 0;
    g_GameState.StaticInstances = fixed_array<model_instance_data_t>(MaxStaticInstances, MemoryType::Level);

    SDL_SetWindowRelativeMouseMode(SDLMainWindow, true);

    if (g_GameState.LevelLoaded)
        UnloadPreviousLevel();

    if (LoadGameMap(&g_GameState, MapPath) == false)
    {
        LogError("Failed to load game map: %s", MapPath);
        return;
    }

    ASSERT(Physics.PhysicsSystem);
    CreateAndRegisterLevelCollider();
    ASSERT(CreateRecastNavMesh());

    g_GameState.Player.CharacterController->SetPosition(ToJoltVector(g_GameState.PlayerStartPosition));
    // TODO Apply rotation to camera rotation instead
    // g_GameState.Player.mCharacter->SetRotation(ToJoltQuat(EulerToQuat(g_GameState.PlayerStartRotation)));

    g_GameState.LevelLoaded = true;
}

void UnloadPreviousLevel()
{
    if (!g_GameState.LevelColliderBodyId.IsInvalid())
    {
        Physics.BodyInterface->RemoveBody(g_GameState.LevelColliderBodyId);
        Physics.BodyInterface->DestroyBody(g_GameState.LevelColliderBodyId);
    }

    EnemySystem.RemoveAll();
    DestroyRecastNavMesh();

    g_GameState.LevelLoaded = false;
    StaticLevelMemory.ArenaOffset = 0;
}

void NonPhysicsTick()
{
    if (!(EnemySystem.Enemies[0].Flags & EnemyFlag_Active))
    {
        EnemySystem.SpawnEnemy(&g_GameState);
        while (EnemySystem.Enemies[0].Position.y > 10.f)
        {
            GetRandomPointOnNavMesh((float*)&EnemySystem.Enemies[0].Position);
        }
    }

    NonPhysicsTickAllEnemies(&g_GameState);

    NonPhysicsUpdateProjectiles(&g_GameState);

    g_GameState.Player.HandleInput();

    if (g_GameState.Player.Health <= 0.f)
    {
        // ASSERT(0);
    }
    GUI::PrimitiveTextFmt(20, 180, 9, GUI::Align::LEFT, "kills %d", g_GameState.KillEnemyCounter);
    GUI::PrimitiveTextFmt(180, 650, 18, GUI::Align::RIGHT, "%d", (int)ceilf(g_GameState.Player.Health));

    if (DebugShowNumberOfPhysicsBodies)
    {
        GUI::PrimitiveTextFmt(8, 48, GUI::GetFontSize(), GUI::Align::LEFT, 
            "JPH NumActiveBodies: %d", Physics.PhysicsSystem->GetNumActiveBodies(JPH::EBodyType::RigidBody));
    }
}

void PrePhysicsTick()
{
    particle_emitter FountainTest;
    FountainTest.WorldP = vec3(0.f,0.f,0.f);
    FountainTest.PSpread = vec3(0.f,0.f,0.f);
    FountainTest.dP = vec3(0.f,220.f,0.f);
    FountainTest.dPSpread = vec3(32.f,0.f,32.f);
    FountainTest.ddP = vec3(0.f,FromJoltUnit(-9.8f),0.f);
    FountainTest.Color = vec4(1,1,1,1.4f);
    FountainTest.ColorSpread = vec4(0,0,0,0.1f);
    FountainTest.dColor = vec4(0,0,0,-1.35f);
    FountainTest.Timer = 0.f;
    FountainTest.ParticleLifeTimer = 2.f;
    g_GameState.BloodParticles.Emitters.put(FountainTest);

    PrePhysicsTickAllEnemies(&g_GameState);

    PrePhysicsUpdateProjectiles();

    g_GameState.Player.PrePhysicsUpdate(&g_GameState);
}

void PostPhysicsTick()
{
    PostPhysicsTickAllEnemies(&g_GameState);

    PostPhysicsUpdateProjectiles(&g_GameState);

    g_GameState.Player.PostPhysicsUpdate();
}

void LateNonPhysicsTick()
{
    g_GameState.Player.LateNonPhysicsTick();

    random_series &EmitterRNG = g_GameState.ParticlesRNG;
    UpdateParticles(g_GameState.BloodParticles, EmitterRNG);
}

void DebugDrawGame()
{
    //vec3 p = g_GameState.Player.Root;// +g_GameState.Player.CamOffsetFromRoot;
    // size_t pi = LightCacheVolume->IndexByPosition(p);
    // for (size_t i = 0; i < g_GameState.LightCacheVolume->CubePositions.lenu(); ++i)
    // {
    //     const vec3 &CubePos = g_GameState.LightCacheVolume->CubePositions[i];
    //     lc_ambient_t &AmbientCube = g_GameState.LightCacheVolume->AmbientCubes[i];
    //     SupportRenderer.DrawColoredCube(CubePos, 1.f, 
    //         vec4(AmbientCube.PosX,AmbientCube.PosX,AmbientCube.PosX,1),
    //         vec4(AmbientCube.NegX,AmbientCube.NegX,AmbientCube.NegX,1),
    //         vec4(AmbientCube.PosY,AmbientCube.PosY,AmbientCube.PosY,1),
    //         vec4(AmbientCube.NegY,AmbientCube.NegY,AmbientCube.NegY,1),
    //         vec4(AmbientCube.PosZ,AmbientCube.PosZ,AmbientCube.PosZ,1),
    //         vec4(AmbientCube.NegZ,AmbientCube.NegZ,AmbientCube.NegZ,1)
    //     );
    // }

#ifdef JPH_DEBUG_RENDERER
    JoltDebugDrawer->Ready();

    if (DebugDrawLevelColliderFlag)
    {        
        Physics.BodyInterface->GetShape(g_GameState.LevelColliderBodyId)->Draw(JoltDebugDrawer,
            Physics.BodyInterface->GetCenterOfMassTransform(g_GameState.LevelColliderBodyId),
            JPH::Vec3::sReplicate(1.0f), JPH::Color(0,255,0,60), true, false);
    }

    if (DebugDrawEnemyCollidersFlag)
        DebugDrawEnemyColliders();

    if (DebugDrawProjectileCollidersFlag)
    {
        for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
        {
            projectile_t P = LiveProjectiles[i];
            Physics.BodyInterface->GetShape(P.BodyId)->Draw(JoltDebugDrawer,
                Physics.BodyInterface->GetCenterOfMassTransform(P.BodyId),
                JPH::Vec3::sReplicate(1.0f), JPH::Color(255,0,0,255), false, true);
        }
    }
#endif // JPH_DEBUG_RENDERER
}

void DoGameLoop()
{
    g_GameState.DynamicInstances = fixed_array<model_instance_data_t>(MaxDynamicInstances, MemoryType::Frame);

    if (g_GameState.GameLoopCanRun) 
    {
        NonPhysicsTick();

        static float Accumulator = 0.f;
        Accumulator += DeltaTime;
        while (Accumulator >= FixedDeltaTime)
        {
            PrePhysicsTick();
            Physics.Tick();
            PostPhysicsTick();

            Accumulator -= FixedDeltaTime;
        }

        LateNonPhysicsTick();

        for (size_t i = 0; i < g_GameState.AnimatorPool.length; ++i)
        {
            if (g_GameState.AnimatorPool[i].HasOwner)
            {
                g_GameState.AnimatorPool[i].UpdateGlobalPoses(DeltaTime);
                g_GameState.AnimatorPool[i].CalculateSkinningMatrixPalette();
            }
        }
    }

    DebugDrawGame();
    RenderGameLayer();

    if (g_GameState.GameLoopCanRun) 
    {
        UpdateGameGUI();
    }
}

void UpdateGameGUI()
{
    // temp crosshair
    ivec2 guiwh = ivec2(RenderTargetGUI.width, RenderTargetGUI.height);
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 3, guiwh.y / 2 - 3, 6, 6), vec4(0, 0, 0, 1));
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 2, guiwh.y / 2 - 2, 4, 4), vec4(1, 1, 1, 1));

    if (DebugShowGameMemoryUsage)
    {
        GUI::PrimitiveTextFmt(8, 58, GUI::GetFontSize(), GUI::Align::LEFT, 
            "Game Memory usage  [%zd/%zd KB]", 
            StaticGameMemory.ArenaOffset/1000, StaticGameMemory.ArenaSize/1000);
        GUI::PrimitiveTextFmt(8, 68, GUI::GetFontSize(), GUI::Align::LEFT, 
            "Level Memory usage [%zd/%zd KB]", 
            StaticLevelMemory.ArenaOffset/1000, StaticLevelMemory.ArenaSize/1000);
        GUI::PrimitiveTextFmt(8, 78, GUI::GetFontSize(), GUI::Align::LEFT, 
            "Frame Memory usage [%zd/%zd KB]", 
            FrameMemory.ArenaOffset/1000, FrameMemory.ArenaSize/1000);
    }
}

void RenderGameLayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGame.fbo);
    glViewport(0, 0, RenderTargetGame.width, RenderTargetGame.height);
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //RGBHEXTO1(0x6495ed), 1.f);//(RGB255TO1(211, 203, 190), 1.f);//(0.674f, 0.847f, 1.0f, 1.f); //RGB255TO1(46, 88, 120)
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    float aspectratio = float(BackbufferWidth) / float(BackbufferHeight);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    mat4 perspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 viewMatrix = g_GameState.Player.PlayerCam.ViewFromWorldMatrix();
    mat4 ClipFromWorld = perspectiveMatrix * viewMatrix;

    UseShader(Sha_GameLevel);
    glEnable(GL_CULL_FACE);

    GLBind4f(Sha_GameLevel, "MuzzleFlash", 
        g_GameState.Player.Weapon.MuzzleFlash.x, 
        g_GameState.Player.Weapon.MuzzleFlash.y, 
        g_GameState.Player.Weapon.MuzzleFlash.z, 
        g_GameState.Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_GameLevel, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(Sha_GameLevel, "viewMatrix", 1, viewMatrix.ptr());

    for (size_t i = 0; i < g_GameState.GameLevelFaceBatches.size(); ++i)
    {
        face_batch_t fb = g_GameState.GameLevelFaceBatches.at(i);
        RenderFaceBatch(&Sha_GameLevel, &fb);
    }

    RenderEnemies(&g_GameState, perspectiveMatrix, viewMatrix);

    RenderWeapon(&g_GameState.Player.Weapon, perspectiveMatrix.ptr(), viewMatrix.GetInverse().ptr());

    RenderProjectiles(&g_GameState, perspectiveMatrix, viewMatrix);

    SortAndDrawInstancedModels(&g_GameState, g_GameState.StaticInstances, g_GameState.DynamicInstances, 
        perspectiveMatrix, viewMatrix);

    UseShader(Sha_ParticlesDefault);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE); // Particles should depth test but not write to depth buffer
    GLBindMatrix4fv(Sha_ParticlesDefault, "ClipFromWorld", 1, ClipFromWorld.ptr());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Assets.DefaultMissingTexture.id);
    g_GameState.PQuadBuf.setlen(0);
    vec3 QuadDirection = -g_GameState.Player.PlayerCam.Direction;
    AssembleParticleQuads(g_GameState.BloodParticles, QuadDirection, g_GameState.PQuadBuf);
    g_GameState.BloodParticlesVB.Draw(g_GameState.PQuadBuf.data, g_GameState.PQuadBuf.lenu());
    GLHasErrors();
    glDepthMask(GL_TRUE);

    // PRIMITIVES
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, 
        vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

#if INTERNAL_BUILD
    DebugDrawGame();
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, 
        vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));
#endif // INTERNAL_BUILD
}

void CreateAndRegisterLevelCollider()
{
    static JPH::TriangleList LevelColliderTriangles;
    LevelColliderTriangles.clear();
    int LoadingLevelColliderPointsIterator = 0;
    for (u32 ColliderIndex = 0; ColliderIndex < (u32)LoadingLevelColliderSpans.size(); ++ColliderIndex)
    {
        u32 Span = LoadingLevelColliderSpans[ColliderIndex];
        vec3 *PointCloudPtr = &LoadingLevelColliderPoints[LoadingLevelColliderPointsIterator];

        vec3 First = PointCloudPtr[0];
        for (u32 i = 2; i < Span; ++i)
        {
            vec3 Second = PointCloudPtr[i-1];
            vec3 Third = PointCloudPtr[i];

            JPH::Float3 jph_first(ToJoltUnit(First.x), ToJoltUnit(First.y), ToJoltUnit(First.z));
            JPH::Float3 jph_second(ToJoltUnit(Second.x), ToJoltUnit(Second.y), ToJoltUnit(Second.z));
            JPH::Float3 jph_third(ToJoltUnit(Third.x), ToJoltUnit(Third.y), ToJoltUnit(Third.z));

            LevelColliderTriangles.push_back(JPH::Triangle(jph_first, jph_second, jph_third));
        }

        LoadingLevelColliderPointsIterator += Span;
    }
    JPH::MeshShapeSettings LevelColliderSettings = JPH::MeshShapeSettings(LevelColliderTriangles);
    LevelColliderSettings.mBuildQuality = JPH::MeshShapeSettings::EBuildQuality::FavorRuntimePerformance;
    LevelColliderSettings.SetEmbedded();

    // Create the shape
    JPH::ShapeSettings::ShapeResult LevelShapeResult = LevelColliderSettings.Create();
    JPH::ShapeRefC LevelShape = LevelShapeResult.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()
    if(LevelShapeResult.HasError())
    {
        LogMessage("%s", LevelShapeResult.GetError().c_str());
    }

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    JPH::BodyCreationSettings LevelBodySettings(LevelShape, JPH::RVec3(0.0, 0.0, 0.0), 
        JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    LevelBodySettings.mEnhancedInternalEdgeRemoval = true;

    // Create the actual rigid body
    JPH::Body *LevelCollider = Physics.BodyInterface->CreateBody(LevelBodySettings); // Note that if we run out of bodies this can return nullptr

    g_GameState.LevelColliderBodyId = LevelCollider->GetID();

    // Add it to the world
    // NOTE(Kevin 2025-01-30): Why is this DontActivate?
    Physics.BodyInterface->AddBody(LevelCollider->GetID(), JPH::EActivation::DontActivate);

    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    Physics.PhysicsSystem->OptimizeBroadPhase();
}
