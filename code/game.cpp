

skeleton_t *Skeleton_Humanoid;
skinned_model_t *Model_Attacker;
animator_t Animator;

// extern
std::vector<face_batch_t> GameLevelFaceBatches;
bool GameLoopCanRun = true;

// internal
bool LevelLoaded = false;
physics_t Physics;
player_t Player;
JPH::BodyID LevelColliderBodyId;


void InitializeGame()
{
    Skeleton_Humanoid = new skeleton_t();
    LoadSkeleton_GLTF2Bin(model_path("attacker.glb").c_str(), Skeleton_Humanoid);
    Model_Attacker = new skinned_model_t(Skeleton_Humanoid);
    LoadSkinnedModel_GLTF2Bin(model_path("attacker.glb").c_str(), Model_Attacker);

    ASSERT(TestAnimClip0);
    Animator.PlayAnimation(TestAnimClip0);

    Physics.Initialize();

    Player.Init();

    Enemies.free();

#if INTERNAL_BUILD
    RecastDebugDrawer.Init();
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    JoltDebugDrawer = new jph_debug_draw_gl3_t();
    JoltDebugDrawer->Init();
#endif // JPH_DEBUG_RENDERER
}

void DestroyGame()
{
    Physics.Destroy();
    Enemies.free();

#if INTERNAL_BUILD
    RecastDebugDrawer.Destroy();
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    JoltDebugDrawer->Destroy();
    delete JoltDebugDrawer;
#endif // JPH_DEBUG_RENDERER
}

void LoadLevel(const char *MapPath)
{
    ASSERT(Physics.PhysicsSystem);

    SDL_SetRelativeMouseMode(SDL_TRUE);

    if (LevelLoaded)
        UnloadPreviousLevel();

    map_load_result_t MapLoadResult = LoadGameMap(MapPath);
    if (MapLoadResult.Success == false)
    {
        LogError("failed to load game map");
        return;
    }

    CreateAndRegisterLevelCollider();
    ASSERT(CreateRecastNavMesh());

    Player.CharacterController->SetPosition(ToJoltVector(MapLoadResult.PlayerStartPosition));
    // TODO Apply rotation to camera rotation instead
    // Player.mCharacter->SetRotation(ToJoltQuat(EulerToQuat(MapLoadResult.PlayerStartRotation)));

    enemy_t Enemy0;
    // enemy_t Enemy1;
    // enemy_t Enemy2;

    // Enemy0.Position = vec3(0, 0, 30);
    // Enemy1.Position = vec3(0, 0, -30);
    GetRandomPointOnNavMesh((float*)&Enemy0.Position);
    // GetRandomPointOnNavMesh((float*)&Enemy1.Position);
    // GetRandomPointOnNavMesh((float*)&Enemy2.Position);

    Enemy0.Init();
    // Enemy1.Init();
    // Enemy2.Init();

    Enemies.put(Enemy0);
    // Enemies.put(Enemy1);
    // Enemies.put(Enemy2);

    LevelLoaded = true;
}

void UnloadPreviousLevel()
{
    if (!LevelColliderBodyId.IsInvalid())
    {
        Physics.BodyInterface->RemoveBody(LevelColliderBodyId);
        Physics.BodyInterface->DestroyBody(LevelColliderBodyId);
    }

    Enemies.free();
    DestroyRecastNavMesh();

    LevelLoaded = false;
}

void NonPhysicsTick()
{
    Player.HandleInput();
}

void PrePhysicsTick()
{
    PrePhysicsTickAllEnemies();

    Player.PrePhysicsUpdate();
}

void PostPhysicsTick()
{
    PostPhysicsTickAllEnemies();

    Player.PostPhysicsUpdate();
}

void LateNonPhysicsTick()
{
#ifdef JPH_DEBUG_RENDERER
    JoltDebugDrawer->Ready();

    if (DebugDrawLevelColliderFlag)
    {        
        Physics.BodyInterface->GetShape(LevelColliderBodyId)->Draw(JoltDebugDrawer,
            Physics.BodyInterface->GetCenterOfMassTransform(LevelColliderBodyId),
            JPH::Vec3::sReplicate(1.0f), JPH::Color(0,255,0,60), true, false);
    }

    if (DebugDrawEnemyCollidersFlag)
        DebugDrawEnemyColliders();
#endif // JPH_DEBUG_RENDERER

    Player.LateNonPhysicsTick();
}

void DoGameLoop()
{
    if (!GameLoopCanRun) 
        return;

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

    // Do animation loop
    Animator.UpdateGlobalPoses(DeltaTime);
    Animator.GetSkinningMatrixPalette();

    // Do render loop
    RenderGameLayer();

    UpdateGameGUI();
}

void UpdateGameGUI()
{
    // temp crosshair
    ivec2 guiwh = ivec2(RenderTargetGUI.width, RenderTargetGUI.height);
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 3, guiwh.y / 2 - 3, 6, 6), vec4(0, 0, 0, 1));
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 2, guiwh.y / 2 - 2, 4, 4), vec4(1, 1, 1, 1));
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
    mat4 viewMatrix = Player.PlayerCam.ViewFromWorldMatrix();

    UseShader(GameLevelShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(GameLevelShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(GameLevelShader, "viewMatrix", 1, viewMatrix.ptr());

    for (size_t i = 0; i < GameLevelFaceBatches.size(); ++i)
    {
        face_batch_t fb = GameLevelFaceBatches.at(i);
        RenderFaceBatch(&GameLevelShader, &fb);
    }

    // PRIMITIVES    
    static bool DoPrimitivesDepthTest = false;
    if (KeysPressed[SDL_SCANCODE_X])
        DoPrimitivesDepthTest = !DoPrimitivesDepthTest;
    if (DoPrimitivesDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, 
        vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

#if INTERNAL_BUILD
    if (DebugDrawNavMeshFlag || DebugDrawEnemyPathingFlag)
        RecastDebugDrawer.Ready(perspectiveMatrix.ptr(), viewMatrix.ptr());
    if (DebugDrawNavMeshFlag)
        DebugDrawRecast(DRAWMODE_NAVMESH);
    if (DebugDrawEnemyPathingFlag)
    {
        DebugDrawFollowPath();
        SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, 
            vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));
    }
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    mat4 ViewProjectionMatrix = perspectiveMatrix * viewMatrix;
    JoltDebugDrawer->Flush(ViewProjectionMatrix.ptr());
#endif // JPH_DEBUG_RENDERER

    UseShader(GameAnimatedCharacterShader);
    glEnable(GL_DEPTH_TEST);
    GLBindMatrix4fv(GameAnimatedCharacterShader, "Projection", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(GameAnimatedCharacterShader, "View", 1, viewMatrix.ptr());

    mat4 ModelMatrix = TranslationMatrix(Enemies[0].Position) * 
        RotationMatrix(Enemies[0].Orientation) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);

    GLBindMatrix4fv(GameAnimatedCharacterShader, "Model", 1, ModelMatrix.ptr());

    GLBindMatrix4fv(GameAnimatedCharacterShader, "FinalBonesMatrices[0]", MAX_BONES, 
        Animator.SkinningMatrixPalette[0].ptr());

    for (size_t i = 0; i < arrlenu(Model_Attacker->Meshes); ++i)
    {
        skinned_mesh_t m = Model_Attacker->Meshes[i];
        GPUTexture t = Model_Attacker->Textures[i];

        glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, t.id);
        glBindTexture(GL_TEXTURE_2D, Assets.DefaultEditorTexture.gputex.id);

        glBindVertexArray(m.VAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.IBO);
        glDrawElements(GL_TRIANGLES, m.IndicesCount, GL_UNSIGNED_INT, nullptr);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // RenderModelGLTF(Model_Knight);
    // ModelMatrix = TranslationMatrix(Enemies[1].Position) * RotationMatrix(Enemies[1].Orientation);
    // GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, ModelMatrix.ptr());
    // RenderModelGLTF(Model_Knight);
    // ModelMatrix = TranslationMatrix(Enemies[2].Position) * RotationMatrix(Enemies[2].Orientation);
    // GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, ModelMatrix.ptr());
    // RenderModelGLTF(Model_Knight);

    RenderWeapon(&Player.Weapon, perspectiveMatrix.ptr(), viewMatrix.GetInverse().ptr());
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
    LevelColliderSettings.SetEmbedded();

    // Create the shape
    JPH::ShapeSettings::ShapeResult LevelShapeResult = LevelColliderSettings.Create();
    JPH::ShapeRefC LevelShape = LevelShapeResult.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()
    if(LevelShapeResult.HasError())
    {
        LogMessage("%s", LevelShapeResult.GetError().c_str());
    }

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    JPH::BodyCreationSettings LevelBodySettings(LevelShape, JPH::RVec3(0.0, 0.0, 0.0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    LevelBodySettings.mEnhancedInternalEdgeRemoval = true;

    // Create the actual rigid body
    JPH::Body *LevelCollider = Physics.BodyInterface->CreateBody(LevelBodySettings); // Note that if we run out of bodies this can return nullptr

    LevelColliderBodyId = LevelCollider->GetID();

    // Add it to the world
    // NOTE(Kevin 2025-01-30): Why is this DontActivate?
    Physics.BodyInterface->AddBody(LevelCollider->GetID(), JPH::EActivation::DontActivate);

    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    Physics.PhysicsSystem->OptimizeBroadPhase();
}
