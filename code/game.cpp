
Mix_Chunk *sfx_Jump;
ModelGLTF Model_Knight;

// extern
std::vector<face_batch_t> GameLevelFaceBatches;
bool GameLoopCanRun = true;

// internal
bool LevelLoaded = false;
physics_t Physics;
player_t Player;
mat4 GameViewMatrix;
JPH::BodyID LevelColliderBodyId; 

#ifdef JPH_DEBUG_RENDERER
jph_debug_draw_gl3_t *JoltDebugDraw;
#endif // JPH_DEBUG_RENDERER

void InitializeGame()
{

    // testing stuff here
    sfx_Jump = Mixer_LoadChunk(wd_path("gunshot-37055.ogg").c_str());
    LoadModelGLTF2Bin(&Model_Knight, wd_path("models/knight.glb").c_str());

    Physics.Initialize();

    CreateAndRegisterPlayerPhysicsController();

    Enemies.free();

#if INTERNAL_BUILD
    RecastDebugDrawer.Init();
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    JoltDebugDraw = new jph_debug_draw_gl3_t();
    JoltDebugDraw->Init();
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
    JoltDebugDraw->Destroy();
    delete JoltDebugDraw;
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

    Player.mCharacter->SetPositionAndRotation(
        ToJoltVec3(MapLoadResult.PlayerStartPosition),
        ToJoltQuat(EulerToQuat(MapLoadResult.PlayerStartRotation)));
    // Apply to Jolt Character or my camera rotation?

    enemy_t Enemy0;
    enemy_t Enemy1;
    enemy_t Enemy2;

    Enemy0.Init();
    Enemy1.Init();
    Enemy2.Init();

    GetRandomPointOnNavMesh((float*)&Enemy0.Position);
    GetRandomPointOnNavMesh((float*)&Enemy1.Position);
    GetRandomPointOnNavMesh((float*)&Enemy2.Position);

    Enemies.put(Enemy0);
    Enemies.put(Enemy1);
    Enemies.put(Enemy2);

    LevelLoaded = true;
}

void UnloadPreviousLevel()
{
    if (!LevelColliderBodyId.IsInvalid())
    {
        Physics.BodyInterface->RemoveBody(LevelColliderBodyId);
        Physics.BodyInterface->DestroyBody(LevelColliderBodyId);
    }

    // Enemies.free();
    DestroyRecastNavMesh();

    LevelLoaded = false;
}

void PrePhysicsTick()
{
    UpdateAllEnemies();

    // CALCULATE PLAYER FACING DIRECTION
    float camYawDelta = MouseDelta.x * 0.085f;
    float camPitchDelta = MouseDelta.y * 0.085f;
    Player.CameraRotation.y -= camYawDelta;
    Player.CameraRotation.z -= camPitchDelta;
    if (Player.CameraRotation.z > 89.f)
        Player.CameraRotation.z = 89.f;
    if (Player.CameraRotation.z < -89.f)
        Player.CameraRotation.z = -89.f;
    Player.CameraDirection = Normalize(OrientationToDirection(EulerToQuat(Player.CameraRotation * GM_DEG2RAD)));
    Player.CameraRight = Normalize(Cross(Player.CameraDirection, GM_UP_VECTOR));
    Player.CameraUp = Normalize(Cross(Player.CameraRight, Player.CameraDirection));
    Player.WalkDirectionRight = Player.CameraRight;
    Player.WalkDirectionForward = Normalize(Cross(GM_UP_VECTOR, Player.WalkDirectionRight));

    // PLAYER MOVE
    vec3 desiredMoveDirection;
    if (KeysCurrent[SDL_SCANCODE_W])
        desiredMoveDirection += Player.WalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_A])
        desiredMoveDirection += -Player.WalkDirectionRight;
    if (KeysCurrent[SDL_SCANCODE_S])
        desiredMoveDirection += -Player.WalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_D])
        desiredMoveDirection += Player.WalkDirectionRight;
    

    // Cancel movement in opposite direction of normal when touching something we can't walk up
    JPH::Vec3 movement_direction = JPH::Vec3(desiredMoveDirection.x, desiredMoveDirection.y, desiredMoveDirection.z);
    JPH::Character::EGroundState ground_state = Player.mCharacter->GetGroundState();
    if (ground_state == JPH::Character::EGroundState::OnSteepGround || 
        ground_state == JPH::Character::EGroundState::NotSupported)
    {
        JPH::Vec3 normal = Player.mCharacter->GetGroundNormal();
        normal.SetY(0.0f);
        float dot = normal.Dot(movement_direction);
        if (dot < 0.0f)
            movement_direction -= (dot * normal) / normal.LengthSq();
    }

    //// Stance switch
    //if (inSwitchStance)
    //    Player.mCharacter->SetShape(Player.mCharacter->GetShape() == mStandingShape ? mCrouchingShape : mStandingShape, 1.5f * mPhysicsSystem->GetPhysicsSettings().mPenetrationSlop);
    const float sCharacterSpeed = 6.0f*32.f;
    const float sJumpSpeed = 4.0f*32.f;
    if (/*sControlMovementDuringJump || */ Player.mCharacter->IsSupported())
    {
        // Update velocity
        JPH::Vec3 current_velocity = Player.mCharacter->GetLinearVelocity();
        // try printing magnitude of current velocity
        JPH::Vec3 desired_velocity = sCharacterSpeed * movement_direction;
        if (!desired_velocity.IsNearZero() || current_velocity.GetY() < 0.0f || !Player.mCharacter->IsSupported())
            desired_velocity.SetY(current_velocity.GetY());
        JPH::Vec3 new_velocity = 0.75f * current_velocity + 0.25f * desired_velocity;

        // Jump
        if (KeysPressed[SDL_SCANCODE_SPACE] && ground_state == JPH::Character::EGroundState::OnGround)
        {
            new_velocity += JPH::Vec3(0, sJumpSpeed, 0);

            // static int channelrotationtesting = 0;
            // Mix_VolumeChunk(sfx_Jump, 48);
            // Mix_PlayChannel(channelrotationtesting++%3, sfx_Jump, 0);
        }

        // Update the velocity
        Player.mCharacter->SetLinearVelocity(new_velocity);
    }

#ifdef JPH_DEBUG_RENDERER
    JoltDebugDraw->Ready();

    // Physics.BodyInterface->GetShape(LevelColliderBodyId)->Draw(JoltDebugDraw,
    //     Physics.BodyInterface->GetCenterOfMassTransform(LevelColliderBodyId),
    //     JPH::Vec3::sReplicate(1.0f), JPH::Color(0,255,0,70), true, false);

    // Player.mCharacter->GetShape()->Draw(JoltDebugDraw, 
    //     Physics.BodyInterface->GetCenterOfMassTransform(Player.mCharacter->GetBodyID()), 
    //     JPH::Vec3::sReplicate(1.0f), JPH::Color::sGreen, false, true);

    // JoltDebugDrawCharacterState(JoltDebugDraw, Player.mCharacter,   
    //     Player.mCharacter->GetWorldTransform(), 
    //     Player.mCharacter->GetLinearVelocity());
#endif // JPH_DEBUG_RENDERER
}

void PostPhysicsTick()
{
    // NOTE(Kevin): Jolt sample uses about 3.7% of the character height, but 
    //              big value causes glitches for me
    static const float cCollisionTolerance = 0.05f;
    Player.mCharacter->PostSimulation(cCollisionTolerance);

    JPH::RVec3 cpos = Player.mCharacter->GetPosition();
    // LogMessage("character pos %f, %f, %f", cpos.GetX(), cpos.GetY(), cpos.GetZ());

    Player.Root.x = cpos.GetX();
    Player.Root.y = cpos.GetY();
    Player.Root.z = cpos.GetZ();


    // PLAYER CAMERA 
    vec3 CameraPosOffsetFromRoot = vec3(0,40,0);
    vec3 CameraPosition = Player.Root + CameraPosOffsetFromRoot;
    // LogMessage("pos %f, %f, %f", playerControllerRoot.x, playerControllerRoot.y, playerControllerRoot.z);
    // LogMessage("dir y %f, z %f\n", cameraRotation.y, cameraRotation.z);

    static float camLean = 0.f;
    static float desiredCamLean = 0.f;
    const float camLeanSpeed = 15;
    const float maxCamLean = 0.07f;
    desiredCamLean = 0.f;
    if(KeysCurrent[SDL_SCANCODE_D])
        desiredCamLean += maxCamLean;
    if(KeysCurrent[SDL_SCANCODE_A])
        desiredCamLean += -maxCamLean;
    camLean = Lerp(camLean, desiredCamLean, DeltaTime * camLeanSpeed);

    quat fromto = RotationFromTo(Player.CameraUp, Player.CameraRight);
    quat sle = Slerp(quat(), fromto, camLean);
    vec3 CameraUpWithSway = RotateVector(Player.CameraUp, sle);
    float dot = Dot(Normalize(Cross(CameraUpWithSway, Player.CameraRight)), Player.CameraDirection);
    if (dot < 0.99f)
        printf("bad cam up %f\n", dot);
    GameViewMatrix = ViewMatrixLookAt(CameraPosition, CameraPosition + Player.CameraDirection, CameraUpWithSway);
}

void DoGameLoop()
{
    if (!GameLoopCanRun) 
        return;

    // pre physics tick
    PrePhysicsTick();

    // physics tick
    Physics.Tick();

    // post physics tick
    PostPhysicsTick();

    // Do animation loop

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
    mat4 viewMatrix = GameViewMatrix;

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
    SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

#if INTERNAL_BUILD
    if (KeysCurrent[SDL_SCANCODE_B])
    {
        DoDebugDrawRecast(perspectiveMatrix.ptr(), viewMatrix.ptr(), DRAWMODE_NAVMESH);
        DebugDrawFollowPath();
    }
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    mat4 ViewProjectionMatrix = perspectiveMatrix * viewMatrix;
    JoltDebugDraw->Flush(ViewProjectionMatrix.ptr());
#endif // JPH_DEBUG_RENDERER

    UseShader(EditorShader_Scene);
    glEnable(GL_DEPTH_TEST);
    GLBindMatrix4fv(EditorShader_Scene, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(EditorShader_Scene, "viewMatrix", 1, viewMatrix.ptr());

    mat4 ModelMatrix = TranslationMatrix(Enemies[0].Position) * RotationMatrix(Enemies[0].Orientation);
    GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, ModelMatrix.ptr());
    RenderModelGLTF(Model_Knight);
    ModelMatrix = TranslationMatrix(Enemies[1].Position) * RotationMatrix(Enemies[1].Orientation);
    GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, ModelMatrix.ptr());
    RenderModelGLTF(Model_Knight);
    ModelMatrix = TranslationMatrix(Enemies[2].Position) * RotationMatrix(Enemies[2].Orientation);
    GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, ModelMatrix.ptr());
    RenderModelGLTF(Model_Knight);
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

            JPH::Float3 jph_first(First.x, First.y, First.z);
            JPH::Float3 jph_second(Second.x, Second.y, Second.z);
            JPH::Float3 jph_third(Third.x, Third.y, Third.z);

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

void CreateAndRegisterPlayerPhysicsController()
{
    // make the character collider
    static constexpr float cCharacterHeightStanding = 48.f;
    static constexpr float cCharacterRadiusStanding = 8.f;
    JPH::RefConst<JPH::Shape> mStandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * cCharacterHeightStanding + cCharacterRadiusStanding, 0), 
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * cCharacterHeightStanding, cCharacterRadiusStanding)).Create().Get();

    JPH::Ref<JPH::CharacterSettings> settings = new JPH::CharacterSettings();
    settings->mMaxSlopeAngle = JPH::DegreesToRadians(45.0f);
    settings->mLayer = Layers::MOVING;
    settings->mShape = mStandingShape;
    settings->mFriction = 0.5f;
    settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -cCharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    Player.mCharacter = new JPH::Character(settings, JPH::RVec3(0,32,0), JPH::Quat::sIdentity(), 0, Physics.PhysicsSystem);
    Player.mCharacter->AddToPhysicsSystem(JPH::EActivation::Activate);
}
