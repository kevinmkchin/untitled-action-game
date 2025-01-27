
Mix_Chunk *sfx_Jump;
ModelGLTF Model_Knight;


std::vector<vec3> GameLevelColliderPoints;
std::vector<FlatPolygonCollider> GameLevelColliders;
std::vector<face_batch_t> GameLevelFaceBatches;


physics_t Physics;
player_t Player;
mat4 GameViewMatrix;

void OpenGame()
{
// #if SUNLIGHT_TEST
//     EditorDeserializeMap(wd_path("House.emf").c_str());
//     // EditorDeserializeMap(wd_path("IrradianceCachingTest.emf").c_str());
// #else
//     EditorDeserializeMap(wd_path("LightTest.emf").c_str());
// #endif
//     BuildGameMap(wd_path("buildtest.map").c_str());

    sfx_Jump = Mixer_LoadChunk(wd_path("gunshot-37055.ogg").c_str());
    LoadModelGLTF2Bin(&Model_Knight, wd_path("models/knight.glb").c_str());

    Physics.Initialize();

    LoadLevel();
}

void CloseGame()
{
    Physics.Destroy();
}

void Stuff2();
void LoadLevel()
{
    if (LoadGameMap(wd_path("buildtest.map").c_str()) == false)
        LogError("failed to load game map");

    SDL_SetRelativeMouseMode(SDL_TRUE);

    Stuff2();
}

JPH::Character *mCharacter;

void Stuff2()
{

    // // Next we can create a rigid body to serve as the floor, we make a large box
    // // Create the settings for the collision volume (the shape).
    // // Note that for simple shapes (like boxes) you can also directly construct a BoxShape.
    // BoxShapeSettings floor_shape_settings(Vec3(100.0f, 1.0f, 100.0f));
    // floor_shape_settings.SetEmbedded(); // A ref counted object on the stack (base class RefTarget) should be marked as such to prevent it from being freed when its reference count goes to 0.

    static JPH::TriangleList triangles;
    for (FlatPolygonCollider& collider : GameLevelColliders)
    {
        vec3 first = collider.pointCloudPtr[0];
        for (u32 i = 2; i < collider.pointCount; ++i)
        {
            vec3 second = collider.pointCloudPtr[i-1];
            vec3 third = collider.pointCloudPtr[i];

            JPH::Float3 jph_first(first.x, first.y, first.z);
            JPH::Float3 jph_second(second.x, second.y, second.z);
            JPH::Float3 jph_third(third.x, third.y, third.z);

            triangles.push_back(JPH::Triangle(jph_first, jph_second, jph_third));
        }
    }
    static JPH::MeshShapeSettings myLevelColliderSettings = JPH::MeshShapeSettings(triangles);
    myLevelColliderSettings.SetEmbedded();

    // Create the shape
    static JPH::ShapeSettings::ShapeResult level_shape_result = myLevelColliderSettings.Create();
    static JPH::ShapeRefC level_shape = level_shape_result.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()
    if(level_shape_result.HasError())
    {
        LogMessage("%s", level_shape_result.GetError().c_str());
    }

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    static JPH::BodyCreationSettings level_settings(level_shape, JPH::RVec3(0.0, 0.0, 0.0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    level_settings.mEnhancedInternalEdgeRemoval = true;

    // Create the actual rigid body
    JPH::Body *floor = Physics.BodyInterface->CreateBody(level_settings); // Note that if we run out of bodies this can return nullptr

    // Add it to the world
    Physics.BodyInterface->AddBody(floor->GetID(), JPH::EActivation::DontActivate);


    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    Physics.PhysicsSystem->OptimizeBroadPhase();


    // make the character collider
    static constexpr float  cCharacterHeightStanding = 48.f;
    static constexpr float  cCharacterRadiusStanding = 8.f;
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
    mCharacter = new JPH::Character(settings, JPH::RVec3(0,32,0), JPH::Quat::sIdentity(), 0, Physics.PhysicsSystem);
    mCharacter->AddToPhysicsSystem(JPH::EActivation::Activate);
}


void PrePhysicsTick()
{
#if !SUNLIGHT_TEST
    // PrimitiveDrawSolidDisc(TestLightSource, -CameraDirection, 3.f);
    // PrimitiveDrawSolidDisc(TestLightSource2, -CameraDirection, 3.f);
#endif

    // ENEMY 0
    // vec3 toPlayer = PlayerControllerRoot - Enemy0.root;
    // Enemy0.facing = Normalize(vec3(toPlayer.x, 0.f, toPlayer.z));
    // // enemy0.root += enemy0.facing * 32.f * g_DeltaTime;

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

    // temp crosshair
    ivec2 guiwh = ivec2(RenderTargetGUI.width, RenderTargetGUI.height);
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 3, guiwh.y / 2 - 3, 6, 6), vec4(0, 0, 0, 1));
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 2, guiwh.y / 2 - 2, 4, 4), vec4(1, 1, 1, 1));


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
    JPH::Character::EGroundState ground_state = mCharacter->GetGroundState();
    if (ground_state == JPH::Character::EGroundState::OnSteepGround || 
        ground_state == JPH::Character::EGroundState::NotSupported)
    {
        JPH::Vec3 normal = mCharacter->GetGroundNormal();
        normal.SetY(0.0f);
        float dot = normal.Dot(movement_direction);
        if (dot < 0.0f)
            movement_direction -= (dot * normal) / normal.LengthSq();
    }

    //// Stance switch
    //if (inSwitchStance)
    //    mCharacter->SetShape(mCharacter->GetShape() == mStandingShape ? mCrouchingShape : mStandingShape, 1.5f * mPhysicsSystem->GetPhysicsSettings().mPenetrationSlop);
    const float sCharacterSpeed = 6.0f*32.f;
    const float sJumpSpeed = 4.0f*32.f;
    if (/*sControlMovementDuringJump || */ mCharacter->IsSupported())
    {
        // Update velocity
        JPH::Vec3 current_velocity = mCharacter->GetLinearVelocity();
        // try printing magnitude of current velocity
        JPH::Vec3 desired_velocity = sCharacterSpeed * movement_direction;
        if (!desired_velocity.IsNearZero() || current_velocity.GetY() < 0.0f || !mCharacter->IsSupported())
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
        mCharacter->SetLinearVelocity(new_velocity);
    }

}

void PostPhysicsTick()
{
    static const float cCollisionTolerance = 0.05f;
    mCharacter->PostSimulation(cCollisionTolerance);

    JPH::RVec3 cpos = mCharacter->GetPosition();
    LogMessage("character pos %f, %f, %f", cpos.GetX(), cpos.GetY(), cpos.GetZ());

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
    // pre physics tick
    PrePhysicsTick();

    // physics tick
    Physics.Update();

    // post physics tick
    PostPhysicsTick();

    // draw
    RenderGameLayer();
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

    // modelMatrix = TranslationMatrix(enemy0.root) * RotationMatrix(DirectionToOrientation(enemy0.facing));
    // GLBindMatrix4fv(gameLevelShader, "modelMatrix", 1, modelMatrix.ptr());
    // RenderModelGLTF(model_Knight);

    // PRIMITIVES
    if (KeysPressed[SDL_SCANCODE_X])
        DoPrimitivesDepthTest = !DoPrimitivesDepthTest;
    if (DoPrimitivesDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));
}
