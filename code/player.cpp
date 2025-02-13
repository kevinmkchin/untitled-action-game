
void player_t::Init()
{
    mStandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * cCharacterHeightStanding + cCharacterRadiusStanding, 0), 
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * cCharacterHeightStanding, cCharacterRadiusStanding)).Create().Get();
    mCrouchingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * cCharacterHeightCrouching + cCharacterRadiusCrouching, 0), 
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * cCharacterHeightCrouching, cCharacterRadiusCrouching)).Create().Get();

    AddToPhysicsSystem();
}

void player_t::Destroy()
{
    delete mCharacter;
}

void player_t::LateNonPhysicsTick()
{
#ifdef JPH_DEBUG_RENDERER
    // JPH::RMat44 COM = mCharacter->GetCenterOfMassTransform();
    // JPH::RMat44 WorldTransform = mCharacter->GetWorldTransform();

    // if (mCharacter->GetShape() == mStandingShape)
    //     JoltDebugDraw->DrawCapsule(COM, 0.5f * cCharacterHeightStanding, cCharacterRadiusStanding + mCharacter->GetCharacterPadding(), JPH::Color::sGreen, JPH::DebugRenderer::ECastShadow::Off, JPH::DebugRenderer::EDrawMode::Wireframe);
    // else
    //     JoltDebugDraw->DrawCapsule(COM, 0.5f * cCharacterHeightCrouching, cCharacterRadiusCrouching + mCharacter->GetCharacterPadding(), JPH::Color::sGreen, JPH::DebugRenderer::ECastShadow::Off, JPH::DebugRenderer::EDrawMode::Wireframe);

    // // Doesn't really work
    // // static JPH::RVec3 OldPosition = mCharacter->GetPosition();
    // // JPH::RVec3 NewPosition = mCharacter->GetPosition();
    // // JPH::Vec3 EffectiveVelocity = JPH::Vec3(NewPosition - OldPosition) / DeltaTime;
    // // OldPosition = NewPosition;

    // JoltDebugDrawCharacterState(JoltDebugDraw, mCharacter,   
    //     WorldTransform, mCharacter->GetLinearVelocity());
#endif
}

void player_t::PrePhysicsUpdate()
{
    Player.DoMovement(DesiredMoveDirection, JumpRequested, false);
    JumpRequested = false;

    // Settings for our update function
    JPH::CharacterVirtual::ExtendedUpdateSettings UpdateSettings;

    const bool sEnableStickToFloor = true;
    const bool sEnableWalkStairs = true;

    // TODO get rid of these after we scale down units for physics
    UpdateSettings.mStickToFloorStepDown.SetY(-16.f);
    UpdateSettings.mWalkStairsStepUp.SetY(6.f);

    if (!sEnableStickToFloor)
        UpdateSettings.mStickToFloorStepDown = JPH::Vec3::sZero();
    else
        UpdateSettings.mStickToFloorStepDown = -mCharacter->GetUp() * UpdateSettings.mStickToFloorStepDown.Length();

    if (!sEnableWalkStairs)
        UpdateSettings.mWalkStairsStepUp = JPH::Vec3::sZero();
    else
        UpdateSettings.mWalkStairsStepUp = mCharacter->GetUp() * UpdateSettings.mWalkStairsStepUp.Length();

    // Update the character position
    mCharacter->ExtendedUpdate(FixedDeltaTime,
        -mCharacter->GetUp() * Physics.PhysicsSystem->GetGravity().Length(),
        UpdateSettings,
        Physics.PhysicsSystem->GetDefaultBroadPhaseLayerFilter(Layers::MOVING),
        Physics.PhysicsSystem->GetDefaultLayerFilter(Layers::MOVING),
        { },
        { },
        *Physics.TempAllocator);
}

void player_t::PostPhysicsUpdate()
{
    Player.Root = FromJoltVec3(Player.mCharacter->GetPosition());
}

void player_t::DoMovement(vec3 MovementDirection, bool inJump, bool inSwitchStance)
{
    JPH::Vec3Arg inMovementDirection = ToJoltVec3(MovementDirection);
    JPH::PhysicsSystem *mPhysicsSystem = Physics.PhysicsSystem;

    const bool sEnableCharacterInertia = true;
    const bool sControlMovementDuringJump = false;
    bool player_controls_horizontal_velocity = sControlMovementDuringJump || mCharacter->IsSupported();
    if (player_controls_horizontal_velocity)
    {
        // Smooth the player input
        mDesiredVelocity = sEnableCharacterInertia 
            ? 0.25f * inMovementDirection * sCharacterSpeed + 0.75f * mDesiredVelocity 
            : inMovementDirection * sCharacterSpeed;

        // True if the player intended to move
        mAllowSliding = !inMovementDirection.IsNearZero();
    }
    else
    {
        // While in air we allow sliding
        mAllowSliding = true;
    }

    // Update the character rotation and its up vector to match the up vector set by the user settings
    const float sUpRotationX = 0.f;
    const float sUpRotationZ = 0.f;
    JPH::Quat character_up_rotation = JPH::Quat::sEulerAngles(JPH::Vec3(sUpRotationX, 0.f, sUpRotationZ));
    mCharacter->SetUp(character_up_rotation.RotateAxisY());
    mCharacter->SetRotation(character_up_rotation);

    // A cheaper way to update the character's ground velocity,
    // the platforms that the character is standing on may have changed velocity
    mCharacter->UpdateGroundVelocity();

    // Determine new basic velocity
    JPH::Vec3 current_vertical_velocity = mCharacter->GetLinearVelocity().Dot(mCharacter->GetUp()) * mCharacter->GetUp();
    JPH::Vec3 ground_velocity = mCharacter->GetGroundVelocity();
    JPH::Vec3 new_velocity;
    bool moving_towards_ground = (current_vertical_velocity.GetY() - ground_velocity.GetY()) < 0.1f;
    if (mCharacter->GetGroundState() == JPH::CharacterVirtual::EGroundState::OnGround    // If on ground
        && (sEnableCharacterInertia?
            moving_towards_ground                                                   // Inertia enabled: And not moving away from ground
            : !mCharacter->IsSlopeTooSteep(mCharacter->GetGroundNormal())))         // Inertia disabled: And not on a slope that is too steep
    {
        // Assume velocity of ground when on ground
        new_velocity = ground_velocity;

        // Jump
        if (inJump && moving_towards_ground)
            new_velocity += sJumpSpeed * mCharacter->GetUp();
    }
    else
        new_velocity = current_vertical_velocity;

    // Gravity
    new_velocity += (character_up_rotation * mPhysicsSystem->GetGravity()) * FixedDeltaTime;

    if (player_controls_horizontal_velocity)
    {
        // Player input
        new_velocity += character_up_rotation * mDesiredVelocity;
    }
    else
    {
        // Preserve horizontal velocity
        JPH::Vec3 current_horizontal_velocity = mCharacter->GetLinearVelocity() - current_vertical_velocity;
        new_velocity += current_horizontal_velocity;
    }

    // Update character velocity
    mCharacter->SetLinearVelocity(new_velocity);

    // // Stance switch
    // if (inSwitchStance)
    // {
    //     bool is_standing = mCharacter->GetShape() == mStandingShape;
    //     const JPH::Shape *shape = is_standing ? mCrouchingShape : mStandingShape;
    //     if (mCharacter->SetShape(shape, 1.5f * mPhysicsSystem->GetPhysicsSettings().mPenetrationSlop, mPhysicsSystem->GetDefaultBroadPhaseLayerFilter(Layers::MOVING), mPhysicsSystem->GetDefaultLayerFilter(Layers::MOVING), { }, { }, *mTempAllocator))
    //     {
    //         const JPH::Shape *inner_shape = is_standing? mInnerCrouchingShape : mInnerStandingShape;
    //         mCharacter->SetInnerBodyShape(inner_shape);
    //     }
    // }
}

void player_t::AddToPhysicsSystem()
{
    JPH::Ref<JPH::CharacterVirtualSettings> Settings = new JPH::CharacterVirtualSettings();
    Settings->mMaxSlopeAngle = JPH::DegreesToRadians(45.0f);
    Settings->mMaxStrength = 100.0f; // Maximum force with which the character can push other bodies (N)
    Settings->mShape = mStandingShape;
    Settings->mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
    Settings->mCharacterPadding = 0.02f;
    Settings->mPenetrationRecoverySpeed = 1.0f;
    Settings->mPredictiveContactDistance = 3.f;
    Settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -cCharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    Settings->mEnhancedInternalEdgeRemoval = false;
    Settings->mInnerBodyShape = nullptr;
    Settings->mInnerBodyLayer = Layers::MOVING;
    Player.mCharacter = new JPH::CharacterVirtual(Settings, JPH::RVec3::sZero(), JPH::Quat::sIdentity(), 0, Physics.PhysicsSystem);
    Player.mCharacter->SetCharacterVsCharacterCollision(&Physics.CharacterVirtualsHandler);
    Physics.CharacterVirtualsHandler.Add(Player.mCharacter);

    // // Install contact listener for all characters
    // for (CharacterVirtual *character : mCharacterVsCharacterCollision.mCharacters)
    //     character->SetListener(this);
}
