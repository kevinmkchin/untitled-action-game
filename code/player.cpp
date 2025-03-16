
void player_t::Init()
{
    StandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * CharacterHeightStanding + CharacterRadiusStanding, 0),
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * CharacterHeightStanding, CharacterRadiusStanding)).Create().Get();
    CrouchingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * CharacterHeightCrouching + CharacterRadiusCrouching, 0),
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * CharacterHeightCrouching, CharacterRadiusCrouching)).Create().Get();

    AddToPhysicsSystem();
}

void player_t::Destroy()
{
    delete CharacterController;
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
    UpdateSettings.mStickToFloorStepDown.SetY(-12.f*GAME_UNIT_TO_SI_UNITS);
    UpdateSettings.mWalkStairsStepUp.SetY(6.f*GAME_UNIT_TO_SI_UNITS);

    if (!sEnableStickToFloor)
        UpdateSettings.mStickToFloorStepDown = JPH::Vec3::sZero();
    else
        UpdateSettings.mStickToFloorStepDown = -CharacterController->GetUp() * UpdateSettings.mStickToFloorStepDown.Length();

    if (!sEnableWalkStairs)
        UpdateSettings.mWalkStairsStepUp = JPH::Vec3::sZero();
    else
        UpdateSettings.mWalkStairsStepUp = CharacterController->GetUp() * UpdateSettings.mWalkStairsStepUp.Length();

    // Update the character position
    CharacterController->ExtendedUpdate(FixedDeltaTime,
                                        -CharacterController->GetUp() * Physics.PhysicsSystem->GetGravity().Length(),
                                        UpdateSettings,
                                        Physics.PhysicsSystem->GetDefaultBroadPhaseLayerFilter(Layers::MOVING),
                                        Physics.PhysicsSystem->GetDefaultLayerFilter(Layers::MOVING),
                                        { },
                                        { },
                                        *Physics.TempAllocator);
}

void player_t::PostPhysicsUpdate()
{
    Player.Root = FromJoltVector(Player.CharacterController->GetPosition());
}

void player_t::DoMovement(vec3 MovementDirection, bool inJump, bool inSwitchStance)
{
    JPH::Vec3Arg inMovementDirection = ToJoltVectorNoConvert(MovementDirection);
    JPH::PhysicsSystem *mPhysicsSystem = Physics.PhysicsSystem;

    const bool sEnableCharacterInertia = true;
    const bool sControlMovementDuringJump = true;
    bool player_controls_horizontal_velocity = sControlMovementDuringJump || CharacterController->IsSupported();
    if (player_controls_horizontal_velocity)
    {
        // Smooth the player input
        DesiredVelocity = sEnableCharacterInertia
            ? 0.25f * inMovementDirection * MoveSpeed + 0.75f * DesiredVelocity
            : inMovementDirection * MoveSpeed;

        // True if the player intended to move
        AllowSliding = !inMovementDirection.IsNearZero();
    }
    else
    {
        // While in air we allow sliding
        AllowSliding = true;
    }

    // Update the character rotation and its up vector to match the up vector set by the user settings
    const float sUpRotationX = 0.f;
    const float sUpRotationZ = 0.f;
    JPH::Quat character_up_rotation = JPH::Quat::sEulerAngles(JPH::Vec3(sUpRotationX, 0.f, sUpRotationZ));
    CharacterController->SetUp(character_up_rotation.RotateAxisY());
    CharacterController->SetRotation(character_up_rotation);

    // A cheaper way to update the character's ground velocity,
    // the platforms that the character is standing on may have changed velocity
    CharacterController->UpdateGroundVelocity();

    // Determine new basic velocity
    JPH::Vec3 current_vertical_velocity = CharacterController->GetLinearVelocity().Dot(CharacterController->GetUp()) * CharacterController->GetUp();
    JPH::Vec3 ground_velocity = CharacterController->GetGroundVelocity();
    JPH::Vec3 new_velocity;
    bool moving_towards_ground = (current_vertical_velocity.GetY() - ground_velocity.GetY()) < 0.1f;
    if (CharacterController->GetGroundState() == JPH::CharacterVirtual::EGroundState::OnGround    // If on ground
        && (sEnableCharacterInertia?
            moving_towards_ground                                                   // Inertia enabled: And not moving away from ground
            : !CharacterController->IsSlopeTooSteep(CharacterController->GetGroundNormal())))         // Inertia disabled: And not on a slope that is too steep
    {
        // Assume velocity of ground when on ground
        new_velocity = ground_velocity;

        // Jump
        if (inJump && moving_towards_ground)
            new_velocity += JumpSpeed * CharacterController->GetUp();
    }
    else
    {
        new_velocity = current_vertical_velocity;
        if (!moving_towards_ground)
        {
            const JPH::CharacterVirtual::ContactList& ContactResults = CharacterController->GetActiveContacts();
            for (const JPH::CharacterVirtual::Contact& Result : ContactResults)
            {
                if (Result.mBodyB == LevelColliderBodyId
                    && Result.mContactNormal.Dot(JPH::Vec3(0.f,-1.f,0.f)) > 0.9f)
                {
                    new_velocity = JPH::Vec3(0.f,0.f,0.f);
                    break;
                }
            }
        }
    }

    // Gravity
    new_velocity += (character_up_rotation * mPhysicsSystem->GetGravity()) * FixedDeltaTime;

    if (player_controls_horizontal_velocity)
    {
        // Player input
        new_velocity += character_up_rotation * DesiredVelocity;
    }
    else
    {
        // Preserve horizontal velocity
        JPH::Vec3 current_horizontal_velocity = CharacterController->GetLinearVelocity() - current_vertical_velocity;
        new_velocity += current_horizontal_velocity;
    }

    // Update character velocity
    CharacterController->SetLinearVelocity(new_velocity);

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
    Settings->mShape = StandingShape;
    Settings->mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
    Settings->mCharacterPadding = 0.02f;
    Settings->mPenetrationRecoverySpeed = 1.0f;
    Settings->mPredictiveContactDistance = 0.1f;
    Settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -CharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    Settings->mEnhancedInternalEdgeRemoval = false;
    Settings->mInnerBodyShape = nullptr;
    Settings->mInnerBodyLayer = Layers::MOVING;
    Player.CharacterController = new JPH::CharacterVirtual(Settings, JPH::RVec3::sZero(), JPH::Quat::sIdentity(), 0, Physics.PhysicsSystem);
    Player.CharacterController->SetCharacterVsCharacterCollision(&Physics.CharacterVirtualsHandler);
    Physics.CharacterVirtualsHandler.Add(Player.CharacterController);

    // // Install contact listener for all characters
    // for (CharacterVirtual *character : mCharacterVsCharacterCollision.mCharacters)
    //     character->SetListener(this);
}

void player_t::LateNonPhysicsTick()
{

}
