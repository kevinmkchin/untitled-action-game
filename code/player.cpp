
void player_t::Init()
{
    StandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, PlayerCapsuleHalfHeightStanding + PlayerCapsuleRadiusStanding, 0),
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(PlayerCapsuleHalfHeightStanding, PlayerCapsuleRadiusStanding)).Create().Get();
    CrouchingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, PlayerCapsuleHalfHeightCrouching + PlayerCapsuleRadiusCrouching, 0),
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(PlayerCapsuleHalfHeightCrouching, PlayerCapsuleRadiusCrouching)).Create().Get();

    AddToPhysicsSystem();
}

void player_t::Destroy()
{
    delete CharacterController;
}

void player_t::HandleInput()
{
    // CALCULATE PLAYER FACING DIRECTION
    bool DoMouseLook = SDL_GetRelativeMouseMode();
    PlayerCam.Update(DoMouseLook, 0.085f);
    WalkDirectionRight = PlayerCam.Right;
    WalkDirectionForward = Normalize(Cross(GM_UP_VECTOR, PlayerCam.Right));

    // PLAYER MOVE
    WASD = KeysCurrent[SDL_SCANCODE_W] || KeysCurrent[SDL_SCANCODE_A] ||
        KeysCurrent[SDL_SCANCODE_S] || KeysCurrent[SDL_SCANCODE_D];

    DesiredMoveDirection = vec3();
    if (KeysCurrent[SDL_SCANCODE_W])
        DesiredMoveDirection += WalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_A])
        DesiredMoveDirection += -WalkDirectionRight;
    if (KeysCurrent[SDL_SCANCODE_S])
        DesiredMoveDirection += -WalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_D])
        DesiredMoveDirection += WalkDirectionRight;

    if (KeysPressed[SDL_SCANCODE_SPACE])
        JumpRequested = true;

    // SHOOT
    bool LMBPressed = MouseCurrent & SDL_BUTTON(SDL_BUTTON_LEFT);
    bool RMBPressed = MouseCurrent & SDL_BUTTON(SDL_BUTTON_RIGHT);
    Weapon.Owner = this;
    TickWeapon(&Weapon, LMBPressed, RMBPressed);
}

void player_t::PrePhysicsUpdate()
{
    DoMovement(DesiredMoveDirection, JumpRequested, false);
    JumpRequested = false;

    // Settings for our update function
    JPH::CharacterVirtual::ExtendedUpdateSettings UpdateSettings;

    const bool sEnableStickToFloor = true;
    const bool sEnableWalkStairs = true;

    // TODO get rid of these after we scale down units for physics
    UpdateSettings.mStickToFloorStepDown.SetY(ToJoltUnit(-12.f));
    UpdateSettings.mWalkStairsStepUp.SetY(ToJoltUnit(6.f));

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
                                        Physics.PhysicsSystem->GetDefaultBroadPhaseLayerFilter(Layers::PLAYER),
                                        Physics.PhysicsSystem->GetDefaultLayerFilter(Layers::PLAYER),
                                        { },
                                        { },
                                        *Physics.TempAllocator);
}

void player_t::PostPhysicsUpdate()
{
    Root = FromJoltVector(CharacterController->GetPosition());

    if (!FlyCamActive)
        PlayerCam.Position = Root + CamOffsetFromRoot;
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
    Settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -PlayerCapsuleRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    Settings->mEnhancedInternalEdgeRemoval = false;
    Settings->mInnerBodyShape = nullptr;
    Settings->mInnerBodyLayer = Layers::PLAYER;
    CharacterController = new JPH::CharacterVirtual(Settings, JPH::RVec3::sZero(), JPH::Quat::sIdentity(), 0, Physics.PhysicsSystem);
    CharacterController->SetCharacterVsCharacterCollision(&Physics.CharacterVirtualsHandler);
    Physics.CharacterVirtualsHandler.Add(CharacterController);

    // // Install contact listener for all characters
    // for (CharacterVirtual *character : mCharacterVsCharacterCollision.mCharacters)
    //     character->SetListener(this);
}

void player_t::LateNonPhysicsTick()
{
    if (FlyCamActive)
    {
        PlayerCam.DoFlyCamMovement(250.f);
    }
    else
    {
        PlayerCam.UpdateKnockbackAndStrafeTilt(KeysCurrent[SDL_SCANCODE_A], KeysCurrent[SDL_SCANCODE_D]);

        // Move cam up and down while walking
        static float WalkYOffset = 0.f;
        static float WalkYChangePerSecond = 60.f;
        if (!WASD && WalkYChangePerSecond < 0.f)
        {
            WalkYChangePerSecond = 25.f;
        }
        if (WASD && CharacterController->IsSupported()
            && WalkYChangePerSecond > 0.f && WalkYOffset > 0.0f)
        {
            WalkYOffset = 0.f;
            WalkYChangePerSecond = -40.f;
        }
        if (WalkYOffset < -11.f)
            WalkYChangePerSecond = 25.f;
        if (WalkYOffset <= 0.f)
            WalkYOffset += WalkYChangePerSecond * DeltaTime;
        CamOffsetFromRoot.y = YOffsetFromRoot + fmin(0.f, WalkYOffset);
    }

}
