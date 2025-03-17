#pragma once


struct player_t
{
    void Init();
    void Destroy();

    void HandleInput();
    void PrePhysicsUpdate();
    void PostPhysicsUpdate();
    void LateNonPhysicsTick();

    vec3 Root;
    vec3 DesiredMoveDirection;
    bool JumpRequested = false;

    vec3 WalkDirectionForward;
    vec3 WalkDirectionRight;
    vec3 CameraRotation;
    vec3 CameraDirection;
    vec3 CameraRight;
    vec3 CameraUp;

    float MoveSpeed = 9.f;
    float JumpSpeed = 5.f;

    JPH::CharacterVirtual *CharacterController;

    weapon_state_t Weapon;

private:
    void AddToPhysicsSystem();
    void DoMovement(vec3 MovementDirection, bool DoJump, bool DoSwitchStance);

    static constexpr float PlayerHeightStanding = 1.7f;
    static constexpr float PlayerCapsuleRadiusStanding = 0.25f;
    static constexpr float PlayerCapsuleHalfHeightStanding
        = (PlayerHeightStanding - PlayerCapsuleRadiusStanding * 2.f) * 0.5f;
    static constexpr float PlayerHeightCrouching = 0.9f;
    static constexpr float PlayerCapsuleRadiusCrouching = 0.25f;
    static constexpr float PlayerCapsuleHalfHeightCrouching
        = (PlayerHeightCrouching - PlayerCapsuleRadiusCrouching * 2.f) * 0.5f;
    JPH::RefConst<JPH::Shape> StandingShape;
    JPH::RefConst<JPH::Shape> CrouchingShape;
    JPH::Vec3 DesiredVelocity = JPH::Vec3::sZero(); // Smoothed value of the player input
    bool AllowSliding = false;
};

