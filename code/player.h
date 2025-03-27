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
    bool WASD;

    vec3 WalkDirectionForward;
    vec3 WalkDirectionRight;

    float MoveSpeed = 9.f;
    float JumpSpeed = 5.f;

    float Health = 100.f;

    JPH::CharacterVirtual *CharacterController;

    camera_t PlayerCam;
    static constexpr float YOffsetFromRoot = 64.f; 
    vec3 CamOffsetFromRoot = vec3(0,YOffsetFromRoot,0);

    weapon_state_t Weapon;

private:

    void DoPhysicsMovement(vec3 MovementDirection, bool DoJump, bool DoSwitchStance);

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

