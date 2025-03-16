#pragma once


struct player_t
{
    void Init();
    void Destroy();

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

    float MoveSpeed = 6.f;
    float JumpSpeed = 4.f;

    JPH::CharacterVirtual *CharacterController;

private:
    void AddToPhysicsSystem();
    void DoMovement(vec3 MovementDirection, bool DoJump, bool DoSwitchStance);

    static constexpr float CharacterHeightStanding = 1.5f;
    static constexpr float CharacterRadiusStanding = 0.25f;
    static constexpr float CharacterHeightCrouching = 0.9f;
    static constexpr float CharacterRadiusCrouching = 0.25f;
    JPH::RefConst<JPH::Shape> StandingShape;
    JPH::RefConst<JPH::Shape> CrouchingShape;
    JPH::Vec3 DesiredVelocity = JPH::Vec3::sZero(); // Smoothed value of the player input
    bool AllowSliding = false;
};

