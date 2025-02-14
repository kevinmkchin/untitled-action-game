#pragma once


struct player_t
{
    void Init();
    void Destroy();

    void LateNonPhysicsTick();
    void PrePhysicsUpdate();
    void PostPhysicsUpdate();

    vec3 Root;
    vec3 DesiredMoveDirection;
    bool JumpRequested = false;

    vec3 WalkDirectionForward;
    vec3 WalkDirectionRight;
    vec3 CameraRotation;
    vec3 CameraDirection;
    vec3 CameraRight;
    vec3 CameraUp;

    float MoveSpeed = 192.f;
    float JumpSpeed = 128.f;

    JPH::CharacterVirtual *CharacterController;

private:
    void AddToPhysicsSystem();
    void DoMovement(vec3 MovementDirection, bool DoJump, bool DoSwitchStance);

    static constexpr float CharacterHeightStanding = 48.f;
    static constexpr float CharacterRadiusStanding = 8.f;
    static constexpr float CharacterHeightCrouching = 30.f;
    static constexpr float CharacterRadiusCrouching = 8.f;
    JPH::RefConst<JPH::Shape> StandingShape;
    JPH::RefConst<JPH::Shape> CrouchingShape;
    JPH::Vec3 DesiredVelocity = JPH::Vec3::sZero(); // Smoothed value of the player input
    bool AllowSliding = false;
};

