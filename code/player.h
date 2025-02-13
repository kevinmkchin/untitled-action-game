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

    float sCharacterSpeed = 192.f;
    float sJumpSpeed = 128.f;

    JPH::CharacterVirtual *mCharacter;

private:
    void AddToPhysicsSystem();
    void DoMovement(vec3 MovementDirection, bool DoJump, bool DoSwitchStance);

    static constexpr float cCharacterHeightStanding = 48.f;
    static constexpr float cCharacterRadiusStanding = 8.f;
    static constexpr float cCharacterHeightCrouching = 30.f;
    static constexpr float cCharacterRadiusCrouching = 8.f;
    JPH::RefConst<JPH::Shape> mStandingShape;
    JPH::RefConst<JPH::Shape> mCrouchingShape;
    JPH::Vec3 mDesiredVelocity = JPH::Vec3::sZero(); // Smoothed value of the player input
    bool mAllowSliding = false;
};

