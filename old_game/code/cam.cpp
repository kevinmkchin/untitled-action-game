#include "cam.h"

void camera_t::Update(bool DoMouseLook, vec2 MouseDelta, float LookSensitivity)
{
    if (DoMouseLook)
    {
        float YawDelta = MouseDelta.x * LookSensitivity;
        float PitchDelta = MouseDelta.y * LookSensitivity;
        Rotation.y -= YawDelta;
        Rotation.z -= PitchDelta;
        if (Rotation.z > 89.f)
            Rotation.z = 89.f;
        if (Rotation.z < -89.f)
            Rotation.z = -89.f;
    }

    Orientation = EulerToQuat(Rotation * GM_D2R);
    Direction = Normalize(OrientationToDirection(Orientation));
    Right = Normalize(Cross(Direction, GM_UP_VECTOR));
    Up = Normalize(Cross(Right, Direction));

    ViewPosition = Position;
    ViewForward = Direction;
    ViewUp = Up;
}

void camera_t::DoFlyCamMovement(float MoveSpeed)
{
    vec3 PositionDelta;
    if (KeysCurrent[SDL_SCANCODE_W])
        PositionDelta += Direction * MoveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_A])
        PositionDelta += -Right * MoveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_S])
        PositionDelta += -Direction * MoveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_D])
        PositionDelta += Right * MoveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_Q])
        PositionDelta += -GM_UP_VECTOR * MoveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_E])
        PositionDelta += GM_UP_VECTOR * MoveSpeed * DeltaTime;
    Position += PositionDelta;
}

mat4 camera_t::ViewFromWorldMatrix() 
{
    return ViewMatrixLookAt(ViewPosition, ViewPosition + ViewForward, ViewUp);
}

void camera_t::ApplyKnockback(float KnockbackInRadians, float RecoveryPerSecond)
{
    this->CurrentKnockback = KnockbackInRadians;
    this->RecoveryPerSecond = RecoveryPerSecond;
}

void camera_t::UpdateKnockbackAndStrafeTilt(bool StrafeLeft, bool StrafeRight)
{
    // Knockback
    quat RotKnock = EulerToQuat(0.f,0.f,CurrentKnockback);
    ViewForward = Normalize(OrientationToDirection(Normalize(Mul(Orientation, RotKnock))));
    ViewUp = Normalize(Cross(Right, ViewForward));

    if (CurrentKnockback > 0.f)
    {
        CurrentKnockback -= RecoveryPerSecond * DeltaTime;
        CurrentKnockback = GM_max(0.f, CurrentKnockback);
    }

    // Stafe tilt
    const float TiltSpeed = 15;
    const float MaxTilt = 0.035f;
    float DesiredCamTilt = 0.f;
    if(StrafeRight)
        DesiredCamTilt += MaxTilt;
    if(StrafeLeft)
        DesiredCamTilt += -MaxTilt;
    StrafeTilt = Lerp(StrafeTilt, DesiredCamTilt, DeltaTime * TiltSpeed);
    quat fromto = RotationFromTo(ViewUp, Right);
    quat sle = Slerp(quat(), fromto, StrafeTilt);
    vec3 CameraUpWithSway = RotateVector(ViewUp, sle);
    ViewUp = CameraUpWithSway;
}

