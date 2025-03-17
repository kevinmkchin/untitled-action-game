#include "cam.h"

void camera_t::Update(bool DoMouseLook, float LookSensitivity)
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

    Direction = Normalize(OrientationToDirection(EulerToQuat(Rotation * GM_D2R)));
    Right = Normalize(Cross(Direction, GM_UP_VECTOR));
    Up = Normalize(Cross(Right, Direction));
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

// void Shake();

mat4 camera_t::ViewFromWorldMatrix() 
{
    return ViewMatrixLookAt(Position, Position + Direction, Up);
}
