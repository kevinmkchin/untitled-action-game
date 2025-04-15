#pragma once

#include "common.h"

struct camera_t
{
    vec3 Position;
    vec3 Rotation; // Roll, Yaw, Pitch
    vec3 Direction;
    vec3 Right;
    vec3 Up;

    // Used for ViewMatrix calculation i.e. rendering
    // This lets me shake or tilt the camera for rendering only
    vec3 ViewPosition;
    vec3 ViewForward;
    vec3 ViewUp;

    // used for spawning projectiles as this is calc'ed from yaw and pitch
    // rather than from Direction (which loses yaw and pitch info)
    quat Orientation;

    void Update(bool DoMouseLook, vec2 MouseDelta, float LookSensitivity);
    void DoFlyCamMovement(float MoveSpeed);
    mat4 ViewFromWorldMatrix();

    // Camera extensions
    void ApplyKnockback(float KnockbackInRadians, float RecoveryPerSecond);

    void UpdateKnockbackAndStrafeTilt(bool StrafeLeft, bool StrafeRight);

private:
    float CurrentKnockback = 0.f;
    float RecoveryPerSecond = 0.f;

    float StrafeTilt = 0.f;
};



