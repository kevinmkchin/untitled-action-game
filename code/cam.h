#pragma once

struct camera_t
{
    vec3 Position;
    vec3 Rotation; // Roll, Yaw, Pitch
    vec3 Direction;
    vec3 Right;
    vec3 Up;

    // used for spawning projectiles as this is calc'ed from yaw and pitch
    // rather than from Direction (which loses yaw and pitch info)
    quat Orientation;

    void Update(bool DoMouseLook, float LookSensitivity);

    void DoFlyCamMovement(float MoveSpeed);

    // void Shake();

    mat4 ViewFromWorldMatrix();
};
