#pragma once


struct player_t
{
    vec3 Root;
    vec3 DesiredMoveDirection;
    bool JumpRequested = false;

    vec3 WalkDirectionForward;
    vec3 WalkDirectionRight;
    vec3 CameraRotation;
    vec3 CameraDirection;
    vec3 CameraRight;
    vec3 CameraUp;

    JPH::Character *mCharacter;
};

