#pragma once

enum weapon_types_t
{
    NAILGUN,
    WEAPON_TYPES_COUNT
};

struct weapon_state_t
{
    weapon_types_t ActiveType = NAILGUN;

    float Cooldown = 0.f;
};

void TickWeapon(weapon_state_t *State, bool LMB, bool RMB);
void RenderWeapon(weapon_state_t *State, float *ProjFromView, float *WorldFromView);

extern ModelGLTF Model_Nailgun;
