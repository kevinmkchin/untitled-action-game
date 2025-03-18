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

constexpr u32 ProjectileFlag_Dead   = 0x0001;  // bit 0
constexpr u32 ProjectileFlag_Bullet = 0x0002;  // bit 1
// constexpr u32                    = 0x0004;  // bit 2
// constexpr u32 ProjectileFlag_Gib = 0x0008;  // bit 3
constexpr u32 DamageType_Small      = 0x0010;  // bit 4
// constexpr u32 DamageType_Big        = 0x0020;  // bit 5
// constexpr u32                    = 0x0040;  // bit 6
// constexpr u32                    = 0x0080;  // bit 7
// constexpr u32                    = 0x0100;  // bit 8
// constexpr u32                    = 0x0200;  // bit 9
// constexpr u32                    = 0x1000;  // bit 12
// constexpr u32                    = 0x2000;  // bit 13
// constexpr u32                    = 0x4000;  // bit 14
// constexpr u32                    = 0x8000;  // bit 15

struct projectile_t
{
    // projectile damage type (flags?)
    float Damage;
    u16 Flags = 0;
    // damage radius and falloff if explosive type
    // velocity
    // body
    JPH::BodyID BodyId;
};
dynamic_array<projectile_t> LiveProjectiles;
void SpawnProjectile();
void PrePhysicsUpdateProjectiles();
void PostPhysicsUpdateProjectiles();

extern ModelGLTF Model_Nailgun;
