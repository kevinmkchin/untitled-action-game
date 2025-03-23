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

    vec4 MuzzleFlash; // w is timer

    struct player_t *Owner;

    // Nailgun
    float NailgunRotation = GM_HALFPI;
    float NailgunRotationVelocity = 0.f;
    static constexpr float NailgunRotationMaxVelocity = 16.f;
    static constexpr float NailgunRotationAcceleration = -16.f;
};

void TickWeapon(weapon_state_t *State, bool LMB, bool RMB);
void RenderWeapon(weapon_state_t *State, float *ProjFromView, float *WorldFromView);

constexpr u32 ProjectileFlag_Dead   = 0x0001;  // bit 0
constexpr u32 ProjectileFlag_Bullet = 0x0002;  // bit 1
// constexpr u32                    = 0x0004;  // bit 2
// constexpr u32 ProjectileFlag_Gib = 0x0008;  // bit 3
constexpr u32 DamageType_Small      = 0x0010;  // bit 4
// constexpr u32 DamageType_Big     = 0x0020;  // bit 5
// constexpr u32                    = 0x0040;  // bit 6
// constexpr u32                    = 0x0080;  // bit 7
// constexpr u32                    = 0x0100;  // bit 8
// constexpr u32                    = 0x0200;  // bit 9
// constexpr u32                    = 0x1000;  // bit 12
// constexpr u32                    = 0x2000;  // bit 13
// constexpr u32                    = 0x4000;  // bit 14
// constexpr u32                    = 0x8000;  // bit 15

enum projectile_type : u32
{
    PROJECTILE_NAIL = 0,
    PROJECTILE_TYPE_COUNT = 1
};

struct projectile_data_t
{
    float BulletDamage = 0.f;
};

struct projectile_t
{
    u32 Flags = 0;
    projectile_type Type = PROJECTILE_TYPE_COUNT;
    quat RenderOrientation;
    JPH::BodyID BodyId;
};

struct projectile_hit_info_t
{
    const JPH::Body *Body1;
    const JPH::Body *Body2;
    const JPH::ContactManifold *Manifold;
};

extern projectile_data_t ProjectileDatabase[PROJECTILE_TYPE_COUNT];
extern dynamic_array<projectile_t> LiveProjectiles;
extern dynamic_array<projectile_hit_info_t> ProjectileHitInfos;

void PopulateProjectileDatabase(); // Called once at start of game
void SpawnProjectile(vec3 Pos, vec3 Dir, quat Orient);
void KillProjectile(projectile_t *ProjectileToKill);
void PrePhysicsUpdateProjectiles();
void PostPhysicsUpdateProjectiles();
void RenderProjectiles(const mat4 &ProjFromView, const mat4 &WorldFromView);

extern ModelGLTF Model_Nailgun;
// TODO (Kevin): instance drawing of nail projectiles
extern ModelGLTF Model_Nail;
