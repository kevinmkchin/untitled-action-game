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


enum projectile_type_enum : u32
{
    PROJECTILE_NAIL = 0,
    PROJECTILE_GENERIC_GIB_0 = 1,
    PROJECTILE_GENERIC_GIB_1 = 2,
    PROJECTILE_TYPE_COUNT = 3
};

struct projectile_breed_t
{
    float BulletDamage;
    float LinearVelocity;
    ModelGLTF *TexturedModel;
    JPH::ObjectLayer ObjectLayer;
    JPH::EMotionQuality MotionQuality;
    JPH::RefConst<JPH::Shape> PhysicsShape;
    float Mass_kg; // manually setting mass will affect density and therefore inertia
    bool SetFriction;
    float Friction;
    float GravityFactor;
    float KillAfterTimer;
    bool KillAfterSlowingDown;
    bool RemainAfterDead;
    // Death callback? e.g. gib adds itself as a "corpse" or rocket does splash damage?
};

struct projectile_t
{
    projectile_breed_t *Type = nullptr;
    bool IsDead = false;
    JPH::BodyID BodyId;

    float BeenAliveTimer = 0.f;
};

struct projectile_hit_info_t
{
    const JPH::Body *Body1;
    const JPH::Body *Body2;
    const JPH::ContactManifold *Manifold;
};

extern fixed_array<projectile_breed_t> ProjectilesData;
extern fixed_array<projectile_t> LiveProjectiles;
extern fixed_array<projectile_hit_info_t> ProjectileHitInfos;

void SetupProjectilesDataAndAllocateMemory(); // Called once at start of game
void SpawnProjectile(projectile_type_enum Type, vec3 Pos, vec3 Dir, 
    quat Orient, vec3 Impulse, vec3 AngularImpulse);
void KillProjectile(game_state *GameState, projectile_t *ProjectileToKill);
void NonPhysicsUpdateProjectiles(game_state *GameState);
void PrePhysicsUpdateProjectiles();
void PostPhysicsUpdateProjectiles(game_state *GameState);
void RenderProjectiles(game_state *GameState, const mat4 &ProjFromView, const mat4 &ViewFromWorld);
