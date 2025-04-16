#pragma once

#include "common.h"
#include "mem.h"
#include "anim.h"
#include "physics.h"

enum weapon_types_t : u16
{
    NAILGUN,
    ROCKETLAUNCHER,
    WEAPON_TYPES_COUNT
};

struct weapon_state_t
{
    weapon_types_t ActiveType = ROCKETLAUNCHER;

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
void DrawWeaponModel(game_state *GameState);

enum projectile_type_enum : u16
{
    // Sublime Text Arithmetic command is so nice for numbering enums!
    PROJECTILE_INVALID       = 0,
    PROJECTILE_NAIL          = 1,
    PROJECTILE_GIBS_START    = 2,
    PROJECTILE_GENERIC_GIB_0 = 3,
    PROJECTILE_GENERIC_GIB_1 = 4,
    PROJECTILE_GIBS_END      = 5,
    PROJECTILE_ROCKET_0      = 6,
    PROJECTILE_TYPE_COUNT    = 7
    // perhaps modders can add more types with u16 values > TYPE_COUNT
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
    bool BlowUpEnemies = false;
    bool DoSplashDamageOnDead = false;
    float SplashDamageRadius;
    float SplashDamageBase;
};

struct projectile_t
{
    projectile_breed_t *Type = nullptr;
    projectile_type_enum EType = PROJECTILE_INVALID;
    bool IsDead = false;
    JPH::BodyID BodyId;

    float BeenAliveTimer = 0.f;
};

struct projectile_hit_info_t
{
    const JPH::Body *ProjBody;
    const JPH::Body *OtherBody;
    vec3 HitP;
    vec3 HitN; // Direction to move projectile to resolve collision
};

// TODO(Kevin): move these to GameState
extern fixed_array<projectile_breed_t> ProjectilesData;
extern fixed_array<projectile_t> LiveProjectiles;
extern fixed_array<projectile_hit_info_t> ProjectileHitInfos;
extern JPH::Shape *PhysicsShape_Sphere1;
extern JPH::Shape *PhysicsShape_Sphere4;
extern JPH::Shape *PhysicsShape_Sphere8;
extern JPH::Shape *PhysicsShape_Box8;

void SetupProjectilesDataAndAllocateMemory(); // Called once at start of game
void SpawnProjectile(projectile_type_enum Type, vec3 Pos, vec3 Dir, 
    quat Orient, vec3 Impulse, vec3 AngularImpulse);
void KillProjectile(game_state *GameState, projectile_t *ProjectileToKill);
void NonPhysicsUpdateProjectiles(game_state *GameState);
void PrePhysicsUpdateProjectiles(game_state *GameState);
void PostPhysicsUpdateProjectiles(game_state *GameState);
void InstanceProjectilesForDrawing(game_state *GameState);

