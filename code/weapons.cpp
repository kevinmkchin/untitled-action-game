#include "weapons.h"
#include "particles.h"
#include "game_assets.h"
#include "enemy.h"
#include "instanced.h"
#include "game.h"


// external
fixed_array<projectile_breed_t> ProjectilesData;
fixed_array<projectile_t> LiveProjectiles;
fixed_array<projectile_hit_info_t> ProjectileHitInfos;
JPH::Shape *PhysicsShape_Sphere1 = nullptr;
JPH::Shape *PhysicsShape_Sphere4 = nullptr;
JPH::Shape *PhysicsShape_Sphere8 = nullptr;
JPH::Shape *PhysicsShape_Box8 = nullptr;

// internal
static float GunRecoil = 0.f;
static random_series SOUNDRNG; // move to game_state

void TickWeapon(weapon_state_t *State, bool LMB, bool RMB)
{
    if (State->Cooldown > 0.f)
        State->Cooldown -= DeltaTime;
    if (State->MuzzleFlash.w > 0.f)
        State->MuzzleFlash.w -= DeltaTime;

    switch (State->ActiveType)
    {
        case NAILGUN:
        {
            GunRecoil = GM_max(0.f, GunRecoil-9.0f*DeltaTime);
            if (LMB && State->Cooldown <= 0.f)
            {
                // shoot
                static int ChannelIndex = 0;
                Mix_VolumeChunk(Assets.Sfx_Shoot0, 24 + SOUNDRNG.NextInt(-2, 2)); // Volume variation
                Mix_PlayChannel(ChannelIndex++%3, Assets.Sfx_Shoot0, 0);
                // State->Cooldown = RMB ? 0.080f : 0.150f;
                State->Cooldown = 0.080f;

                GunRecoil = 0.9f;

                camera_t *Cam = &(State->Owner->PlayerCam);
                Cam->ApplyKnockback(0.034f, 0.12f);
                vec3 GunTip = Cam->Position + Cam->Direction * 32.f
                    + vec3(0.f,-5.f,0.f);
                SpawnProjectile(PROJECTILE_NAIL, GunTip, Cam->Direction, 
                    Cam->Orientation, vec3(), vec3());

                State->MuzzleFlash.w = 0.080f;
                State->MuzzleFlash.x = GunTip.x;
                State->MuzzleFlash.y = GunTip.y;
                State->MuzzleFlash.z = GunTip.z;
            }
            if (LMB || RMB)
            {
                State->NailgunRotationVelocity = State->NailgunRotationMaxVelocity;
            }

            State->NailgunRotation += State->NailgunRotationVelocity * DeltaTime;
            State->NailgunRotationVelocity = fmax(0.f, State->NailgunRotationVelocity + State->NailgunRotationAcceleration * DeltaTime);
            if (State->NailgunRotationVelocity > 0.1f && State->NailgunRotationVelocity <= 3.f)
            {
                State->NailgunRotationVelocity = fmax(3.f, State->NailgunRotationVelocity);
                if (abs(fmodf(State->NailgunRotation, GM_PI) - GM_HALFPI) < 0.01f)
                {
                    State->NailgunRotationVelocity = 0.f;
                }
            }
            break;
        }
        case ROCKETLAUNCHER:
        {
            GunRecoil = GM_max(0.f, GunRecoil-9.0f*DeltaTime);
            if (LMB && State->Cooldown <= 0.f)
            {
                static int ChannelIndex = 0;
                Mix_VolumeChunk(Assets.Sfx_ShootRocket, 24 + SOUNDRNG.NextInt(-2, 2));
                Mix_PlayChannel(ChannelIndex++%3, Assets.Sfx_ShootRocket, 0);
                State->Cooldown = 0.90f;

                GunRecoil = 4.9f;

                camera_t *Cam = &(State->Owner->PlayerCam);
                Cam->ApplyKnockback(0.050f, 0.12f);
                vec3 GunTip = Cam->Position + Cam->Direction * 32.f
                    + vec3(0.f,0.f,0.f);
                SpawnProjectile(PROJECTILE_ROCKET_0, GunTip, Cam->Direction, 
                    Cam->Orientation, vec3(), vec3());

                State->MuzzleFlash.w = 0.080f;
                State->MuzzleFlash.x = GunTip.x;
                State->MuzzleFlash.y = GunTip.y;
                State->MuzzleFlash.z = GunTip.z;
            }
            break;
        }
    }
}

void DrawWeaponModel(game_state *GameState)
{
    // Request draw weapon model
    // This function is to handle drawing weapon models because they
    // may require special handling e.g. rotating barrel

    weapon_state_t &WeaponState = GameState->Player.Weapon;
    camera_t &PlayerCam = GameState->Player.PlayerCam;

    vec4 World_GunForward = vec4(PlayerCam.ViewForward, 0);
    vec4 World_GunUp = vec4(PlayerCam.ViewUp, 0);
    vec4 World_GunRight = vec4(Normalize(Cross(PlayerCam.ViewForward, PlayerCam.ViewUp)), 0);
    vec4 World_GunPosition = vec4(PlayerCam.ViewPosition, 1);

    vec4 World_ModifiedGunPosition = World_GunPosition;
    // TODO(Kevin): read weapon model information from breed/type pattern
    //              and move WorldFromGun matrix assembly here

    textured_lit_drawinfo DrawInfo;
    FillModelInstanceData(GameState,
        &DrawInfo.RenderingInfo,
        World_GunPosition.xyz,
        vec3(),
        quat(),
        nullptr);

    switch (WeaponState.ActiveType)
    {
        case ROCKETLAUNCHER:
        {
            World_ModifiedGunPosition += World_GunForward * -GunRecoil;
            World_ModifiedGunPosition += World_GunUp * -3.8f;
            World_ModifiedGunPosition += World_GunRight * 1.8f;
            mat4 WorldFromGun = mat4(
                World_GunForward,
                World_GunUp,
                World_GunRight,
                World_ModifiedGunPosition);

            DrawInfo.RenderingInfo.WorldFromModel = WorldFromGun
                * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
            DrawInfo.M = Assets.ModelsTextured[MT_WPN_ROCKETLAUNCHER].meshes[0];
            DrawInfo.T = Assets.ModelsTextured[MT_WPN_ROCKETLAUNCHER].color[0];
            GameState->TexturedLitRenderData.put(DrawInfo);
        } break;
        case NAILGUN:
        {
            World_ModifiedGunPosition += World_GunForward * -GunRecoil;
            World_ModifiedGunPosition += World_GunUp * -4.0f;
            World_ModifiedGunPosition += World_GunRight * 1.8f;
            mat4 WorldFromGun = mat4(
                World_GunForward,
                World_GunUp,
                World_GunRight,
                World_ModifiedGunPosition);

            DrawInfo.RenderingInfo.WorldFromModel = WorldFromGun
                * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
            DrawInfo.M = Assets.ModelsTextured[MT_WPN_TYPE1].meshes[0];
            DrawInfo.T = Assets.ModelsTextured[MT_WPN_TYPE1].color[0];
            GameState->TexturedLitRenderData.put(DrawInfo);

            DrawInfo.RenderingInfo.WorldFromModel = WorldFromGun
                * RotationMatrix(EulerToQuat(WeaponState.NailgunRotation,0,0))
                * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
            DrawInfo.M = Assets.ModelsTextured[MT_WPN_TYPE1].meshes[1];
            DrawInfo.T = Assets.ModelsTextured[MT_WPN_TYPE1].color[1];
            GameState->TexturedLitRenderData.put(DrawInfo);
        } break;
    }
}

void InstanceProjectilesForDrawing(game_state *GameState)
{
    for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
    {
        projectile_t& P = LiveProjectiles[i];
        vec3 ProjectileRenderPos = FromJoltVector(Physics.BodyInterface->GetPosition(P.BodyId));
        quat ProjectileRenderRot = FromJoltQuat(Physics.BodyInterface->GetRotation(P.BodyId));

        if (Magnitude(ProjectileRenderPos) > WORLD_LIMIT_F)
        {
            // Putting out of world bound check here because we have position here
            KillProjectile(GameState, &P);
            continue;
        }

        ++GameState->DynamicInstances.length;
        FillModelInstanceData(GameState,
            GameState->DynamicInstances.end()-1, ProjectileRenderPos,
            ProjectileRenderPos, ProjectileRenderRot, P.Type->TexturedModel);
    }
}

void SetupProjectilesDataAndAllocateMemory()
{
    ProjectilesData = fixed_array<projectile_breed_t>(PROJECTILE_TYPE_COUNT, MemoryType::Game);
    ProjectilesData.setlen(PROJECTILE_TYPE_COUNT);
    LiveProjectiles = fixed_array<projectile_t>(512, MemoryType::Game);
    ProjectileHitInfos = fixed_array<projectile_hit_info_t>(128, MemoryType::Game);

    // Jolt is annoying I can't really not allocate these on the heap...
    PhysicsShape_Sphere1 = new JPH::SphereShape(ToJoltUnit(1));
    PhysicsShape_Sphere4 = new JPH::SphereShape(ToJoltUnit(4));
    PhysicsShape_Sphere8 = new JPH::SphereShape(ToJoltUnit(8));
    PhysicsShape_Box8 = new JPH::BoxShape(ToJoltVector(vec3(4.f,4.f,4.f)));

    ProjectilesData[PROJECTILE_NAIL].BulletDamage = 16.f;
    ProjectilesData[PROJECTILE_NAIL].LinearVelocity = 1650.f;
    ProjectilesData[PROJECTILE_NAIL].TexturedModel = &Assets.ModelsTextured[MT_PRJ_NAIL];
    ProjectilesData[PROJECTILE_NAIL].ObjectLayer = Layers::PROJECTILE;
    ProjectilesData[PROJECTILE_NAIL].MotionQuality = JPH::EMotionQuality::LinearCast;
    ProjectilesData[PROJECTILE_NAIL].PhysicsShape = PhysicsShape_Sphere4;
    ProjectilesData[PROJECTILE_NAIL].Mass_kg = 0.008f;
    ProjectilesData[PROJECTILE_NAIL].SetFriction = false;
    ProjectilesData[PROJECTILE_NAIL].Friction = -1.f;
    ProjectilesData[PROJECTILE_NAIL].GravityFactor = 0.f;
    ProjectilesData[PROJECTILE_NAIL].KillAfterTimer = 8.f;
    ProjectilesData[PROJECTILE_NAIL].KillAfterSlowingDown = false;
    ProjectilesData[PROJECTILE_NAIL].RemainAfterDead = false;
    ProjectilesData[PROJECTILE_NAIL].BlowUpEnemies = false;
    ProjectilesData[PROJECTILE_NAIL].DoSplashDamageOnDead = false;

    ProjectilesData[PROJECTILE_ROCKET_0].BulletDamage = 50.f;
    ProjectilesData[PROJECTILE_ROCKET_0].LinearVelocity = 650.f;
    ProjectilesData[PROJECTILE_ROCKET_0].TexturedModel = &Assets.ModelsTextured[MT_PRJ_ROCKET];
    ProjectilesData[PROJECTILE_ROCKET_0].ObjectLayer = Layers::PROJECTILE;
    ProjectilesData[PROJECTILE_ROCKET_0].MotionQuality = JPH::EMotionQuality::LinearCast;
    ProjectilesData[PROJECTILE_ROCKET_0].PhysicsShape = PhysicsShape_Sphere4;
    ProjectilesData[PROJECTILE_ROCKET_0].Mass_kg = 0.08f;
    ProjectilesData[PROJECTILE_ROCKET_0].SetFriction = false;
    ProjectilesData[PROJECTILE_ROCKET_0].Friction = -1.f;
    ProjectilesData[PROJECTILE_ROCKET_0].GravityFactor = 0.f;
    ProjectilesData[PROJECTILE_ROCKET_0].KillAfterTimer = 10.f;
    ProjectilesData[PROJECTILE_ROCKET_0].KillAfterSlowingDown = false;
    ProjectilesData[PROJECTILE_ROCKET_0].RemainAfterDead = false;
    ProjectilesData[PROJECTILE_ROCKET_0].BlowUpEnemies = true;
    ProjectilesData[PROJECTILE_ROCKET_0].DoSplashDamageOnDead = true;
    ProjectilesData[PROJECTILE_ROCKET_0].SplashDamageRadius = 96.f;
    ProjectilesData[PROJECTILE_ROCKET_0].SplashDamageBase = 60.f;

    ProjectilesData[PROJECTILE_GENERIC_GIB_0].BulletDamage = 0.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].LinearVelocity = 0.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].TexturedModel = &Assets.ModelsTextured[MT_GENERIC_GIB_0];
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].ObjectLayer = Layers::GIB;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].MotionQuality = JPH::EMotionQuality::Discrete;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].PhysicsShape = PhysicsShape_Box8;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].Mass_kg = 3.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].SetFriction = true;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].Friction = 100.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].GravityFactor = 1.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].KillAfterTimer = 10.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].KillAfterSlowingDown = true;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].RemainAfterDead = true;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].BlowUpEnemies = false;
    ProjectilesData[PROJECTILE_GENERIC_GIB_0].DoSplashDamageOnDead = false;

    ProjectilesData[PROJECTILE_GENERIC_GIB_1].BulletDamage = 0.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].LinearVelocity = 0.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].TexturedModel = &Assets.ModelsTextured[MT_GENERIC_GIB_1];
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].ObjectLayer = Layers::GIB;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].MotionQuality = JPH::EMotionQuality::Discrete;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].PhysicsShape = PhysicsShape_Box8;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].Mass_kg = 3.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].SetFriction = true;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].Friction = 100.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].GravityFactor = 1.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].KillAfterTimer = 10.f;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].KillAfterSlowingDown = true;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].RemainAfterDead = true;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].BlowUpEnemies = false;
    ProjectilesData[PROJECTILE_GENERIC_GIB_1].DoSplashDamageOnDead = false;
}

// The quality of projectiles is that a lot of them get created in bursts and they die quickly
// New ones are created often and live ones die quickly
void SpawnProjectile(projectile_type_enum Type, vec3 Pos, vec3 Dir, 
    quat Orient, vec3 Impulse, vec3 AngularImpulse)
{
    projectile_breed_t *PrjInfo = &ProjectilesData[Type];

    JPH::BodyCreationSettings ProjectileCreationSettings(
        PrjInfo->PhysicsShape,
        JPH::RVec3::sZero(),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        PrjInfo->ObjectLayer);

    ProjectileCreationSettings.mPosition = ToJoltVector(Pos);
    ProjectileCreationSettings.mRotation = ToJoltQuat(Orient);
    ProjectileCreationSettings.mLinearVelocity = ToJoltVector(Dir * PrjInfo->LinearVelocity);
    ProjectileCreationSettings.mMotionQuality = PrjInfo->MotionQuality;
    if (PrjInfo->Mass_kg >= 0.f)
    {
        ProjectileCreationSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        ProjectileCreationSettings.mMassPropertiesOverride.mMass = PrjInfo->Mass_kg;
    }
    if (PrjInfo->SetFriction)
    {
        ProjectileCreationSettings.mFriction = PrjInfo->Friction;
    }
    ProjectileCreationSettings.mGravityFactor = PrjInfo->GravityFactor;

    // this body gets new-ed...
    JPH::BodyID ProjectileBodyId = Physics.BodyInterface->CreateAndAddBody(
        ProjectileCreationSettings, JPH::EActivation::Activate);

    if (Impulse != vec3())
        Physics.BodyInterface->AddImpulse(ProjectileBodyId, ToJoltVector(Impulse));
    if (AngularImpulse != vec3())
        Physics.BodyInterface->AddAngularImpulse(ProjectileBodyId, ToJoltVector(AngularImpulse));

    projectile_t Projectile;
    Projectile.Type = PrjInfo;
    Projectile.EType = Type;
    Projectile.IsDead = false;
    Projectile.BodyId = ProjectileBodyId;
    Projectile.BeenAliveTimer = 0.f;

    LiveProjectiles.put(Projectile);
}

void KillProjectile(game_state *GameState, projectile_t *ProjectileToKill)
{
    ProjectileToKill->IsDead = true;

    if (ProjectileToKill->Type->RemainAfterDead)
    {
        vec3 CorpsePosition = FromJoltVector(Physics.BodyInterface->GetPosition(ProjectileToKill->BodyId));
        quat CorpseRotation = FromJoltQuat(Physics.BodyInterface->GetRotation(ProjectileToKill->BodyId));
        ModelGLTF *CorpseModel = ProjectileToKill->Type->TexturedModel;
        ++GameState->StaticInstances.length;
        FillModelInstanceData(GameState,
            &GameState->StaticInstances[GameState->StaticInstances.length-1], 
            CorpsePosition, CorpsePosition, CorpseRotation, CorpseModel);
    }

    if (ProjectileToKill->Type->DoSplashDamageOnDead)
    {
        class splash_damage_collector : public JPH::CollideShapeCollector
        {
        public:
            // JPH_OVERRIDE_NEW_DELETE

            void AddHit(const JPH::CollideShapeResult &Result) override
            {
                JPH::BodyID bodyID = Result.mBodyID2;

                JPH::Vec3 ResolutionDir = Result.mPenetrationAxis.Normalized();
                float ResolutionDepth = Result.mPenetrationDepth;
                // Scale damage by hit distance
                // I like curve of y = (x)^(1/3)
                float DistFromCenter10 = FromJoltUnit(ResolutionDepth)/SplashRadius; // 1 if close, 0 if out of radius
                float DamageScale = powf(DistFromCenter10, 0.334f);
                JPH::Vec3 KnockbackVector = DamageScale*(ResolutionDir*240.f+JPH::Vec3(0,200,0));
                Physics.BodyInterface->AddImpulse(bodyID, KnockbackVector);
                u32 EnemyIndex = (u32)Physics.BodyInterface->GetUserData(bodyID);
                if (EnemyIndex != BAD_UINDEX)
                {
                    HurtEnemy(GameState, EnemyIndex, BaseDamage*DamageScale, BlowUpEnemies);
                }
            }

            game_state *GameState = nullptr;
            float BaseDamage = 0.f;
            float SplashRadius = 0.f;
            bool BlowUpEnemies = false;
        };

        splash_damage_collector SplashDamageCollector;
        SplashDamageCollector.GameState = GameState;
        SplashDamageCollector.BaseDamage = ProjectileToKill->Type->SplashDamageBase;
        SplashDamageCollector.SplashRadius = ProjectileToKill->Type->SplashDamageRadius;
        SplashDamageCollector.BlowUpEnemies = ProjectileToKill->Type->BlowUpEnemies;

        JPH::Vec3 ProjectilePosSI = Physics.BodyInterface->GetPosition(ProjectileToKill->BodyId);
        Physics.PhysicsSystem->GetNarrowPhaseQuery().CollideShape(
            // TODO access these shapes through some interface
            PhysicsShape_Sphere1,
            JPH::Vec3(
                ProjectileToKill->Type->SplashDamageRadius,
                ProjectileToKill->Type->SplashDamageRadius,
                ProjectileToKill->Type->SplashDamageRadius),
            JPH::Mat44::sTranslation(ProjectilePosSI),
            { },
            ProjectilePosSI, // hit results returned as offset from this
            SplashDamageCollector,
            JPH::SpecifiedBroadPhaseLayerFilter(BroadPhaseLayers::MOVING),
            JPH::SpecifiedObjectLayerFilter(Layers::ENEMY),
            { },
            { });

        particle_emitter Explosion;
        Explosion.WorldP = FromJoltVector(ProjectilePosSI);
        Explosion.PSpread = vec3(0.f,0.f,0.f);
        Explosion.dP = vec3();
        Explosion.dPSpread = vec3();
        Explosion.ddP = vec3();
        Explosion.Color = vec4(0.95f,0.90f,0.00f,0.7f);
        Explosion.ColorSpread = vec4(0,0,0,0.1f);
        Explosion.dColor = vec4(0,0,0,-0.05f);
        Explosion.HalfWidth = ProjectileToKill->Type->SplashDamageRadius*0.5f;
        Explosion.HalfWidthSpread = 0.3f;
        Explosion.dHalfWidth = ProjectileToKill->Type->SplashDamageRadius*5.f;
        Explosion.Timer = 0.f;
        Explosion.ParticleLifeTimer = 0.1f;
        GameState->BloodParticles.Emitters.put(Explosion);

        Mix_VolumeChunk(Assets.Sfx_ExplodeRocket, 24 + SOUNDRNG.NextInt(-2, 2));
        Mix_PlayChannel(-1, Assets.Sfx_ExplodeRocket, 0);
    }
}

void NonPhysicsUpdateProjectiles(game_state *GameState)
{
    for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
    {
        projectile_t& Projectile = LiveProjectiles[i];
        if (!Projectile.IsDead)
        {
            Projectile.BeenAliveTimer += DeltaTime;

            if (Projectile.Type->KillAfterTimer < Projectile.BeenAliveTimer)
            {
                KillProjectile(GameState, &Projectile);
            }
            else if (Projectile.Type->KillAfterSlowingDown && Projectile.BeenAliveTimer > 3.f)
            {
                vec3 LinVel = FromJoltVector(Physics.BodyInterface->GetLinearVelocity(Projectile.BodyId));
                if (Magnitude(LinVel) < 32.f)
                    KillProjectile(GameState, &Projectile);
            }
        }
    }
}

void PrePhysicsUpdateProjectiles(game_state *GameState)
{
    for (size_t IndexAtPrePhysicsTick = 0; 
        IndexAtPrePhysicsTick < LiveProjectiles.lenu(); 
        ++IndexAtPrePhysicsTick)
    {
        projectile_t& Projectile = LiveProjectiles[IndexAtPrePhysicsTick];

        if (!Projectile.IsDead && Projectile.Type->ObjectLayer == Layers::PROJECTILE)
        {
            u64 UserData = IndexAtPrePhysicsTick;
            Physics.BodyInterface->SetUserData(Projectile.BodyId, UserData);
        }

        if (!Projectile.IsDead &&
            PROJECTILE_GIBS_START < Projectile.EType && Projectile.EType < PROJECTILE_GIBS_END)
        {
            vec3 GibP = FromJoltVector(Physics.BodyInterface->GetPosition(Projectile.BodyId));
            particle_emitter GibTrail;
            GibTrail.WorldP = GibP;
            GibTrail.PSpread = vec3();
            GibTrail.dP = vec3();
            GibTrail.dPSpread = vec3();
            GibTrail.ddP = vec3(0.f,FromJoltUnit(-4.5f),0.f);
            GibTrail.Color = vec4(0.3f,0.02f,0.02f,1.4f);
            GibTrail.ColorSpread = vec4(0,0,0,0.1f);
            GibTrail.dColor = vec4(0,0,0,-1.8f);
            GibTrail.HalfWidth = 3.f;
            GibTrail.HalfWidthSpread = 0.0f;
            GibTrail.dHalfWidth = -6.f;
            GibTrail.Timer = 0.0f;
            GibTrail.ParticleLifeTimer = 2.0f;
            GameState->BloodParticles.Emitters.put(GibTrail);
        }
    }
}

static void RemoveDeadProjectiles()
{
    JPH::BodyID ProjectileBodyIdsToRemove[32];
    int BodyIdsToRemoveCount = 0;

    for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
    {
        projectile_t& Projectile = LiveProjectiles[i];
        if (Projectile.IsDead)
        {
            ProjectileBodyIdsToRemove[BodyIdsToRemoveCount] = Projectile.BodyId;
            ++BodyIdsToRemoveCount;

            LiveProjectiles.delswap((int)i);
            --i;
        }
    }

    if (BodyIdsToRemoveCount > 0)
    {
        Physics.BodyInterface->RemoveBodies(ProjectileBodyIdsToRemove, BodyIdsToRemoveCount);
        Physics.BodyInterface->DestroyBodies(ProjectileBodyIdsToRemove, BodyIdsToRemoveCount);
    }
}

static void ProcessProjectileHitInfos(game_state *GameState)
{
    // NOTE(Kevin): Physics threads could still be running as we process this.
    //              Any data that can be mutated from ContactListener must be
    //              protected!
    std::lock_guard Lock(Physics.ContactListener.ProjectileHitMutex);

    for (size_t i = 0; i < ProjectileHitInfos.lenu(); ++i)
    {
        projectile_hit_info_t Info = ProjectileHitInfos[i];
        if (Info.ProjBody->GetObjectLayer() != Layers::PROJECTILE)
        {
            LogWarning("projectile_hit_info_t's projectile body is not in Layers::PROJECTILE!");
            if (Info.OtherBody->GetObjectLayer() == Layers::PROJECTILE)
                std::swap(Info.ProjBody, Info.OtherBody);
            else
                continue;
        }

        size_t ProjectileIdx = (size_t)Info.ProjBody->GetUserData();

        if (ProjectileIdx >= LiveProjectiles.lenu())
        {
            LogError("GAME RUNTIME ERROR: Bad user data from projectile body...something went wrong!");
            continue;
        }

        projectile_t *Projectile = &LiveProjectiles[ProjectileIdx];

        if (Projectile->IsDead)
        {
            // the projectile is already dead/used.
            // NOTE(Kevin): sometimes I get duplicate projectile hit infos.
            //              should be okay to ignore.
            // NOTE(Kevin): I am using a mutex lock, but still getting dupes...
            continue;
        }

        JPH::ObjectLayer SecondBodyLayer = Info.OtherBody->GetObjectLayer();

        if (SecondBodyLayer == Layers::ENEMY)
        {
            if (Projectile->Type->BulletDamage > 0.f)
            {
                u32 EnemyIndex = (u32)Info.OtherBody->GetUserData();
                HurtEnemy(GameState, EnemyIndex, Projectile->Type->BulletDamage, 
                    Projectile->Type->BlowUpEnemies);

                particle_emitter BloodBurst;
                BloodBurst.WorldP = Info.HitP;
                BloodBurst.PSpread = vec3(0.f,0.f,0.f);
                BloodBurst.dP = Info.HitN * 128.f + vec3(0.f,70.f,0.f);
                BloodBurst.dPSpread = BloodBurst.dP*(0.3f);
                BloodBurst.ddP = vec3(0.f,FromJoltUnit(-9.8f),0.f);
                BloodBurst.Color = vec4(0.3f,0.02f,0.02f,1.4f);
                BloodBurst.ColorSpread = vec4(0,0,0,0.1f);
                BloodBurst.dColor = vec4(0,0,0,-1.35f);
                BloodBurst.HalfWidth = 3.f;
                BloodBurst.HalfWidthSpread = 0.3f;
                BloodBurst.dHalfWidth = 0.f;
                BloodBurst.Timer = 0.f;
                BloodBurst.ParticleLifeTimer = 2.f;
                GameState->BloodParticles.Emitters.put(BloodBurst);

                BloodBurst.WorldP = Info.HitP;
                BloodBurst.PSpread = vec3(0.f,0.f,0.f);
                BloodBurst.dP = Info.HitN * 6.f + vec3(0.f,96.f,0.f);
                BloodBurst.dPSpread = BloodBurst.dP*(0.5f);
                BloodBurst.ddP = vec3(0.f,FromJoltUnit(-9.8f),0.f);
                BloodBurst.Color = vec4(0.3f,0.02f,0.02f,1.4f);
                BloodBurst.ColorSpread = vec4(0,0,0,0.1f);
                BloodBurst.dColor = vec4(0,0,0,-1.35f);
                BloodBurst.HalfWidth = 3.f;
                BloodBurst.HalfWidthSpread = 0.3f;
                BloodBurst.dHalfWidth = 0.f;
                BloodBurst.Timer = 0.f;
                BloodBurst.ParticleLifeTimer = 2.f;
                GameState->BloodParticles.Emitters.put(BloodBurst);
            }

            KillProjectile(GameState, Projectile);
        }
        else if (SecondBodyLayer == Layers::STATIC)
        {
            if (Projectile->EType == PROJECTILE_NAIL && SOUNDRNG.NextInt(0,2) < 1)
            {
                Mix_Chunk *RicochetSnd = Assets.Sfx_Ricochet[SOUNDRNG.NextInt(0,2)];
                Mix_VolumeChunk(RicochetSnd, 24 + SOUNDRNG.NextInt(-2, 2));
                Mix_PlayChannel(-1, RicochetSnd, 0);
            }

            KillProjectile(GameState, Projectile);
        }
    }

    ProjectileHitInfos.setlen(0);
}

void PostPhysicsUpdateProjectiles(game_state *GameState)
{
    ProcessProjectileHitInfos(GameState);
    RemoveDeadProjectiles();
}

