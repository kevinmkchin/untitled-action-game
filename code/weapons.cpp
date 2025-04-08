#include "weapons.h"

// external
fixed_array<projectile_breed_t> ProjectilesData;
fixed_array<projectile_t> LiveProjectiles;
fixed_array<projectile_hit_info_t> ProjectileHitInfos;

// internal
static float GunRecoil = 0.f;


void TickWeapon(weapon_state_t *State, bool LMB, bool RMB)
{
    if (State->Cooldown > 0.f)
        State->Cooldown -= DeltaTime;
    if (State->MuzzleFlash.w > 0.f)
        State->MuzzleFlash.w -= DeltaTime;

    // switch on State->ActiveType

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
        vec3 NailgunTip = Cam->Position + Cam->Direction * 32.f
            + vec3(0.f,-8.f,0.f);
        SpawnProjectile(PROJECTILE_NAIL, NailgunTip, Cam->Direction, 
            Cam->Orientation, vec3(), vec3());

        State->MuzzleFlash.w = 0.080f;
        State->MuzzleFlash.x = NailgunTip.x;
        State->MuzzleFlash.y = NailgunTip.y;
        State->MuzzleFlash.z = NailgunTip.z;
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
}

void RenderWeapon(weapon_state_t *State, float *ProjFromView, float *WorldFromView)
{
    UseShader(Sha_Gun);
    glEnable(GL_DEPTH_TEST);
    GLBindMatrix4fv(Sha_Gun, "WorldFromView", 1, WorldFromView);
    GLBindMatrix4fv(Sha_Gun, "ProjFromView", 1, ProjFromView);

    GPUMeshIndexed m;
    GPUTexture t;
    mat4 GunOffsetAndScale = TranslationMatrix(0,-4,GunRecoil) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
    GLBindMatrix4fv(Sha_Gun, "ViewFromModel", 1, GunOffsetAndScale.ptr());
    m = Assets.ModelsTextured[MT_WPN_TYPE1].meshes[0];
    t = Assets.ModelsTextured[MT_WPN_TYPE1].color[0];
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, t.id);
    RenderGPUMeshIndexed(m);
    GunOffsetAndScale = TranslationMatrix(0,-4,GunRecoil) 
        * RotationMatrix(EulerToQuat(0,0,State->NailgunRotation)) 
        * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
    GLBindMatrix4fv(Sha_Gun, "ViewFromModel", 1, GunOffsetAndScale.ptr());
    m = Assets.ModelsTextured[MT_WPN_TYPE1].meshes[1];
    t = Assets.ModelsTextured[MT_WPN_TYPE1].color[1];
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, t.id);
    RenderGPUMeshIndexed(m);
}

void RenderProjectiles(game_state *GameState, const mat4 &ProjFromView, const mat4 &ViewFromWorld)
{
    UseShader(Sha_ModelTexturedLit);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(Sha_ModelTexturedLit, "MuzzleFlash", 
        GameState->Player.Weapon.MuzzleFlash.x, 
        GameState->Player.Weapon.MuzzleFlash.y, 
        GameState->Player.Weapon.MuzzleFlash.z, 
        GameState->Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_ModelTexturedLit, "ProjFromView", 1, ProjFromView.ptr());
    GLBindMatrix4fv(Sha_ModelTexturedLit, "ViewFromWorld", 1, ViewFromWorld.ptr());

    for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
    {
        projectile_t& P = LiveProjectiles[i];
        vec3 ProjectileRenderPos = FromJoltVector(Physics.BodyInterface->GetPosition(P.BodyId));
        quat ProjectileRenderRot = FromJoltQuat(Physics.BodyInterface->GetRotation(P.BodyId));
        // Putting out of world bound check here because we have position here
        if (Magnitude(ProjectileRenderPos) > WORLD_LIMIT_F)
        {
            KillProjectile(GameState, &P);
            continue;
        }

        ++GameState->DynamicInstances.length;        
        FillModelInstanceData(GameState,
            GameState->DynamicInstances.end()-1, ProjectileRenderPos,
            ProjectileRenderPos, ProjectileRenderRot, P.Type->TexturedModel);

        // BindUniformsForModelLighting(Sha_ModelTexturedLit, RuntimeMapInfo, ProjectileRenderPos);
        // mat4 WorldFromModel = TranslationMatrix(ProjectileRenderPos) 
        //                     * RotationMatrix(ProjectileRenderRot)
        //                     * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
        // GLBindMatrix4fv(Sha_ModelTexturedLit, "WorldFromModel", 1, WorldFromModel.ptr());

        // RenderModelGLTF(*());
    }
}

void SetupProjectilesDataAndAllocateMemory()
{
    ProjectilesData = fixed_array<projectile_breed_t>(PROJECTILE_TYPE_COUNT, MemoryType::Game);
    ProjectilesData.setlen(PROJECTILE_TYPE_COUNT);
    LiveProjectiles = fixed_array<projectile_t>(512, MemoryType::Game);
    ProjectileHitInfos = fixed_array<projectile_hit_info_t>(128, MemoryType::Game);

    // Jolt is annoying I can't really not allocate these on the heap...
    JPH::Shape *PhysicsShape_Sphere4 = new JPH::SphereShape(ToJoltUnit(4));
    JPH::Shape *PhysicsShape_Box8 = new JPH::BoxShape(ToJoltVector(vec3(4.f,4.f,4.f)));

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

void PrePhysicsUpdateProjectiles()
{
    for (size_t IndexAtPrePhysicsTick = 0; IndexAtPrePhysicsTick < LiveProjectiles.lenu(); ++IndexAtPrePhysicsTick)
    {
        projectile_t& Projectile = LiveProjectiles[IndexAtPrePhysicsTick];
        if (!Projectile.IsDead && Projectile.Type->ObjectLayer == Layers::PROJECTILE)
        {
            u64 UserData = IndexAtPrePhysicsTick;
            Physics.BodyInterface->SetUserData(Projectile.BodyId, UserData);
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

        if (LiveProjectiles[ProjectileIdx].IsDead)
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
            projectile_breed_t *PrjInfo = LiveProjectiles[ProjectileIdx].Type;

            if (PrjInfo->BulletDamage > 0.f)
            {
                u32 EnemyIndex = (u32)Info.OtherBody->GetUserData();
                HurtEnemy(GameState, EnemyIndex, PrjInfo->BulletDamage);

                particle_emitter BloodBurst;
                BloodBurst.WorldP = Info.HitP;
                BloodBurst.PSpread = vec3(0.f,0.f,0.f);
                BloodBurst.dP = Info.HitN * 128.f + vec3(0.f,70.f,0.f);
                BloodBurst.dPSpread = BloodBurst.dP*(0.3f);
                BloodBurst.ddP = vec3(0.f,FromJoltUnit(-9.8f),0.f);
                BloodBurst.Color = vec4(1,1,1,1.4f);
                BloodBurst.ColorSpread = vec4(0,0,0,0.1f);
                BloodBurst.dColor = vec4(0,0,0,-1.35f);
                BloodBurst.Timer = 0.f;
                BloodBurst.ParticleLifeTimer = 2.f;
                g_GameState.BloodParticles.Emitters.put(BloodBurst);

                BloodBurst.WorldP = Info.HitP;
                BloodBurst.PSpread = vec3(0.f,0.f,0.f);
                BloodBurst.dP = Info.HitN * 6.f + vec3(0.f,96.f,0.f);
                BloodBurst.dPSpread = BloodBurst.dP*(0.5f);
                BloodBurst.ddP = vec3(0.f,FromJoltUnit(-9.8f),0.f);
                BloodBurst.Color = vec4(1,1,1,1.4f);
                BloodBurst.ColorSpread = vec4(0,0,0,0.1f);
                BloodBurst.dColor = vec4(0,0,0,-1.35f);
                BloodBurst.Timer = 0.f;
                BloodBurst.ParticleLifeTimer = 2.f;
                g_GameState.BloodParticles.Emitters.put(BloodBurst);
            }

            // Rocket should do splash damage collider check

            KillProjectile(GameState, &LiveProjectiles[ProjectileIdx]);
        }
        else if (SecondBodyLayer == Layers::STATIC)
        {
            if (SOUNDRNG.NextInt(0,2) < 1)
            {
                Mix_Chunk *RicochetSnd = Assets.Sfx_Ricochet[SOUNDRNG.NextInt(0,2)];
                Mix_VolumeChunk(RicochetSnd, 24 + SOUNDRNG.NextInt(-2, 2));
                Mix_PlayChannel(-1, RicochetSnd, 0);
            }

            // Rocket should do splash damage collider check

            KillProjectile(GameState, &LiveProjectiles[ProjectileIdx]);
        }
    }

    ProjectileHitInfos.setlen(0);
}

void PostPhysicsUpdateProjectiles(game_state *GameState)
{
    ProcessProjectileHitInfos(GameState);
    RemoveDeadProjectiles();
}

