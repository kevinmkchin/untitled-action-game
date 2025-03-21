#include "weapons.h"

// external
ModelGLTF Model_Nailgun;
ModelGLTF Model_Nail;

// internal
static float NailgunRotation = GM_HALFPI;
static float NailgunRotationVelocity = 0.f;
static constexpr float NailgunRotationMaxVelocity = 16.f;
static constexpr float NailgunRotationAcceleration = -16.f;

static float GunRecoil = 0.f;


void TickWeapon(weapon_state_t *State, bool LMB, bool RMB)
{
    State->Cooldown -= DeltaTime;

    // switch on State->ActiveType

    GunRecoil = GM_max(0.f, GunRecoil-9.0f*DeltaTime);
    if (LMB && State->Cooldown <= 0.f)
    {
        // shoot
        static int ChannelIndex = 0;
        Mix_VolumeChunk(Assets.Sfx_Shoot0, 32 + RandomInt(-2, 2)); // Volume variation
        Mix_PlayChannel(ChannelIndex++%3, Assets.Sfx_Shoot0, 0);
        State->Cooldown = 0.180f;
        // State->Cooldown = 0.080f;

        GunRecoil = 0.9f;

        camera_t *Cam = &(State->Owner->PlayerCam);
        Cam->ApplyKnockback(0.034f, 0.12f);
        vec3 NailgunTip = Cam->Position + Cam->Direction * 32.f
            + vec3(0.f,-8.f,0.f);
        SpawnProjectile(NailgunTip, Cam->Direction, Cam->Orientation);
    }
    if (LMB || RMB)
    {
        NailgunRotationVelocity = NailgunRotationMaxVelocity;
    }
}

void RenderWeapon(weapon_state_t *State, float *ProjFromView, float *WorldFromView)
{
    UseShader(GunShader);
    glEnable(GL_DEPTH_TEST);
    GLBindMatrix4fv(GunShader, "WorldFromView", 1, WorldFromView);
    GLBindMatrix4fv(GunShader, "ProjFromView", 1, ProjFromView);

    GPUMeshIndexed m;
    GPUTexture t;
    mat4 GunOffsetAndScale = TranslationMatrix(0,-4,GunRecoil) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
    GLBindMatrix4fv(GunShader, "ViewFromModel", 1, GunOffsetAndScale.ptr());
    m = Model_Nailgun.meshes[0];
    t = Model_Nailgun.color[0];
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, t.id);
    RenderGPUMeshIndexed(m);
    NailgunRotation += NailgunRotationVelocity * DeltaTime;
    NailgunRotationVelocity = GM_max(0.f, NailgunRotationVelocity + NailgunRotationAcceleration * DeltaTime);
    if (NailgunRotationVelocity > 0.1f && NailgunRotationVelocity <= 3.f)
    {
        NailgunRotationVelocity = GM_max(3.f, NailgunRotationVelocity);
        if (abs(fmodf(NailgunRotation, GM_PI) - GM_HALFPI) < 0.01f)
        {
            NailgunRotationVelocity = 0.f;
        }
    }
    GunOffsetAndScale = TranslationMatrix(0,-4,GunRecoil) * RotationMatrix(EulerToQuat(0,0,NailgunRotation)) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
    GLBindMatrix4fv(GunShader, "ViewFromModel", 1, GunOffsetAndScale.ptr());
    m = Model_Nailgun.meshes[1];
    t = Model_Nailgun.color[1];
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, t.id);
    RenderGPUMeshIndexed(m);
}

void RenderProjectiles(const mat4 &ProjFromView, const mat4 &WorldFromView)
{
    UseShader(GunShader);
    glEnable(GL_DEPTH_TEST);
    GLBindMatrix4fv(GunShader, "WorldFromView", 1, WorldFromView.ptr());
    GLBindMatrix4fv(GunShader, "ProjFromView", 1, ProjFromView.ptr());

    for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
    {
        projectile_t& P = LiveProjectiles[i];
        vec3 ProjectileRenderPos = FromJoltVector(Physics.BodyInterface->GetPosition(P.BodyId));
        quat ProjectileRenderRot = P.RenderOrientation;
        mat4 ViewFromModel = WorldFromView.GetInverse()
            * TranslationMatrix(ProjectileRenderPos) 
            * RotationMatrix(ProjectileRenderRot)
            * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
        GLBindMatrix4fv(GunShader, "ViewFromModel", 1, ViewFromModel.ptr());
        RenderModelGLTF(Model_Nail);
    }
}


// The quality of projectiles is that a lot of them get created in bursts and they die quickly
// New ones are created often and live ones die quickly
void SpawnProjectile(vec3 Pos, vec3 Dir, quat Orient)
{

    // create projectile_t and a physics body (sphere probably) and add to simulation

    JPH::RefConst<JPH::SphereShape> JPHShape_Sphere4 = new JPH::SphereShape(ToJoltUnit(4));

    // BodyCreationSettings
    JPH::BodyCreationSettings ProjectileCreationSettings(JPHShape_Sphere4, JPH::RVec3::sZero(), 
        JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::PROJECTILE);

    ProjectileCreationSettings.mPosition = ToJoltVector(Pos);
    ProjectileCreationSettings.mLinearVelocity = ToJoltVector(Dir * 1650.f);
    ProjectileCreationSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    // ProjectileCreationSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
    ProjectileCreationSettings.mMassPropertiesOverride.mMass = 0.008f;
    ProjectileCreationSettings.mGravityFactor = 0.f;
    // https://github.com/jrouwe/JoltPhysics/discussions/1040

    JPH::BodyID ProjectileBodyId = Physics.BodyInterface->CreateAndAddBody(
        ProjectileCreationSettings, JPH::EActivation::Activate);

    projectile_t Projectile;
    Projectile.BodyId = ProjectileBodyId;
    Projectile.RenderOrientation = Orient;

    LiveProjectiles.put(Projectile);

    // BodyInterface.SetLinearVelocity(ProjectileBodyId, Vec3(0.0f, -5.0f, 0.0f));
    // LogMessage("%d", Sphere->GetRefCount());
    // LogMessage("%ld", (void*)Sphere.GetPtr());
}


static void RemoveDeadProjectiles()
{
    JPH::BodyID ProjectileBodyIdsToRemove[32];
    int BodyIdsToRemoveCount = 0;

    for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
    {
        projectile_t& Projectile = LiveProjectiles[i];
        if (Projectile.Flags & ProjectileFlag_Dead)
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

void KillProjectile(int LiveProjectileIndex)
{
    projectile_t &ProjectileToKill = LiveProjectiles[LiveProjectileIndex];
    ProjectileToKill.Flags |= ProjectileFlag_Dead;
}

void PrePhysicsUpdateProjectiles()
{
    // go through all live projectiles, set their physics velocities?

}

static void ProcessProjectileHitInfos()
{
    for (size_t i = 0; i < ProjectileHitInfos.lenu(); ++i)
    {
        projectile_hit_info_t Info = ProjectileHitInfos[i];
        if (Info.Body1->GetObjectLayer() != Layers::PROJECTILE)
        {
            ASSERT(Info.Body2->GetObjectLayer() == Layers::PROJECTILE);
            std::swap(Info.Body1, Info.Body2);
        }

        JPH::BodyID ProjectileBodyId = Info.Body1->GetID();

        int ProjectileIdx = -1;

        for (size_t p = 0; p < LiveProjectiles.lenu(); ++p)
        {
            if (LiveProjectiles[p].BodyId == ProjectileBodyId)
            {
                ProjectileIdx = (int)p;
                break;
            }
        }

        if (ProjectileIdx < 0)
        {
            LogError("There is no live projectile with collided Body ID...something went wrong!");
            continue;
        }

        if (LiveProjectiles[ProjectileIdx].Flags & ProjectileFlag_Dead)
        {
            // the projectile is dead. already used.
            // NOTE(Kevin): sometimes I get duplicate projectile hit infos.
            //              should be okay to ignore.
            continue;
        }

        JPH::ObjectLayer SecondBodyLayer = Info.Body2->GetObjectLayer();

        if (SecondBodyLayer == Layers::ENEMY)
        {
            LogMessage("Direct hit on enemy");
            // TODO do direct hit damage to enemy
            KillProjectile(ProjectileIdx);
        }
        else if (SecondBodyLayer == Layers::STATIC)
        {
            LogMessage("Hit world geometry");

            if (RandomInt(0,2) < 1)
            {
                Mix_Chunk *RicochetSnd = Assets.Sfx_Ricochet[RandomInt(0,2)];
                Mix_VolumeChunk(RicochetSnd, 32 + RandomInt(-2, 2));
                Mix_PlayChannel(-1, RicochetSnd, 0);
            }

            KillProjectile(ProjectileIdx);
        }
    }

    ProjectileHitInfos.setlen(0);
}

void PostPhysicsUpdateProjectiles()
{
    ProcessProjectileHitInfos();
    RemoveDeadProjectiles();
}

