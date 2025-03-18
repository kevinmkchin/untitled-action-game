#include "weapons.h"

// external
ModelGLTF Model_Nailgun;

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
        Mix_VolumeChunk(Assets.Sfx_Shoot0, 16 
            + RandomInt(-1, 1)); // Volume variation
        Mix_PlayChannel(ChannelIndex++%4, Assets.Sfx_Shoot0, 0);
        State->Cooldown = 0.09f;

        GunRecoil = 0.9f;

        SpawnProjectile();
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

void SpawnProjectile()
{
    // create projectile_t and a physics body (sphere probably) and add to simulation

    JPH::RefConst<JPH::SphereShape> JPHShape_Sphere4 = new JPH::SphereShape(ToJoltUnit(4));

    // BodyCreationSettings
    JPH::BodyCreationSettings ProjectileCreationSettings(JPHShape_Sphere4, JPH::RVec3::sZero(), 
        JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
    
    ProjectileCreationSettings.mPosition = ToJoltVector(vec3(0,100,0));
    ProjectileCreationSettings.mLinearVelocity = ToJoltVector(vec3(100,0,0));
    ProjectileCreationSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    // ProjectileCreationSettings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
    // ProjectileCreationSettings.mMassPropertiesOverride.mMass = 0.008f;
    // ProjectileCreationSettings.mGravityFactor = 0.001f;
    // https://github.com/jrouwe/JoltPhysics/discussions/1040

    JPH::BodyID ProjectileBodyId = Physics.BodyInterface->CreateAndAddBody(
        ProjectileCreationSettings, JPH::EActivation::Activate);

    projectile_t Projectile;
    Projectile.BodyId = ProjectileBodyId;

    LiveProjectiles.put(Projectile);

    // BodyInterface.SetLinearVelocity(ProjectileBodyId, Vec3(0.0f, -5.0f, 0.0f));
    // LogMessage("%d", Sphere->GetRefCount());
    // LogMessage("%ld", (void*)Sphere.GetPtr());
}

// The quality of projectiles is that a lot of them get created in bursts and they die quickly
// New ones are created often and live ones die quickly

void PrePhysicsUpdateProjectiles()
{
    // go through all live projectiles, set their physics velocities

}

void PostPhysicsUpdateProjectiles()
{
    // if still live, read back their position/orientation for rendering

}

