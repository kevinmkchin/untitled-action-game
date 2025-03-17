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


