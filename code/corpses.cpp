#include "corpses.h"


static TripleBufferedSSBO CorpsesSSBO;

void Corpses_AcquireGPUResources()
{
    CorpsesSSBO.Init(sizeof(model_instance_data_t) * MaxCorpsesInLevel);
}

void Corpses_ReleaseGPUResources()
{
    CorpsesSSBO.Destroy();
}

void FillModelInstanceData(model_instance_data_t *InstanceData, vec3 ModelCentroid, 
    vec3 RenderPosition, quat RenderRotation, ModelGLTF *InstanceModel)
{
    size_t LightCacheIndex = RuntimeMapInfo.LightCacheVolume->IndexByPosition(ModelCentroid);
    lc_light_indices_t LightIndices = RuntimeMapInfo.LightCacheVolume->SignificantLightIndices[LightCacheIndex];
    int DoSunLight = 0;
    c_array<vec4, 4> PointLightsPos;
    c_array<float, 4> PointLightsAttLin;
    c_array<float, 4> PointLightsAttQuad;
    for (int i = 0; i < 4; ++i)
    {
        short LightIndex = *(((short*)&LightIndices) + i);
        if (LightIndex < 0)
            continue;
        if (LightIndex == SUNLIGHTINDEX)
        {
            DoSunLight = 1;
            continue;
        }

        const static_point_light_t &PointLight = RuntimeMapInfo.PointLights[LightIndex];
        PointLightsPos.put(vec4(PointLight.Pos.x, PointLight.Pos.y, PointLight.Pos.z, 1.f));
        PointLightsAttLin.put(PointLight.AttenuationLinear);
        PointLightsAttQuad.put(PointLight.AttenuationQuadratic);
    }
    InstanceData->WorldFromModel = TranslationMatrix(RenderPosition) * RotationMatrix(RenderRotation) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
    memcpy(InstanceData->PointLightsPos, PointLightsPos.data, sizeof(vec4)*4);
    memcpy(InstanceData->AmbientCube, &RuntimeMapInfo.LightCacheVolume->AmbientCubes[LightCacheIndex], sizeof(float)*6);
    InstanceData->DoSunLight = DoSunLight;
    InstanceData->PointLightsCount = (i32)PointLightsPos.count;
    memcpy(InstanceData->PointLightsAttLin, PointLightsAttLin.data, sizeof(float)*4);
    memcpy(InstanceData->PointLightsAttQuad, PointLightsAttQuad.data, sizeof(float)*4);
    InstanceData->CorpseModel = InstanceModel;
}

void SortAndDrawCorpses(map_info_t &RuntimeMap, 
    fixed_array<model_instance_data_t> &Corpses,
    const mat4 &ProjFromView, const mat4 &ViewFromWorld)
{
    if (Corpses.lenu() == 0)
        return;

    // NOTE(Kevin): We only ever insert into this array so it should remain sorted
    //              with only the new entries from this frame requiring sorting.
    std::sort(Corpses.begin(), Corpses.end());

    UseShader(Sha_ModelInstancedLit);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(Sha_ModelInstancedLit, "MuzzleFlash",
        Player.Weapon.MuzzleFlash.x, 
        Player.Weapon.MuzzleFlash.y, 
        Player.Weapon.MuzzleFlash.z, 
        Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_ModelInstancedLit, "ProjFromView", 1, ProjFromView.ptr());
    GLBindMatrix4fv(Sha_ModelInstancedLit, "ViewFromWorld", 1, ViewFromWorld.ptr());
    GLBind3f(Sha_ModelInstancedLit, "DirectionToSun", RuntimeMap.DirectionToSun.x,
        RuntimeMap.DirectionToSun.y, RuntimeMap.DirectionToSun.z);

    CorpsesSSBO.BeginFrame();

    auto [MappedPtr, GPUOffset] = CorpsesSSBO.Alloc();
    memcpy(MappedPtr, Corpses.data, Corpses.lenu() * sizeof(model_instance_data_t));

    CorpsesSSBO.Bind(0);

    size_t BaseInstanceIndex = GPUOffset / sizeof(model_instance_data_t);

    u16 CurrentModelId = Corpses[0].CorpseModel->MT_ID;
    u32 CurrentModelCount = 0;
    u32 CurrentOffsetFromBase = 0;
    for (u32 i = 0; i <= Corpses.lenu(); ++i)
    {
        if (i == Corpses.lenu() || Corpses[i].CorpseModel->MT_ID != CurrentModelId)
        {
            // flush
            GLBind1i(Sha_ModelInstancedLit, "BaseInstanceIndex", 
                (GLint)BaseInstanceIndex + CurrentOffsetFromBase);
            DrawModelInstanced(Assets.ModelsTextured[CurrentModelId], CurrentModelCount);

            if (i == Corpses.lenu())
                break;

            CurrentOffsetFromBase += CurrentModelCount;
            CurrentModelId = Corpses[i].CorpseModel->MT_ID;
            CurrentModelCount = 0;
        }
        ++CurrentModelCount;
    }

    CorpsesSSBO.EndFrame();
}


