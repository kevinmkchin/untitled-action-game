#include "instanced.h"


static TripleBufferedSSBO InstanceDataSSBO;

void InstanceDrawing_AcquireGPUResources()
{
    InstanceDataSSBO.Init(sizeof(model_instance_data_t) * (MaxStaticInstances + MaxDynamicInstances));
}

void InstancedDrawing_ReleaseGPUResources()
{
    InstanceDataSSBO.Destroy();
}

void FillModelInstanceData(model_instance_data_t *InstanceData, vec3 ModelCentroid, 
    vec3 RenderPosition, quat RenderRotation, ModelGLTF *InstanceModel)
{
    size_t LightCacheIndex = GameState.LightCacheVolume->IndexByPosition(ModelCentroid);
    lc_light_indices_t LightIndices = GameState.LightCacheVolume->SignificantLightIndices[LightCacheIndex];
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

        const static_point_light_t &PointLight = GameState.PointLights[LightIndex];
        PointLightsPos.put(vec4(PointLight.Pos.x, PointLight.Pos.y, PointLight.Pos.z, 1.f));
        PointLightsAttLin.put(PointLight.AttenuationLinear);
        PointLightsAttQuad.put(PointLight.AttenuationQuadratic);
    }
    InstanceData->WorldFromModel = TranslationMatrix(RenderPosition) * RotationMatrix(RenderRotation) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
    memcpy(InstanceData->PointLightsPos, PointLightsPos.data, sizeof(vec4)*4);
    memcpy(InstanceData->AmbientCube, &GameState.LightCacheVolume->AmbientCubes[LightCacheIndex], sizeof(float)*6);
    InstanceData->DoSunLight = DoSunLight;
    InstanceData->PointLightsCount = (i32)PointLightsPos.count;
    memcpy(InstanceData->PointLightsAttLin, PointLightsAttLin.data, sizeof(float)*4);
    memcpy(InstanceData->PointLightsAttQuad, PointLightsAttQuad.data, sizeof(float)*4);
    InstanceData->CorpseModel = InstanceModel;
}

void SortAndDrawInstancedModels(game_state *RuntimeMap,
    fixed_array<model_instance_data_t> &StaticInstances,
    fixed_array<model_instance_data_t> &DynamicInstances,
    const mat4 &ProjFromView, const mat4 &ViewFromWorld)
{
    if (StaticInstances.lenu() == 0 && DynamicInstances.lenu() == 0)
        return;

    // NOTE(Kevin): We only ever insert into this array so it should remain sorted
    //              with only the new entries from this frame requiring sorting.
    std::sort(StaticInstances.begin(), StaticInstances.end());

    u32 InstancesCount = StaticInstances.lenu() + DynamicInstances.lenu();
    fixed_array<model_instance_data_t> InstancesCopy(InstancesCount, MemoryType::Frame); 
    InstancesCopy.setlen(InstancesCount);
    memcpy(InstancesCopy.data, StaticInstances.data, 
        StaticInstances.length * sizeof(model_instance_data_t));
    memcpy(InstancesCopy.data + StaticInstances.length, DynamicInstances.data, 
        DynamicInstances.length * sizeof(model_instance_data_t));

    std::sort(InstancesCopy.begin(), InstancesCopy.end());

    UseShader(Sha_ModelInstancedLit);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(Sha_ModelInstancedLit, "MuzzleFlash",
        Player.Weapon.MuzzleFlash.x, 
        Player.Weapon.MuzzleFlash.y, 
        Player.Weapon.MuzzleFlash.z, 
        Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_ModelInstancedLit, "ProjFromView", 1, ProjFromView.ptr());
    GLBindMatrix4fv(Sha_ModelInstancedLit, "ViewFromWorld", 1, ViewFromWorld.ptr());
    GLBind3f(Sha_ModelInstancedLit, "DirectionToSun", RuntimeMap->DirectionToSun.x,
        RuntimeMap->DirectionToSun.y, RuntimeMap->DirectionToSun.z);

    InstanceDataSSBO.BeginFrame();

    auto [MappedPtr, GPUOffset] = InstanceDataSSBO.Alloc();
    memcpy(MappedPtr, InstancesCopy.data, InstancesCopy.lenu() * sizeof(model_instance_data_t));

    InstanceDataSSBO.Bind(0);

    size_t BaseInstanceIndex = GPUOffset / sizeof(model_instance_data_t);

    u16 CurrentModelId = InstancesCopy[0].CorpseModel->MT_ID;
    u32 CurrentModelCount = 0;
    u32 CurrentOffsetFromBase = 0;
    for (u32 i = 0; i <= InstancesCopy.lenu(); ++i)
    {
        if (i == InstancesCopy.lenu() || InstancesCopy[i].CorpseModel->MT_ID != CurrentModelId)
        {
            // flush
            GLBind1i(Sha_ModelInstancedLit, "BaseInstanceIndex", 
                (GLint)BaseInstanceIndex + CurrentOffsetFromBase);
            DrawModelInstanced(Assets.ModelsTextured[CurrentModelId], CurrentModelCount);

            if (i == InstancesCopy.lenu())
                break;

            CurrentOffsetFromBase += CurrentModelCount;
            CurrentModelId = InstancesCopy[i].CorpseModel->MT_ID;
            CurrentModelCount = 0;
        }
        ++CurrentModelCount;
    }

    InstanceDataSSBO.EndFrame();
}


