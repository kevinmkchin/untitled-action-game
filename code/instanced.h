#pragma once

#include "common.h"
#include "mem.h"
#include "anim.h"

// corpses are simply instanced static models that are textured and lit with no collision

struct alignas(16) model_instance_data_t
{
    /*  I need this struct to be passable to GLSL shader
        https://community.khronos.org/t/ssbo-std430-layout-rules/109761
        std430 Layout 
    */

    mat4 WorldFromModel;         // 64      : 64
    vec4 PointLightsPos[4];      // 64      : 128
    float AmbientCube[6];        // 4*6=24  : 152
    i32 DoSunLight;              // 4       : 156
    i32 PointLightsCount;        // 4       : 160
    float PointLightsAttLin[4];  // 4*4=16  : 176
    float PointLightsAttQuad[4]; // 4*4=16  : 192

    ModelGLTF *CorpseModel;      // 8       : 200
    u64 _padding_;               // 8       : 208

    bool operator<(const model_instance_data_t &Other) const
    {
        if (!CorpseModel)
            return false;
        if (!Other.CorpseModel)
            return true;
        return CorpseModel->MT_ID < Other.CorpseModel->MT_ID;
    }
};

constexpr u32 MaxStaticInstances = 2000;
constexpr u32 MaxDynamicInstances = 1000;

void FillModelInstanceData(struct game_state *GameState, model_instance_data_t *InstanceData, 
    vec3 ModelCentroid, vec3 RenderPosition, quat RenderRotation, ModelGLTF *InstanceModel);

void InstanceDrawing_AcquireGPUResources();
void InstancedDrawing_ReleaseGPUResources();
void SortAndDrawInstancedModels(struct game_state *RuntimeMap, 
    fixed_array<model_instance_data_t> &StaticInstances,
    fixed_array<model_instance_data_t> &DynamicInstances,
    const mat4 &ProjFromView, const mat4 &ViewFromWorld);
