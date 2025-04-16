#pragma once

#include "common.h"
#include "mem.h"
#include "instanced.h"
#include "anim.h"

struct sm_draw_info
{
    model_instance_data_t RenderingInfo;
    animator_t *Animator;
    skinned_model_t *Model;
};

void AcquireResources();

void FillSkinnedModelDrawInfo(
    sm_draw_info *DrawInfo,
    struct game_state *GameState,
    vec3 ModelCentroid,
    vec3 RenderPosition,
    quat RenderRotation,
    animator_t *Animator,
    skinned_model_t *Model);

void RenderSkinnedModels(
    fixed_array<sm_draw_info> &SMDrawInfos,
    struct game_state *GameState,
    const mat4 &ProjFromView, 
    const mat4 &ViewFromWorld);




