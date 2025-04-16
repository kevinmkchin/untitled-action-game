#pragma once

#include "common.h"
#include "mem.h"
#include "instanced.h"
#include "anim.h"

// This is the Rendering Abstraction Layer
// Renderable data should be collected by the game layer into generic formats
// and provided to this interface.

void AcquireRenderingResources();
void ReleaseRenderingResources();

struct sm_draw_info
{
    model_instance_data_t RenderingInfo;
    animator_t *Animator;
    skinned_model_t *Model;
};

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

void RenderGameState(struct game_state *GameState);


