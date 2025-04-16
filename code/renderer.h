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

struct textured_lit_drawinfo
{
    model_instance_data_t RenderingInfo;
    // perhaps I should do a discriminated union here so I can also draw ModelGLTF with this struct
    GPUMeshIndexed M;
    GPUTexture T;
};

void RenderTexturedLitMeshes(
    fixed_array<textured_lit_drawinfo> &TexLitDrawInfos,
    struct game_state *GameState,
    const mat4 &ProjFromView, 
    const mat4 &ViewFromWorld);

struct sm_drawinfo
{
    model_instance_data_t RenderingInfo;
    animator_t *Animator;
    skinned_model_t *Model;
};

void FillSkinnedModelDrawInfo(
    sm_drawinfo *DrawInfo,
    struct game_state *GameState,
    vec3 ModelCentroid,
    vec3 RenderPosition,
    quat RenderRotation,
    animator_t *Animator,
    skinned_model_t *Model);

void RenderSkinnedModels(
    fixed_array<sm_drawinfo> &SMDrawInfos,
    struct game_state *GameState,
    const mat4 &ProjFromView, 
    const mat4 &ViewFromWorld);

void RenderGameState(struct game_state *GameState);


