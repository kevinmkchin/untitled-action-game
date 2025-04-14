#pragma once

#include "common.h"
#include "mem.h"
#include "enemy.h"

void AcquireResources();

void RenderEnemies(
    fixed_array<enemy_t> &Enemies,
    struct game_state *GameState, 
    const mat4 &ProjFromView, 
    const mat4 &ViewFromWorld);



