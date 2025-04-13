#pragma once

#include "common.h"
#include "shaders.h"
#include "game.h"

void BindUniformsForModelLighting(GPUShader &Shader, game_state *MapInfo, vec3 ModelPosition);

