#ifndef SHOOTER_EDITOR_H
#define SHOOTER_EDITOR_H

#include "shooter.h"
#include "shooter_editmesh.h"

struct game_state;
struct gui_state;

struct editor
{
    bool32 IsInitialized;

    memory_blocks EditorMemory;

    world_mesh *World;
};

void EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                           game_input *Input, struct renderer *RendererAPI);

#endif // SHOOTER_EDITOR_H
