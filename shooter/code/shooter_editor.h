#ifndef SHOOTER_EDITOR_H
#define SHOOTER_EDITOR_H

#include "shooter.h"
#include "shooter_editmesh.h"

struct game_state;
struct gui_state;

struct undo_stack
{
    // NOTE(Kevin): Having separate undo stack and redo stack is
    //              easier to reason about but requires twice the
    //              memory.
    // TODO(Kevin): Undo should also contain selection info and other editor state

    world_mesh WorldStates[20];
    u32 StackP;
    u32 MostRecentP;
    u32 HistoryStartP;
};

struct editor
{
    bool32 IsInitialized;

    memory_blocks EditorMemory;

    undo_stack *UndoStack;

    world_mesh *World;
};

internal void
EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                      game_input *Input, struct renderer *RendererAPI);

#endif // SHOOTER_EDITOR_H
