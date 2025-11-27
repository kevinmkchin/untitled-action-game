#ifndef SHOOTER_EDITOR_H
#define SHOOTER_EDITOR_H

#include "shooter.h"
#include "shooter_editmesh.h"

struct game_state;
struct gui_state;
struct renderer;
struct kbm_input;

enum editor_mode
{
    NothingMode,

    SelectEdge,
    SelectFace,
    BlockTool,
    PolygonTool,
};

struct undo_stack
{
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
    renderer *RenderAPI;    

    undo_stack *UndoStack;

    world_mesh *World;

    face_index FaceSelection[16];
    u32        FaceSelectionCount;

    vec2 ViewportMouseP;
    bool32 ViewportLMBIsDown;
    bool32 ViewportLMBWentDown;
    bool32 ViewportLMBWentUp;

    vec3 CameraDirection;
    vec3 WorldViewportP;

    editor_mode ActiveMode;
    editor_mode QueuedMode;

    edge_index EdgeUnderMouse;
    face_index FaceUnderMouse;

    real32 GridIncrement;
    vec3 GridOrigin;
    vec3 GridUp;
    vec3 GridRight;

    u32 BlockToolStage;
    vec3 RectStartP;
    vec3 RectEndP;
    vec3 RectDrawPlaneN;
    vec3 RectDrawBasisUAxis;
    vec3 RectDrawBasisVAxis;
    vec3 BlockDrawHeight;
};

internal void
EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                      kbm_input *Input, renderer *RenderAPI, real32 dt);

#endif // SHOOTER_EDITOR_H
