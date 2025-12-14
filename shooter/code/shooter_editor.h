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
    TranslateEdgesTool,

    SelectFace,

    BlockTool,

    PolygonTool,
};

struct undo_state
{
    world_mesh World;
    editor_mode ActiveMode;
    // TODO(Kevin): Undo should also contain selection info and other editor state
};

struct undo_stack
{
    undo_state WorldStates[20];
    u32 StackP;
    u32 MostRecentP;
    u32 HistoryStartP;
};

struct element_selection
{
    u32 Selection[16];
    u32 SelectionCount;
};

struct editor
{
    bool32 IsInitialized;

    memory_blocks EditorMemory;
    renderer *RenderAPI;
    gui_state *GuiState;

    undo_stack *UndoStack;

    world_mesh *World;

    element_selection FaceSelection;
    element_selection EdgeSelection;

    bool32 MouseOverViewport;
    bool32 MouseNotCaptured;
    vec2 ViewportMouseP;
    bool32 ViewportLMBIsDown;
    bool32 ViewportLMBWentDown;
    bool32 ViewportLMBWentUp;
    debug_keyboard_input *KeyboardState;

    vec3 CameraDirection;
    vec3 WorldViewportP;

    editor_mode ActiveMode;
    editor_mode QueuedMode;
    editor_mode PrevMode;

    edge_index EdgeUnderMouse;
    face_index FaceUnderMouse;

    real32 GridIncrement;
    vec3 GridOrigin;
    vec3 GridUp;
    vec3 GridRight;

    u32 ToolStage; // NOTE(Kevin): Tool Stage and editor Mode are orthogonal.
    vec3 RectStartP;
    vec3 RectEndP;
    vec3 RectDrawPlaneN;
    vec3 RectDrawBasisUAxis;
    vec3 RectDrawBasisVAxis;
    vec3 BlockDrawHeight;

    vec3 MouseInteractStartP;
    freeable_list_vec3 MoveVertexOldP;

    freeable_list_u32 WorkingSetVerts;
    freeable_list_u32 WorkingSetEdges;
    freeable_list_u32 WorkingSetFaces;
};

internal void
EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                      kbm_input *Input, renderer *RenderAPI, real32 dt);

internal void
EnterNextMode(editor *Editor);


#endif // SHOOTER_EDITOR_H
