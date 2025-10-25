#include "shooter_editor.h"
#include "shooter_editmesh.h"
#include "shooter_editmesh.cpp"

internal void
MakeEditableRectangle(world_mesh *World)
{
    vert_index V0 = CreateVert(World, vec3(-320,0,-320));
    vert_index V1 = CreateVert(World, vec3(-320,0,320));
    vert_index V2 = CreateVert(World, vec3(320,0,320));
    vert_index V3 = CreateVert(World, vec3(320,0,-320));

    edge_index E0 = CreateEdge(World, V0, V1);
    edge_index E1 = CreateEdge(World, V1, V2);
    edge_index E2 = CreateEdge(World, V2, V3);
    edge_index E3 = CreateEdge(World, V3, V0);

    edge_index Edges[] = { E0, E1, E2, E3 };

    CreateFace(World, Edges, ArrayCount(Edges));
}

void EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                           game_input *Input, struct renderer *RendererAPI)
{
    local_persist bool32 EditorInitialized = false;
    if (!EditorInitialized)
    {
        world_mesh *World = (world_mesh *)MemBlocksAlloc(&GameState->EditorMemory, sizeof(world_mesh));
        InitializeWorldMesh(World);
        World->Memory = &GameState->EditorMemory;

        MakeEditableRectangle(World);

        EditorInitialized = true;
    }

    // TODO(Kevin): Make cube
    // TODO(Kevin): Triangulate cube
    // TODO(Kevin): Render cube

}

