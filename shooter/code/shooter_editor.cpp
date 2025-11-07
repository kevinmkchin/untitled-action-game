#include "shooter_editor.h"

#include "shooter_editmesh.cpp"

internal void
GetDrawingRectangleXZAxes(vec3 FloorNormal, vec3 *RectangleXAxis, vec3 *RectangleZAxis)
{
    vec3 FloorNormalNoY = vec3(FloorNormal.X, 0, FloorNormal.Z);
    if (Magnitude(FloorNormalNoY) < 0.001f)
    {
        FloorNormalNoY = GMATH_FORWARD;
    }
    else if (AbsoluteValue(FloorNormal.y) < 0.001f)
    {
        FloorNormalNoY = GMATH_UP;
    }
    else
    {
        FloorNormalNoY = Normalize(FloorNormalNoY);
    }
    *RectangleZAxis = Normalize(Cross(FloorNormal, FloorNormalNoY));
    *RectangleXAxis = Normalize(Cross(FloorNormal, *RectangleZAxis));
}

internal void
MakeEditableRectangle(world_mesh *World, vec3 RectStartPoint, vec3 RectEndPoint,
                      vec3 FloorNormal)
{
    vec3 RectangleXAxis, RectangleZAxis;
    GetDrawingRectangleXZAxes(FloorNormal, &RectangleXAxis, &RectangleZAxis);
    vec3 AlongZAxis = Dot(RectEndPoint - RectStartPoint, RectangleZAxis) * RectangleZAxis;

    vec3 FloorPointA = RectStartPoint;
    vec3 FloorPointB = RectStartPoint + AlongZAxis;
    vec3 FloorPointC = RectEndPoint;
    vec3 FloorPointD = RectEndPoint - AlongZAxis;

#if 1
    vert_index V0 = CreateVert(World, FloorPointA);
    vert_index V1 = CreateVert(World, FloorPointB);
    vert_index V2 = CreateVert(World, FloorPointC);
    vert_index V3 = CreateVert(World, FloorPointD);

    edge_index E0 = CreateEdge(World, V0, V1);
    edge_index E1 = CreateEdge(World, V1, V2);
    edge_index E2 = CreateEdge(World, V2, V3);
    edge_index E3 = CreateEdge(World, V3, V0);

    edge_index Edges[] = { E0, E1, E2, E3 };
#else
    // Ear clipping test!
    vert_index V0 = CreateVert(World, FloorPointA);
    vert_index V1 = CreateVert(World, FloorPointB);
    vert_index V2 = CreateVert(World, FloorPointC);
    vert_index V3 = CreateVert(World, 0.2f*(RectEndPoint - RectStartPoint) + RectStartPoint);
    vert_index V4 = CreateVert(World, FloorPointD);

    edge_index E0 = CreateEdge(World, V0, V1);
    edge_index E1 = CreateEdge(World, V1, V2);
    edge_index E2 = CreateEdge(World, V2, V3);
    edge_index E3 = CreateEdge(World, V3, V4);
    edge_index E4 = CreateEdge(World, V4, V0);

    edge_index Edges[] = { E0, E1, E2, E3, E4 };
#endif

    // TODO(Kevin): Perhaps created faces should be stored in an
    //              array for quickly iterating for rendering etc.
    CreateFace(World, Edges, ArrayCount(Edges));
}

internal u32
NextUndoIndex(u32 Index)
{
    u32 Result = (Index + 1) % ArrayCount(((undo_stack *)0)->WorldStates);
    return Result;
}

internal u32
PrevUndoIndex(u32 Index)
{
    u32 Result = (Index + ArrayCount(((undo_stack *)0)->WorldStates) - 1) % 
                 ArrayCount(((undo_stack *)0)->WorldStates);
    return Result;    
}

internal void
UndoWorldState(undo_stack *Undo, world_mesh *World)
{
    if (Undo->StackP != Undo->HistoryStartP)
    {
        Undo->StackP = PrevUndoIndex(Undo->StackP);
        Copy(World, &(Undo->WorldStates[Undo->StackP]), sizeof(world_mesh));
    }
    else
    {
        // TODO(Kevin): Notify that we are at the bottom of undo stack
    }
}

internal void
RedoWorldState(undo_stack *Undo, world_mesh *World)
{
    if (Undo->StackP != Undo->MostRecentP)
    {
        Undo->StackP = NextUndoIndex(Undo->StackP);
        Copy(World, &(Undo->WorldStates[Undo->StackP]), sizeof(world_mesh));
    }
    else
    {
        // TODO(Kevin): Notify that we are at the top of undo stack (no more redo)
        // or maybe we don't notify in this case?
    }
}

internal void
PushWorldStateToUndoStack(undo_stack *Undo, world_mesh *World)
{
    u32 NextStackP = NextUndoIndex(Undo->StackP);

    if (NextStackP == Undo->HistoryStartP)
    {
        // If we are about to write to HistoryStartP, increment HistoryStartP
        Undo->HistoryStartP = NextUndoIndex(Undo->HistoryStartP);

        // NOTE(Kevin): Previous HistoryStartP is now unreachable
    }

    if (Undo->StackP != Undo->MostRecentP)
    {
        // NOTE(Kevin): Everything from [NextStackP, Undo->MostRecentP] will become unreachable
    }

    Undo->StackP = NextStackP;
    Undo->MostRecentP = NextStackP;

    world_mesh *PushedState = &(Undo->WorldStates[Undo->StackP]);
    Copy(PushedState, World, sizeof(world_mesh));

    // NOTE(Kevin): Rather than keeping all the hw meshes of the undo states alive
    //              (which would require lifetime management i.e. determine whether
    //              a hw mesh is unreachable; difficult if hw meshes allowed to be
    //              shared between undo states; if disallow, then with every modification
    //              I must retriangulate and upload whole new meshes), delete all hw
    //              meshes, retriangulate, and upload new hw meshes upon undo/redo.
    //              This way, I pay this cost at undo/redo time, not at modification
    //              time. Only dirty faces need to be deleted/recreated on modifications.
    //              This keeps modifications snappy, while undo/redo action takes longer.
    // NOTE(Kevin): Don't push any hw mesh handles to undo stack
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(PushedState->Faces); ++FaceIndex)
    {
        mesh_face *Face = &PushedState->Faces[FaceIndex];
        Face->RenderMesh = 0;
    }
}

internal void
DeleteAllFaceRenderMeshes(world_mesh *World, renderer *RendererAPI)
{
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];

        if (Face->RenderMesh)
        {
            RendererAPI->DeleteMesh(Face->RenderMesh);
            Face->RenderMesh = 0;
        }
    }
}

internal void
TriangulateAndUploadFaces(world_mesh *World, editor *Editor, renderer *RendererAPI)
{
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];

        if (Face->Alive)
        {
            if (!Face->RenderMesh)
            {
                vertex_buffer FaceMesh = TriangulateEditMeshFace(&Editor->EditorMemory, World, FaceIndex);
                if (FaceMesh.IC > 2)
                {
                    Face->RenderMesh = RendererAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC,
                                                                   FaceMesh.IB, FaceMesh.IC);
                }
                FreeBlock(&Editor->EditorMemory, FaceMesh.VB);
                FreeBlock(&Editor->EditorMemory, FaceMesh.IB);
            }
        }
    }
}

internal void
EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                      game_input *Input, struct renderer *RendererAPI)
{
    editor *Editor = &(GameState->Editor);
    world_mesh *World = Editor->World;

    if (!Editor->IsInitialized)
    {
        Editor->World = AllocStruct(&Editor->EditorMemory, world_mesh);
        InitializeWorldMesh(Editor->World);
        World = Editor->World;
        World->Memory = &Editor->EditorMemory;

#if 0
        MakeEditableRectangle(World, { 100, 0, 100 }, { -100, 0, -100 }, GMATH_UP);

        edge_index ToExtrude[] = { 1, 3 };
        ExtrudeEdges(World, ToExtrude, ArrayCount(ToExtrude));
        World->Vertices[5].P += vec3(0, 32, 0);
        World->Vertices[6].P += vec3(0, 32, 0);
        World->Vertices[7].P += vec3(0, 16, 0);
        World->Vertices[8].P += vec3(0, 32, 0);

        // SplitEdgeMakeVert(World, 1, 0.3f);
        // SplitEdgeMakeVert(World, 2, 0.5f);
        // SplitEdgeMakeVert(World, 3, 0.5f);
        // SplitEdgeMakeVert(World, 4, 0.5f);
        // ReverseFaceLoop(World, 1);
#else
        for (s32 I = 1; I < 10000; ++I)
        {
            float Y = ((I/1000)-5)*64.f;
            float X = (((I%1000)/32)-16)*64.f;
            float Z = (((I%1000)%32)-16)*64.f;
            MakeEditableRectangle(World, { X, Y, Z }, { X+32.f, Y, Z+32.f }, GMATH_UP);
        }
#endif
        Editor->UndoStack = AllocStruct(&Editor->EditorMemory, undo_stack);
        // NOTE(Kevin): Push initial world state to undo stack
        TriangulateAndUploadFaces(World, Editor, RendererAPI);
        PushWorldStateToUndoStack(Editor->UndoStack, World);
        Editor->UndoStack->HistoryStartP = Editor->UndoStack->StackP;

        // TODO(Kevin): For face colliders, I should first detect whether the
        // face polygon is convex or concave: if concave, then I may need to
        // treat each triangle as a collider; if convex, then I'm fine to treat
        // the whole polygon as a collider.

        Editor->IsInitialized = true;
    }

    if (Input->Controllers[0].ActionUp.EndedDown && 
        Input->Controllers[0].ActionUp.HalfTransitionCount > 0)
    {
        for (u32 VertIndex = 0; VertIndex < ArrayCount(World->Vertices); ++VertIndex)
        {
            mesh_vert *Vert = &World->Vertices[VertIndex];
            Vert->P += vec3(0, 16, 0);
        }

        // TODO(Kevin): Only need to delete+recreate the dirty faces, but right now its all dirty
        DeleteAllFaceRenderMeshes(World, RendererAPI);
        TriangulateAndUploadFaces(World, Editor, RendererAPI);

        // NOTE(Kevin): Create snapshot of world after changes/modification
        PushWorldStateToUndoStack(Editor->UndoStack, World);
    }

    if (Input->Controllers[0].ActionLeft.EndedDown && 
        Input->Controllers[0].ActionLeft.HalfTransitionCount >= 0)
    {
        DeleteAllFaceRenderMeshes(World, RendererAPI);
        UndoWorldState(Editor->UndoStack, World);
        TriangulateAndUploadFaces(World, Editor, RendererAPI);
    }
    if (Input->Controllers[0].ActionRight.EndedDown && 
        Input->Controllers[0].ActionRight.HalfTransitionCount >= 0)
    {
        DeleteAllFaceRenderMeshes(World, RendererAPI);
        RedoWorldState(Editor->UndoStack, World);
        TriangulateAndUploadFaces(World, Editor, RendererAPI);
    }

    // I want all the faces to be drawn & I want all the edges to be drawn!
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];

        if (Face->Alive)
        {
            if (Face->RenderMesh)
            {
                render_item RenderItem;
                RenderItem = {};
                RenderItem.Descriptor.Mesh = Face->RenderMesh;
                RenderItem.Descriptor.Albedo = 0;
                RenderItem.Descriptor.Lightmap = 0;
                RenderItem.UseLightmap = true;
                RenderItem.UseLightCubes = false;
                RenderItem.ForcedAmbient = GetDebugColor3(FaceIndex);
                RenderItem.WorldPosition = vec3();
                RenderItem.WorldRotation = quat();
                RenderItem.WorldScale = vec3(1,1,1);
                RendererAPI->PushRenderItems(&RenderItem, 1);                
            }
        }
    }
}

