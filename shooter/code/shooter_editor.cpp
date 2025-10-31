#include "shooter_editor.h"
#include "shooter_editmesh.h"
#include "shooter_editmesh.cpp"

internal void
GetDrawingRectangleXZAxes(vec3 FloorNormal, vec3 *RectangleXAxis, vec3 *RectangleZAxis)
{
    vec3 FloorNormalNoY = vec3(FloorNormal.X, 0, FloorNormal.Z);
    if (Magnitude(FloorNormalNoY) < 0.001f)
    {
        FloorNormalNoY = GMATH_FORWARD;
    }
    else if (GMATH_abs(FloorNormal.y) < 0.001f)
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

void EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                           game_input *Input, struct renderer *RendererAPI)
{
    local_persist bool32 EditorInitialized = false;
    local_persist mesh_handle TestFace1 = 0;
    local_persist mesh_handle TestFace2 = 0;
    local_persist mesh_handle TestFace3 = 0;
    if (!EditorInitialized)
    {
        world_mesh *World = AllocStruct(&GameState->EditorMemory, world_mesh);
        InitializeWorldMesh(World);
        World->Memory = &GameState->EditorMemory;

        MakeEditableRectangle(World, { 100, 0, 100 }, { -100, 0, -100 }, GMATH_UP);

        edge_index ToExtrude[] = { 1, 3 };
        ExtrudeEdges(World, ToExtrude, ArrayCount(ToExtrude));
        World->Vertices[5].P += vec3(0, 32, 0);
        World->Vertices[6].P += vec3(0, 32, 0);
        World->Vertices[7].P += vec3(0, 0, 0);
        World->Vertices[8].P += vec3(0, 0, 0);

        // SplitEdgeMakeVert(World, 1, 0.3f);
        // SplitEdgeMakeVert(World, 2, 0.5f);
        // SplitEdgeMakeVert(World, 3, 0.5f);
        // SplitEdgeMakeVert(World, 4, 0.5f);
        // ReverseFaceLoop(World, 1);

        // TODO(Kevin): Each face should have mesh_handle to its hw mesh -> should we delete
        //              hw mesh on undo/redo? or only once it is no longer reachable? probably
        //              once its no longer reachable...I shouldn't have to delete all the hw meshes
        //              of one state and recreate all the hw meshes of another just for one undo.
        //              Undo should also contain selection info and other editor state

        vertex_buffer FaceMesh;
        FaceMesh = TriangulateEditMeshFace(&GameState->EditorMemory, World, 1);
        if (FaceMesh.IC > 2)
        {
            TestFace1 = RendererAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC, FaceMesh.IB, FaceMesh.IC);
        }
        FreeBlock(&GameState->EditorMemory, FaceMesh.VB);
        FreeBlock(&GameState->EditorMemory, FaceMesh.IB);

        FaceMesh = TriangulateEditMeshFace(&GameState->EditorMemory, World, 2);
        if (FaceMesh.IC > 2)
        {
            TestFace2 = RendererAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC, FaceMesh.IB, FaceMesh.IC);
        }
        FreeBlock(&GameState->EditorMemory, FaceMesh.VB);
        FreeBlock(&GameState->EditorMemory, FaceMesh.IB);

        FaceMesh = TriangulateEditMeshFace(&GameState->EditorMemory, World, 3);
        if (FaceMesh.IC > 2)
        {
            TestFace3 = RendererAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC, FaceMesh.IB, FaceMesh.IC);
        }
        FreeBlock(&GameState->EditorMemory, FaceMesh.VB);
        FreeBlock(&GameState->EditorMemory, FaceMesh.IB);

        // TODO(Kevin): For face colliders, I should first detect whether the
        // face polygon is convex or concave: if concave, then I may need to
        // treat each triangle as a collider; if convex, then I'm fine to treat
        // the whole polygon as a collider.

        EditorInitialized = true;
    }

    // I want all the faces to be drawn & I want all the edges to be drawn!

    render_item RenderItem;
    if (TestFace1)
    {
        RenderItem = {};
        RenderItem.Descriptor.Mesh = TestFace1;
        RenderItem.Descriptor.Albedo = 0;
        RenderItem.Descriptor.Lightmap = 0;
        RenderItem.UseLightmap = true;
        RenderItem.UseLightCubes = false;
        RenderItem.ForcedAmbient = vec3(0,1,0);
        RenderItem.WorldPosition = vec3();
        RenderItem.WorldRotation = quat();
        RenderItem.WorldScale = vec3(1,1,1);
        RendererAPI->PushRenderItems(&RenderItem, 1);
    }

    if (TestFace2)
    {
        RenderItem = {};
        RenderItem.Descriptor.Mesh = TestFace2;
        RenderItem.Descriptor.Albedo = 0;
        RenderItem.Descriptor.Lightmap = 0;
        RenderItem.UseLightmap = true;
        RenderItem.UseLightCubes = false;
        RenderItem.ForcedAmbient = vec3(1,0,1);
        RenderItem.WorldPosition = vec3();
        RenderItem.WorldRotation = quat();
        RenderItem.WorldScale = vec3(1,1,1);
        RendererAPI->PushRenderItems(&RenderItem, 1);
    }

    if (TestFace3)
    {
        RenderItem = {};
        RenderItem.Descriptor.Mesh = TestFace3;
        RenderItem.Descriptor.Albedo = 0;
        RenderItem.Descriptor.Lightmap = 0;
        RenderItem.UseLightmap = true;
        RenderItem.UseLightCubes = false;
        RenderItem.ForcedAmbient = vec3(1,1,0);
        RenderItem.WorldPosition = vec3();
        RenderItem.WorldRotation = quat();
        RenderItem.WorldScale = vec3(1,1,1);
        RendererAPI->PushRenderItems(&RenderItem, 1);
    }

}

