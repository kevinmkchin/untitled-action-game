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

void EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
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

        // TODO(Kevin): For face colliders, I should first detect whether the
        // face polygon is convex or concave: if concave, then I may need to
        // treat each triangle as a collider; if convex, then I'm fine to treat
        // the whole polygon as a collider.

        Editor->IsInitialized = true;
    }

    // I want all the faces to be drawn & I want all the edges to be drawn!
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];
        if (Face->Alive)
        {
            if (!Face->RenderMesh)
            {
                // TODO(Kevin): Each face should have mesh_handle to its hw mesh -> should we delete
                //              hw mesh on undo/redo? or only once it is no longer reachable?
                //              If delete once no longer reachable (out of undo stack), I need to 
                //              keep alive all hw meshes of the states in the undo stack.
                //              I would need a large HandlesToMesh array (lots of Handles), but
                //              the buffer for instanced drawing only needs enough memory for one
                //              state. I would use more VRAM in total though for all the vertices,
                //              but it wouldn't be that much anyways.
                //              If I delete all the hw meshes of one state and recreate all the 
                //              hw meshes of another, I need to profile how long that takes and
                //              whether it is acceptable.
                //              Undo should also contain selection info and other editor state

                vertex_buffer FaceMesh = TriangulateEditMeshFace(&Editor->EditorMemory, World, FaceIndex);
                if (FaceMesh.IC > 2)
                {
                    Face->RenderMesh = RendererAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC,
                                                                   FaceMesh.IB, FaceMesh.IC);
                }
                FreeBlock(&Editor->EditorMemory, FaceMesh.VB);
                FreeBlock(&Editor->EditorMemory, FaceMesh.IB);
            }

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

