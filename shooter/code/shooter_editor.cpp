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

#if 0
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

internal void
EarClippingTriangulationCCW(memory_blocks *Memory, vec3 *Points, u32 PointCount,
                            u32 **Triangles, u32 *IndicesCount)
{
    u32 *TriangleIndices = AllocArray(Memory, (PointCount-2)*3, u32);
    u32 TriangleIndicesCount = 0;

    vec3 FaceNormal = Normalize(Cross(Points[1]-Points[0], Points[2]-Points[0]));
    vec3 BasisX = Normalize(Points[1]-Points[0]);
    vec3 BasisY = Normalize(Cross(FaceNormal, BasisX));

    struct point_with_index
    {
        vec2 P;
        u32 Index;
    };

    point_with_index *Points2D = AllocArray(Memory, PointCount, point_with_index);
    u32 Points2DCount = PointCount;
    for (u32 Index = 0; Index < PointCount; ++Index)
    {
        Points2D[Index].P = { Dot(Points[Index], BasisX), Dot(Points[Index], BasisY) };
        Points2D[Index].Index = Index;
    }

    while (Points2DCount > 3)
    {
        for (u32 Index = 0; Index < Points2DCount; ++Index)
        {
            u32 PrevIndex = (Points2DCount + Index - 1) % Points2DCount;
            u32 ThisIndex = Index;
            u32 NextIndex = (Index + 1) % Points2DCount;

            point_with_index PrevPoint = Points2D[PrevIndex];
            point_with_index ThisPoint = Points2D[Index];
            point_with_index NextPoint = Points2D[NextIndex];

            vec2 Edge0 = ThisPoint.P - PrevPoint.P;
            vec2 Edge1 = NextPoint.P - ThisPoint.P;
            vec2 Edge2 = PrevPoint.P - NextPoint.P;

            #define ZeroWithEpsilon(Value) GMATH_abs(Value) < 0.0001f ? 0 : Value;
            real32 SignedArea = ZeroWithEpsilon(Cross(Edge0, Edge1));
            bool32 IsConvex = SignedArea > 0; // NOTE(Kevin): Left turn means convex in CCW

            bool32 IsEar = false;
            if (IsConvex)
            {
                for (u32 Check = 0; Check < Points2DCount; ++Check)
                {
                    if (Check != PrevIndex && Check != ThisIndex && Check != NextIndex)
                    {
                        vec2 CheckP = Points2D[Check].P;
                        real32 Sign0 = ZeroWithEpsilon(Cross(Edge0, CheckP - PrevPoint.P));
                        real32 Sign1 = ZeroWithEpsilon(Cross(Edge1, CheckP - ThisPoint.P));
                        real32 Sign2 = ZeroWithEpsilon(Cross(Edge2, CheckP - NextPoint.P));
                        if (Sign0 >= 0 && Sign1 >= 0 && Sign2 >= 0)
                        {
                            break;
                        }
                    }

                    if (Check == Points2DCount-1)
                    {
                        IsEar = true;
                    }
                }
            }
            #undef ZeroWithEpsilon

            if (IsEar)
            {
                TriangleIndices[TriangleIndicesCount++] = PrevPoint.Index;
                TriangleIndices[TriangleIndicesCount++] = ThisPoint.Index;
                TriangleIndices[TriangleIndicesCount++] = NextPoint.Index;

                // Clip
                if (Index < Points2DCount-1)
                {
                    Copy(&Points2D[Index], &Points2D[Index + 1],
                         sizeof(Points2D[0])*(Points2DCount - (Index + 1)));
                }
                --Points2DCount;

                break;
            }

            if (Index == Points2DCount-1)
            {
                // TODO(Kevin): Error! There were no ears found in this triangle.
                Assert(0);
            }
        }
    }
    TriangleIndices[TriangleIndicesCount++] = Points2D[0].Index;
    TriangleIndices[TriangleIndicesCount++] = Points2D[1].Index;
    TriangleIndices[TriangleIndicesCount++] = Points2D[2].Index;

    *Triangles = TriangleIndices;
    *IndicesCount = TriangleIndicesCount;

    FreeBlock(Memory, Points2D);
}

struct mesh_data
{
    PNUV2_vertex *VB;
    u32           VC;
    u32          *IB;
    u32           IC;
};

internal mesh_data
TriangulateEditMeshFace(memory_blocks *Memory, world_mesh *World, face_index Face)
{
    mesh_data Result = {};

    u32 PointCount = World->Faces[Face].LoopCycleLength;

    if (PointCount > 2)
    {
        vec3 *Points = AllocArray(Memory, PointCount, vec3);
        vec2 *LMUVs = AllocArray(Memory, PointCount, vec2);
        PointCount = 0;

        loop_index Head = World->Faces[Face].LoopCycleBase;
        loop_index Current = Head;
        do
        {
            vert_index Vert = World->Loops[Current].Vert;

            Points[PointCount] = World->Vertices[Vert].P;
            LMUVs[PointCount] = World->Loops[Current].LightmapUV;
            ++PointCount;

            Current = World->Loops[Current].Next;
        }
        while (Current != Head);

        PNUV2_vertex *Vertices = AllocArray(Memory, PointCount, PNUV2_vertex);
        u32 VertexCount = 0;
        u32 *Triangles;
        u32 IndicesCount;
        EarClippingTriangulationCCW(Memory, Points, PointCount, &Triangles, &IndicesCount);

        vec3 WorldN = Normalize(Cross(Points[1]-Points[0], Points[2]-Points[0]));

        for (u32 PointIndex = 0; PointIndex < PointCount; ++PointIndex)
        {
            vec3 WorldP = Points[PointIndex];
            vec2 LMUV = LMUVs[PointIndex];

            vec2 AlbedoUV = {};

            // TODO(Kevin): Better UVs with basis vectors?
            real32 UVScale = (1.f / (real32)STANDARD_LENGTH_IN_GAME_UNITS); // [0,1] scaled to 32 units
            real32 XF = GMATH_abs(Dot(WorldN, vec3(1, 0, 0)));
            real32 YF = GMATH_abs(Dot(WorldN, vec3(0, 1, 0)));
            real32 ZF = GMATH_abs(Dot(WorldN, vec3(0, 0, 1)));
            if(XF >= YF && XF >= ZF)
            {
                AlbedoUV.X = WorldP.Z * UVScale;
                AlbedoUV.Y = WorldP.Y * UVScale;
            }
            else if(YF >= XF && YF >= ZF)
            {
                AlbedoUV.X = WorldP.X * UVScale;
                AlbedoUV.Y = WorldP.Z * UVScale;
            }
            else if(ZF >= XF && ZF >= YF)
            {
                AlbedoUV.X = WorldP.X * UVScale;
                AlbedoUV.Y = WorldP.Y * UVScale;
            }

            PNUV2_vertex V;
            V.XYZ = WorldP;
            V.IJK = WorldN;
            V.UV0 = AlbedoUV;
            V.UV1 = LMUV;

            Vertices[VertexCount++] = V;
        }

        Result.VB = Vertices;
        Result.VC = VertexCount;
        Result.IB = Triangles;
        Result.IC = IndicesCount;

        FreeBlock(Memory, Points);
        FreeBlock(Memory, LMUVs);
    }

    return Result;
}

void EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                           game_input *Input, struct renderer *RendererAPI)
{
    local_persist bool32 EditorInitialized = false;
    local_persist mesh_handle TestFace;
    if (!EditorInitialized)
    {
        world_mesh *World = AllocStruct(&GameState->EditorMemory, world_mesh);
        InitializeWorldMesh(World);
        World->Memory = &GameState->EditorMemory;

        MakeEditableRectangle(World, { 100, 0, 100 }, { -100, 0, -100 }, GMATH_UP);

        mesh_data FaceMesh = TriangulateEditMeshFace(&GameState->EditorMemory, World, 1);
        TestFace = RendererAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC, FaceMesh.IB, FaceMesh.IC);
        FreeBlock(&GameState->EditorMemory, FaceMesh.VB);
        FreeBlock(&GameState->EditorMemory, FaceMesh.IB);

        EditorInitialized = true;
    }

    render_item RenderItem;
    RenderItem = {};
    RenderItem.Descriptor.Mesh = TestFace;
    RenderItem.Descriptor.Albedo = GameState->TexWireframeRed;
    RenderItem.Descriptor.Lightmap = 0;
    RenderItem.UseLightmap = false;
    RenderItem.UseLightCubes = true;
    RenderItem.WorldPosition = vec3();
    RenderItem.WorldRotation = quat();
    RenderItem.WorldScale = vec3(1,1,1);

    RendererAPI->PushRenderItems(&RenderItem, 1);
}

