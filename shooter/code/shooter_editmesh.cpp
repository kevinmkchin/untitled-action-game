#include "shooter_editmesh.h"

internal void
InitializeWorldMesh(world_mesh *Mesh)
{
    // NOTE(Kevin): face, edge, vert, loop indices 0 are reserved for invalid elements.

    Mesh->FreeFaceHandlesCount = ArrayCount(Mesh->FreeFaceHandles);
    PopulateFreeHandles(Mesh->FreeFaceHandles, Mesh->FreeFaceHandlesCount--);

    Mesh->FreeEdgeHandlesCount = ArrayCount(Mesh->FreeEdgeHandles);
    PopulateFreeHandles(Mesh->FreeEdgeHandles, Mesh->FreeEdgeHandlesCount--);

    Mesh->FreeVertexHandlesCount = ArrayCount(Mesh->FreeVertexHandles);
    PopulateFreeHandles(Mesh->FreeVertexHandles, Mesh->FreeVertexHandlesCount--);

    Mesh->FreeTopoLoopHandlesCount = ArrayCount(Mesh->FreeTopoLoopHandles);
    PopulateFreeHandles(Mesh->FreeTopoLoopHandles, Mesh->FreeTopoLoopHandlesCount--);
}

inline loop_index
UseNextAvailableLoop(world_mesh *Mesh)
{
    Assert(Mesh->FreeTopoLoopHandlesCount > 0);
    loop_index FreeHandle = Mesh->FreeTopoLoopHandles[--Mesh->FreeTopoLoopHandlesCount];
    return FreeHandle;
}

#define MeshVert(Index) (Mesh->Vertices[Index])
#define MeshEdge(Index) (Mesh->Edges[Index])
#define MeshFace(Index) (Mesh->Faces[Index])
#define MeshLoop(Index) (Mesh->Loops[Index])
#define ZeroWithEpsilon1(Value) (AbsoluteValue(Value) < 0.0001f ? 0 : (Value))
#define ZeroWithEpsilon2(Value, Eps) (AbsoluteValue(Value) < Eps ? 0 : (Value))

internal void
RadialCycleInsertLoop(world_mesh *Mesh, edge_index Edge, loop_index ToInsert)
{
    Assert(ToInsert);
    topo_loop *L = &Mesh->Loops[ToInsert];
    Assert(Edge);
    mesh_edge *E = &Mesh->Edges[Edge];

    if (E->RadialCycleBase)
    {
        loop_index BaseLoopIndex = E->RadialCycleBase;
        topo_loop *BaseLoop = &Mesh->Loops[BaseLoopIndex];

        L->RadialNext = BaseLoop->RadialNext;
        L->RadialPrev = BaseLoopIndex;
        Mesh->Loops[BaseLoop->RadialNext].RadialPrev = ToInsert;
        BaseLoop->RadialNext = ToInsert;
    }
    else
    {
        L->RadialNext = ToInsert;
        L->RadialPrev = ToInsert;
        E->RadialCycleBase = ToInsert;
    }

    ++E->RadialCycleLen;
}

internal void
RadialCycleRemoveLoop(world_mesh *Mesh, edge_index Edge, loop_index ToRemove)
{
    Assert(ToRemove);
    topo_loop *L = &Mesh->Loops[ToRemove];
    Assert(L->Edge);
    Assert(L->Edge == Edge);
    mesh_edge *E = &Mesh->Edges[L->Edge];
    Assert(E->RadialCycleBase);

    loop_index Current = E->RadialCycleBase;
    do
    {
        topo_loop *CurrentLoop = &Mesh->Loops[Current];

        if (Current == ToRemove)
        {
            if (Current == CurrentLoop->RadialPrev)
            {
                // Do I want to allow edges to exist by themselves?
                E->RadialCycleBase = 0;
                --E->RadialCycleLen;
                Assert(E->RadialCycleLen == 0);

                return;
            }

            Mesh->Loops[CurrentLoop->RadialPrev].RadialNext = CurrentLoop->RadialNext;
            Mesh->Loops[CurrentLoop->RadialNext].RadialPrev = CurrentLoop->RadialPrev;

            E->RadialCycleBase = CurrentLoop->RadialPrev;
            --E->RadialCycleLen;

            return;
        }

        Current = CurrentLoop->RadialNext;
    }
    while (Current != E->RadialCycleBase && Current);

    Assert(0);
}

internal void
RadialCycleRemoveLoop(world_mesh *Mesh, loop_index ToRemove)
{
    Assert(ToRemove);
    Assert(MeshLoop(ToRemove).Edge);
    RadialCycleRemoveLoop(Mesh, MeshLoop(ToRemove).Edge, ToRemove);
}

internal void
DiskCycleInsertEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToInsert)
{
    mesh_vert *Vert = &Mesh->Vertices[DiskCycle];

    Vert->ConnectedEdges[Vert->ConnectedEdgesCount++] = ToInsert;
}

internal void
DiskCycleRemoveEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToRemove)
{
    mesh_vert *Vert = &Mesh->Vertices[DiskCycle];

    for (u32 Index = 0; Index < Vert->ConnectedEdgesCount; ++Index)
    {
        if (Vert->ConnectedEdges[Index] == ToRemove)
        {
            Vert->ConnectedEdges[Index] = Vert->ConnectedEdges[Vert->ConnectedEdgesCount-1];
            Vert->ConnectedEdges[Vert->ConnectedEdgesCount-1] = 0;
            --Vert->ConnectedEdgesCount;
            return;
        }
    }

    Assert(0);
}

internal bool32
HasVert(world_mesh *Mesh, edge_index E, vert_index V)
{
    bool32 Result = (V == MeshEdge(E).A || V == MeshEdge(E).B);
    return Result;
}

internal vert_index
SharesVert(world_mesh *Mesh, edge_index A, edge_index B)
{
    mesh_edge E0 = Mesh->Edges[A];
    mesh_edge E1 = Mesh->Edges[B];

    if (E0.A == E1.A || E0.A == E1.B)
    {
        return E0.A;
    }
    else if (E0.B == E1.A || E0.B == E1.B)
    {
        return E0.B;
    }
    else
    {
        return 0;
    }
}

internal vert_index
CreateVert(world_mesh *Mesh, vec3 P)
{
    Assert(Mesh->FreeVertexHandlesCount > 0);

    vert_index Result = Mesh->FreeVertexHandles[--Mesh->FreeVertexHandlesCount];

    Mesh->Vertices[Result] = {};
    Mesh->Vertices[Result].P = P;

    return Result;
}

internal void
KillVert(world_mesh *Mesh, vert_index Vertex)
{
    // TODO(Kevin): Should this kill the edges too? no, I don't think so...
    // that should be the responsibility of the caller...not sure
}

internal edge_index
CreateEdge(world_mesh *Mesh, vert_index V1, vert_index V2)
{
    Assert(Mesh->FreeEdgeHandlesCount > 0);

    edge_index Result = Mesh->FreeEdgeHandles[--Mesh->FreeEdgeHandlesCount];

    Mesh->Edges[Result] = {};
    Mesh->Edges[Result].A = V1;
    Mesh->Edges[Result].B = V2;
    Mesh->Edges[Result].Alive = true;

    DiskCycleInsertEdge(Mesh, V1, Result);
    DiskCycleInsertEdge(Mesh, V2, Result);

    return Result;
}

internal void
KillEdge(world_mesh *Mesh, edge_index Edge)
{
    // TODO(Kevin): remove edge from vertices disk link

    Mesh->Edges[Edge] = {};
    Mesh->FreeEdgeHandles[Mesh->FreeEdgeHandlesCount++] = Edge;
}

internal face_index
CreateFace(world_mesh *Mesh, edge_index *Edges, u32 EdgeCount)
{
    if (EdgeCount < 3)
    {
        Assert(0);
        return 0;
    }

    // Order the edges by adjacency
    u32 UnprocessedEdgesCount = EdgeCount - 1;
    edge_index *UnprocessedEdges = AllocArray(Mesh->Memory, UnprocessedEdgesCount, edge_index);
    Copy(UnprocessedEdges, Edges + 1, sizeof(edge_index)*UnprocessedEdgesCount);

    edge_index *OrderedEdges = AllocArray(Mesh->Memory, EdgeCount, edge_index);
    OrderedEdges[0] = Edges[0];
    u32 OrderedEdgesCount = 1;

    edge_index LastEdgeOrdered = Edges[0];

    while (UnprocessedEdgesCount)
    {
        bool32 Processed = false;

        for (u32 Index = 0; Index < UnprocessedEdgesCount; ++Index)
        {
            edge_index Edge = UnprocessedEdges[Index];

            if (SharesVert(Mesh, LastEdgeOrdered, Edge))
            {
                OrderedEdges[OrderedEdgesCount++] = Edge;
                LastEdgeOrdered = Edge;

                Copy(&UnprocessedEdges[Index], &UnprocessedEdges[Index + 1],
                     (UnprocessedEdgesCount - (Index + 1))*sizeof(edge_index));
                --UnprocessedEdgesCount;

                Processed = true;
                break;
            }
        }

        if (!Processed)
        {
            // TODO(Kevin): Logging error "Failed to create face from provided edges."
            Assert(0);
            return 0;
        }
    }

    // Create face and loops
    Assert(Mesh->FreeFaceHandlesCount > 0);

    face_index Result = Mesh->FreeFaceHandles[--Mesh->FreeFaceHandlesCount];

    loop_index LastLoop = 0;
    loop_index FirstLoop = 0;
    for (u32 Index = 0; Index < OrderedEdgesCount; ++Index)
    {
        edge_index Edge = OrderedEdges[Index];

        Assert(Mesh->FreeTopoLoopHandlesCount > 0);

        loop_index Loop = UseNextAvailableLoop(Mesh);
        vert_index SharedVert = LastLoop ? SharesVert(Mesh, Mesh->Loops[LastLoop].Edge, Edge)
                                         : SharesVert(Mesh, OrderedEdges[OrderedEdgesCount-1], Edge);
        Assert(SharedVert);

        // Store references to base vertex, corresponding edge, and corresponding face
        Mesh->Loops[Loop] = {};
        Mesh->Loops[Loop].Vert = SharedVert;
        Mesh->Loops[Loop].Edge = Edge;
        Mesh->Loops[Loop].Face = Result;

        // Insert loop into Radial Cycle of this edge
        RadialCycleInsertLoop(Mesh, Edge, Loop);

        // Form Loop Cycle of this face
        if (LastLoop)
        {
            Mesh->Loops[Loop].Prev = LastLoop;
            Mesh->Loops[LastLoop].Next = Loop;
        }
        else
        {
            FirstLoop = Loop;
        }

        LastLoop = Loop;
    }
    Mesh->Loops[FirstLoop].Prev = LastLoop;
    Mesh->Loops[LastLoop].Next = FirstLoop;

    Mesh->Faces[Result] = {};
    Mesh->Faces[Result].LoopCycleBase = FirstLoop;
    Mesh->Faces[Result].LoopCycleLen = EdgeCount;
    Mesh->Faces[Result].Alive = true;
    Mesh->Faces[Result].RenderMesh = 0;

    FreeBlock(Mesh->Memory, &UnprocessedEdges);
    FreeBlock(Mesh->Memory, &OrderedEdges);

    return Result;
}

internal void
KillFace(world_mesh *Mesh, face_index Face)
{
    loop_index Head = Mesh->Faces[Face].LoopCycleBase;
    loop_index Current = Head;
    do
    {
        RadialCycleRemoveLoop(Mesh, Current);
        Mesh->FreeTopoLoopHandles[Mesh->FreeTopoLoopHandlesCount++] = Current;

        Current = Mesh->Loops[Current].Next;
    }
    while (Current != Head);

    Mesh->Faces[Face] = {};
    Mesh->FreeFaceHandles[Mesh->FreeFaceHandlesCount++] = Face;
}

internal vert_index
SplitEdgeMakeVert(world_mesh *Mesh, edge_index OldEdge, float t)
{
    Assert(OldEdge);
    mesh_edge *ToSplit = &MeshEdge(OldEdge);
    vec3 VertP = ((MeshVert(ToSplit->B).P - MeshVert(ToSplit->A).P) * GMATH_Clamp01(t)) + 
                 MeshVert(ToSplit->A).P;
    vert_index CreatedVert = CreateVert(Mesh, VertP);

    DiskCycleRemoveEdge(Mesh, ToSplit->B, OldEdge);
    DiskCycleInsertEdge(Mesh, CreatedVert, OldEdge);

    edge_index CreatedEdge = CreateEdge(Mesh, CreatedVert, ToSplit->B);

    ToSplit->B = CreatedVert;

    // Split the loops of the split edge's radial cycle
    loop_index *RadialCycleCopy = AllocArray(Mesh->Memory, ToSplit->RadialCycleLen, loop_index);
    loop_index CurrentLoopIndex = ToSplit->RadialCycleBase;
    for (u32 Index = 0; Index < ToSplit->RadialCycleLen; ++Index)
    {
        RadialCycleCopy[Index] = CurrentLoopIndex;
        CurrentLoopIndex = MeshLoop(CurrentLoopIndex).RadialNext;
    }
    for (u32 Index = 0; Index < ToSplit->RadialCycleLen; ++Index)
    {
        loop_index OldLoop = RadialCycleCopy[Index];
        loop_index NewLoop = UseNextAvailableLoop(Mesh);

        MeshLoop(NewLoop).Vert = CreatedVert;

        if (MeshLoop(OldLoop).Vert == ToSplit->A)
        {
            MeshLoop(NewLoop).Edge = CreatedEdge;
            RadialCycleInsertLoop(Mesh, CreatedEdge, NewLoop);
        }
        else
        {
            Assert(MeshLoop(OldLoop).Vert == ToSplit->B);

            MeshLoop(NewLoop).Edge = OldEdge;
            RadialCycleRemoveLoop(Mesh, OldEdge, OldLoop);
            RadialCycleInsertLoop(Mesh, OldEdge, NewLoop);
            RadialCycleInsertLoop(Mesh, CreatedEdge, OldLoop);
        }

        MeshLoop(NewLoop).Face = MeshLoop(OldLoop).Face;

        MeshLoop(NewLoop).Next = MeshLoop(OldLoop).Next;
        MeshLoop(NewLoop).Prev = OldLoop;
        MeshLoop(OldLoop).Next = NewLoop;
        MeshLoop(MeshLoop(NewLoop).Next).Prev = NewLoop;

        MeshFace(MeshLoop(NewLoop).Face).LoopCycleLen += 1;
    }

    FreeBlock(Mesh->Memory, &RadialCycleCopy);

    return CreatedVert;
}

internal edge_index
JoinEdgeKillVert(world_mesh *Mesh, edge_index Edge, vert_index Vert)
{
    // TODO(Kevin): 
    // Takes a pointer to an edge (ke) and pointer to one of its vertices (kv)
    // and collapses the edge on that vertex. First ke is removed from the disk
    // cycle of both kv and tv. Then the edge oe is relinked to run between ov
    // and tv and is added to the disk cycle of ov. Finally the radial cycle of
    // oe is traversed and all of its face loops are updated. Note that in order
    // for this euler to work, kv must have exactly only two edges coincident 
    // upon it (valance of 2). 
    Assert(0);
    return 0;
}

internal face_index
SplitFaceMakeEdge(world_mesh *Mesh, face_index Face, vert_index V1, vert_index V2)
{
    // TODO(Kevin): 
    Assert(0);
    return 0;
}

internal face_index
JoinFaceKillEdge()
{
    // TODO(Kevin): 
    Assert(0);
    return 0;
}

internal void
ReverseFaceLoop(world_mesh *Mesh, face_index Face)
{
    mesh_face *F = &MeshFace(Face);
    Assert(F->LoopCycleLen > 2);
    Assert(F->LoopCycleBase);

    vert_index BaseVert = MeshLoop(F->LoopCycleBase).Vert;
    loop_index CurrentLoop = F->LoopCycleBase;
    do
    {
        MeshLoop(CurrentLoop).Vert = MeshLoop(MeshLoop(CurrentLoop).Next).Vert;
        loop_index NextLoop = MeshLoop(CurrentLoop).Next;
        MeshLoop(CurrentLoop).Next = MeshLoop(CurrentLoop).Prev;
        MeshLoop(CurrentLoop).Prev = NextLoop;

        if (CurrentLoop == MeshLoop(F->LoopCycleBase).Next)
        {
            MeshLoop(CurrentLoop).Vert = BaseVert;
        }

        CurrentLoop = MeshLoop(CurrentLoop).Prev;
    }
    while (CurrentLoop != F->LoopCycleBase);
}


//
//
// =======================================================
//
//

internal vert_index
ExtrudeEdgesHelper_DuplicateVertex(world_mesh *Mesh, vert_index OriginalVertex,
                                   vert_index **VisitedVertices, u32 *VisitedVerticesCount,
                                   vert_index **CreatedVertices, u32 *CreatedVerticesCount)
{
    for (u32 Check = 0; Check < *VisitedVerticesCount; ++Check)
    {
        if ((*VisitedVertices)[Check] == OriginalVertex)
        {
            return (*CreatedVertices)[Check];
        }
    }

    vert_index DuplicatedVertex = CreateVert(Mesh, MeshVert(OriginalVertex).P);

    (*CreatedVertices)[(*CreatedVerticesCount)++] = DuplicatedVertex;
    (*VisitedVertices)[(*VisitedVerticesCount)++] = OriginalVertex;

    return DuplicatedVertex;
}

internal edge_index
ExtrudeEdgesHelper_CreateEdge(world_mesh *Mesh, vert_index A, vert_index B,
                              edge_index **CreatedEdges, u32 *CreatedEdgesCount)
{
    for (u32 Check = 0; Check < *CreatedEdgesCount; ++Check)
    {
        if (HasVert(Mesh, (*CreatedEdges)[Check], A) &&
            HasVert(Mesh, (*CreatedEdges)[Check], B))
        {
            return (*CreatedEdges)[Check];
        }
    }

    edge_index NewEdge = CreateEdge(Mesh, A, B);

    (*CreatedEdges)[(*CreatedEdgesCount)++] = NewEdge;

    return NewEdge;
}

internal void
ExtrudeEdges(world_mesh *Mesh, edge_index *Edges, u32 EdgesCount,
             freeable_loi *OutCreatedVerts,
             freeable_loi *OutCreatedEdges,
             freeable_loi *OutCreatedFaces)
{
    /*  NOTE(Kevin): Take an array of edges, not necessarily connected.
                     Duplicate an edge, form edge between new and old vertices.
                     Use the 4 edges to form a new face.
    */

    if (OutCreatedVerts->Freeable.Block != 0)
    {
        FreeBlock(OutCreatedVerts->Freeable.Allocator, &OutCreatedVerts->Freeable.Block);
        // TODO(Kevin): Logging warning
    }
    if (OutCreatedEdges->Freeable.Block != 0)
    {
        FreeBlock(OutCreatedEdges->Freeable.Allocator, &OutCreatedEdges->Freeable.Block);
        // TODO(Kevin): Logging warning
    }
    if (OutCreatedFaces->Freeable.Block != 0)
    {
        FreeBlock(OutCreatedFaces->Freeable.Allocator, &OutCreatedFaces->Freeable.Block);
        // TODO(Kevin): Logging warning
    }

    Assert(OutCreatedFaces && OutCreatedFaces->Freeable.Block == 0);
    OutCreatedFaces->Indices = AllocArray(Mesh->Memory, EdgesCount, face_index);
    OutCreatedFaces->Count = 0;
    OutCreatedFaces->Freeable.Block = OutCreatedFaces->Indices;
    OutCreatedFaces->Freeable.Allocator = Mesh->Memory;

    vert_index *VisitedVertices = AllocArray(Mesh->Memory, EdgesCount*2, vert_index);
    u32 VisitedVerticesCount = 0;
    vert_index *CreatedVertices = AllocArray(Mesh->Memory, EdgesCount*2, vert_index);
    u32 CreatedVerticesCount = 0;
    edge_index *CreatedEdges = AllocArray(Mesh->Memory, EdgesCount*2, edge_index);
    u32 CreatedEdgesCount = 0;

    for (u32 Index = 0; Index < EdgesCount; ++Index)
    {
        edge_index EdgeAB = Edges[Index];
        mesh_edge Edge = MeshEdge(EdgeAB);

        vert_index DupA = ExtrudeEdgesHelper_DuplicateVertex(Mesh, Edge.A, &VisitedVertices, 
                                                             &VisitedVerticesCount, &CreatedVertices,
                                                             &CreatedVerticesCount);
        vert_index DupB = ExtrudeEdgesHelper_DuplicateVertex(Mesh, Edge.B, &VisitedVertices, 
                                                             &VisitedVerticesCount, &CreatedVertices,
                                                             &CreatedVerticesCount);
        edge_index EdgeAA = ExtrudeEdgesHelper_CreateEdge(Mesh, Edge.A, DupA, 
                                                          &CreatedEdges, &CreatedEdgesCount);
        edge_index EdgeBB = ExtrudeEdgesHelper_CreateEdge(Mesh, Edge.B, DupB,
                                                          &CreatedEdges, &CreatedEdgesCount);
        edge_index EdgeBA = CreateEdge(Mesh, DupB, DupA);

        edge_index FaceEdges[] = { EdgeAB, EdgeBB, EdgeBA, EdgeAA };
        face_index Face = CreateFace(Mesh, FaceEdges, ArrayCount(FaceEdges));

        OutCreatedFaces->Indices[OutCreatedFaces->Count++] = Face;
    }

    FreeBlock(Mesh->Memory, &VisitedVertices);

    OutCreatedVerts->Indices = CreatedVertices;
    OutCreatedVerts->Count = CreatedVerticesCount;
    OutCreatedVerts->Freeable.Block = CreatedVertices;
    OutCreatedVerts->Freeable.Allocator = Mesh->Memory;

    OutCreatedEdges->Indices = CreatedEdges;
    OutCreatedEdges->Count = CreatedEdgesCount;
    OutCreatedEdges->Freeable.Block = CreatedEdges;
    OutCreatedEdges->Freeable.Allocator = Mesh->Memory;
}

internal vec3
GetFaceNormal(vec3 *Points, u32 PointCount)
{
    vec3 FaceNormal = {};
    real32 Len = 0;

    for (u32 i = 0; ZeroWithEpsilon1(Len) == 0 && i < PointCount; ++i)
    {
        u32 A = i;
        u32 B = (i+1)%PointCount;
        u32 C = (i+2)%PointCount;
        FaceNormal = Normalize(Cross(Points[B]-Points[A], Points[C]-Points[A]));
        Len = Length(FaceNormal);
    }

    return FaceNormal;
}

internal vec3
GetFaceNormal(world_mesh *Mesh, face_index FaceIndex)
{
    mesh_face Face = MeshFace(FaceIndex);

    vec3 *Points = AllocArray(Mesh->Memory, Face.LoopCycleLen, vec3);
    loop_index LoopIndex = Face.LoopCycleBase;
    for (u32 I = 0; I < Face.LoopCycleLen; ++I)
    {
        topo_loop Loop = MeshLoop(LoopIndex);
        Points[I] = MeshVert(Loop.Vert).P;
        LoopIndex = Loop.Next;
    }

    vec3 Result = GetFaceNormal(Points, Face.LoopCycleLen);

    FreeBlock(Mesh->Memory, &Points);

    return Result;
}

internal void
EarClippingTriangulationCCW(memory_blocks *Memory, vec3 *Points, u32 PointCount,
                            u32 **Triangles, u32 *IndicesCount)
{
    vec3 FaceNormal = GetFaceNormal(Points, PointCount);
    if (ZeroWithEpsilon1(Length(FaceNormal)) == 0)
    {
        // TODO(Kevin): Logging - the triangle is degenerate
        *Triangles = 0;
        *IndicesCount = 0;
    }

    u32 *TriangleIndices = AllocArray(Memory, (PointCount-2)*3, u32);
    u32 TriangleIndicesCount = 0;

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

            real32 SignedArea = ZeroWithEpsilon1(Cross(Edge0, Edge1));
            bool32 IsConvex = SignedArea > 0; // NOTE(Kevin): Left turn means convex in CCW

            bool32 IsEar = false;
            if (IsConvex)
            {
                for (u32 Check = 0; Check < Points2DCount; ++Check)
                {
                    vec2 CheckP = Points2D[Check].P;

                    if (Check != PrevIndex && Check != ThisIndex && Check != NextIndex &&
                        // NOTE(Kevin): If points overlap, there must be a degenerate
                        // triangle. Ignore these points during ear check.
                        CheckP != Points2D[PrevIndex].P &&
                        CheckP != Points2D[ThisIndex].P &&
                        CheckP != Points2D[NextIndex].P)
                    {
                        real32 Sign0 = ZeroWithEpsilon1(Cross(Edge0, CheckP - PrevPoint.P));
                        real32 Sign1 = ZeroWithEpsilon1(Cross(Edge1, CheckP - ThisPoint.P));
                        real32 Sign2 = ZeroWithEpsilon1(Cross(Edge2, CheckP - NextPoint.P));
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
                // NOTE(Kevin): There were no ears found after a certain point.
                // Could be that parts of the face results in degenerate triangles.
                // Return the triangles I have generated thus far.
                Points2DCount = 0;
                break;
            }
        }
    }

    if (Points2DCount == 3)
    {
        TriangleIndices[TriangleIndicesCount++] = Points2D[0].Index;
        TriangleIndices[TriangleIndicesCount++] = Points2D[1].Index;
        TriangleIndices[TriangleIndicesCount++] = Points2D[2].Index;
    }

    *Triangles = TriangleIndices;
    *IndicesCount = TriangleIndicesCount;

    FreeBlock(Memory, &Points2D);
}

internal vertex_buffer
TriangulateEditMeshFace(memory_blocks *Memory, world_mesh *World, face_index Face)
{
    vertex_buffer Result = {};

    u32 PointCount = World->Faces[Face].LoopCycleLen;

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

        if (IndicesCount > 2)
        {
            vec3 WorldN = GetFaceNormal(Points, PointCount);
            Assert(ZeroWithEpsilon1(Length(WorldN)) > 0);

            for (u32 PointIndex = 0; PointIndex < PointCount; ++PointIndex)
            {
                vec3 WorldP = Points[PointIndex];
                vec2 LMUV = LMUVs[PointIndex];

                vec2 AlbedoUV = {};

                // TODO(Kevin): Better UVs with basis vectors?
                real32 UVScale = (1.f / (real32)STANDARD_LENGTH_IN_GAME_UNITS); // [0,1] scaled to 32 units
                real32 XF = AbsoluteValue(Dot(WorldN, vec3(1, 0, 0)));
                real32 YF = AbsoluteValue(Dot(WorldN, vec3(0, 1, 0)));
                real32 ZF = AbsoluteValue(Dot(WorldN, vec3(0, 0, 1)));
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
        }

        Result.VB = Vertices;
        Result.VC = VertexCount;
        Result.IB = Triangles;
        Result.IC = IndicesCount;

        FreeBlock(Memory, &Points);
        FreeBlock(Memory, &LMUVs);
    }

    return Result;
}

#undef MeshVert
#undef MeshEdge
#undef MeshFace
#undef MeshLoop
#undef ZeroWithEpsilon1
#undef ZeroWithEpsilon2

