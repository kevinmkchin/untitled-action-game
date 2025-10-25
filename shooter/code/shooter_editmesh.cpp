#include "shooter_editmesh.h"

internal void
InitializeWorldMesh(world_mesh *Mesh)
{
    // NOTE(Kevin): face,edge,vert,loop indices 0 are reserved for invalid elements.

    Mesh->FreeFaceHandlesCount = ArrayCount(Mesh->FreeFaceHandles);
    PopulateFreeHandles(Mesh->FreeFaceHandles, Mesh->FreeFaceHandlesCount--);

    Mesh->FreeEdgeHandlesCount = ArrayCount(Mesh->FreeEdgeHandles);
    PopulateFreeHandles(Mesh->FreeEdgeHandles, Mesh->FreeEdgeHandlesCount--);

    Mesh->FreeVertexHandlesCount = ArrayCount(Mesh->FreeVertexHandles);
    PopulateFreeHandles(Mesh->FreeVertexHandles, Mesh->FreeVertexHandlesCount--);

    Mesh->FreeTopoLoopHandlesCount = ArrayCount(Mesh->FreeTopoLoopHandles);
    PopulateFreeHandles(Mesh->FreeTopoLoopHandles, Mesh->FreeTopoLoopHandlesCount--);
}

internal void
RadialCycleInsertLoop(world_mesh *Mesh, edge_index Edge, loop_index ToInsert)
{
    // TODO(Kevin): 
}

internal void
RadialCycleRemoveLoop(world_mesh *Mesh, loop_index ToRemove)
{
    // TODO(Kevin): 
}

internal void
DiskCycleInsertEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToInsert)
{
    // TODO(Kevin): 
}

internal void
DiskCycleRemoveEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToRemove)
{
    // TODO(Kevin): 
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

    DiskCycleInsertEdge(Mesh, V1, Result);
    DiskCycleInsertEdge(Mesh, V2, Result);

    return Result;
}

internal void
KillEdge(world_mesh *Mesh, edge_index Edge)
{
    // TODO(Kevin): remove edge from vertices disk link
}

internal face_index
CreateFace(world_mesh *Mesh, edge_index *Edges, u32 EdgeCount)
{
    if (EdgeCount < 3)
    {
        return 0; // TODO(Kevin): Should I reserve 0 as invalid?
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

        loop_index Loop = Mesh->FreeTopoLoopHandles[--Mesh->FreeTopoLoopHandlesCount];
        vert_index SharedVert = LastLoop ? SharesVert(Mesh, Mesh->Loops[LastLoop].Edge, Edge)
                                         : SharesVert(Mesh, OrderedEdges[OrderedEdgesCount-1], Edge);
        Assert(SharedVert);
        // Store references to base vertex, corresponding edge, and corresponding face
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

    Mesh->Faces[Result].LoopCycleBase = FirstLoop;
    Mesh->Faces[Result].LoopCycleLength = EdgeCount;

    FreeBlock(Mesh->Memory, UnprocessedEdges);
    FreeBlock(Mesh->Memory, OrderedEdges);

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

        Current = Mesh->Loops[Current].Next;
    }
    while (Current != Head);

    Mesh->Faces[Face] = {};
}

// mesh_vert *SplitEdgeMakeVert(edit_mesh *Mesh, mesh_edge *Edge, float T = 0.5f);
// mesh_edge *JoinEdgeKillVert(edit_mesh *Mesh, );
// mesh_face *SplitFaceMakeEdge(edit_mesh *Mesh, mesh_face *Face, mesh_vert *V1, mesh_vert *V2);
// mesh_face *JoinFaceKillEdge(edit_mesh *Mesh, );

