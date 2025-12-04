#ifndef SHOOTER_EDITMESH_H
#define SHOOTER_EDITMESH_H

/*
    Editable Mesh API

    NOTE(Kevin): The connections b/w topologies entities (edges, verts, faces) are
                 defined by series of loops (cycles) around edges and faces.
    NOTE(Kevin): "Loop Cycle" is a series of face edges around a polygon. Implicitly
                 stores face direction.
*/

typedef u32 loop_index;
typedef u32 vert_index;
typedef u32 edge_index;
typedef u32 face_index;

struct topo_loop
{
    loop_index Next; // CCW order
    loop_index Prev;
    loop_index RadialNext;
    loop_index RadialPrev;

    edge_index Edge;
    vert_index Vert;
    face_index Face;

    // Per face vertex data
    vec2 LightmapUV;
};

struct mesh_face
{
    loop_index LoopCycleBase;
    u32 LoopCycleLen;

    // Editor helper data
    bool32 Alive;
    bool32 Dirty;
    mesh_handle RenderMesh;
};

struct mesh_edge
{
    vert_index A;
    vert_index B;
    loop_index RadialCycleBase;
    u32 RadialCycleLen;

    // Editor helper data
    bool32 Alive;
};

struct mesh_vert
{
    vec3 P;

    // TODO(Kevin): Conserve memory by making the "disk link" a linked list
    //              of the edges themselves!
    edge_index ConnectedEdges[8];
    u32 ConnectedEdgesCount;

    // Editor helper data?

};

struct world_mesh
{
    /*
    NOTE(Kevin): Aiming for ~10MB size. Then, even if I kept a ring buffer 
                 of 100 of these in memory, it would only be 1GB.
    */

    memory_blocks *Memory;

    mesh_face Faces[10000];
    face_index FreeFaceHandles[10000];
    u32 FreeFaceHandlesCount;

    mesh_edge Edges[40000];
    edge_index FreeEdgeHandles[40000];
    u32 FreeEdgeHandlesCount;

    mesh_vert Vertices[40000];
    vert_index FreeVertexHandles[40000];
    u32 FreeVertexHandlesCount;

    topo_loop Loops[80000];
    loop_index FreeTopoLoopHandles[80000];
    u32 FreeTopoLoopHandlesCount;
};

internal void
InitializeWorldMesh(world_mesh *Mesh);

internal void
RadialCycleInsertLoop(world_mesh *Mesh, edge_index Edge, loop_index ToInsert);
internal void
RadialCycleRemoveLoop(world_mesh *Mesh, edge_index Edge, loop_index ToRemove);

internal void
DiskCycleInsertEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToInsert);
internal void
DiskCycleRemoveEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToRemove);

internal bool32
HasVert(world_mesh *Mesh, edge_index E, vert_index V);
internal vert_index
SharesVert(world_mesh *Mesh, edge_index A, edge_index B);

//
// Euler/atomic operators
//
internal vert_index
CreateVert(world_mesh *Mesh, vec3 P);
internal void
KillVert(world_mesh *Mesh, vert_index Vertex);

internal edge_index
CreateEdge(world_mesh *Mesh, vert_index V1, vert_index V2);
internal void
KillEdge(world_mesh *Mesh, edge_index Edge);

internal face_index
CreateFace(world_mesh *Mesh, edge_index *Edges, u32 EdgeCount, vert_index HintLoopStart = 0);
internal void
KillFace(world_mesh *Mesh, face_index Face);

internal vert_index
SplitEdgeMakeVert(world_mesh *Mesh, edge_index Edge, float t = 0.5f);
internal edge_index
JoinEdgeKillVert(world_mesh *Mesh, edge_index Edge, vert_index Vert);
internal face_index
SplitFaceMakeEdge(world_mesh *Mesh, face_index Face, vert_index V1, vert_index V2);
internal face_index
JoinFaceKillEdge(/* TODO */);

internal void
ReverseFaceLoop(world_mesh *Mesh, face_index Face);


/*

    High-level Mesh Editing API

*/

struct freeable_loi // list of indices
{
    freeable_block Freeable;
    u32 *List;
    u32 Count;
    // Capacity
};

inline void
AllocListOfIndices(memory_blocks *Memory, freeable_loi *FreeableList, u32 Count)
{
    Assert(FreeableList && FreeableList->Freeable.Block == 0);
    FreeableList->List = AllocArray(Memory, Count, u32);
    FreeableList->Count = 0;
    FreeableList->Freeable.Block = FreeableList->List;
    FreeableList->Freeable.Allocator = Memory;
}

inline void
FreeListOfIndices(freeable_loi *FreeableList)
{
    FreeFreeableBlock(&FreeableList->Freeable);
    *FreeableList = {};
}

inline s32
GetIndexOf(u32 *Indices, u32 IndicesCount, u32 SearchFor)
{
    s32 Index = -1;

    for (u32 Check = 0; Check < IndicesCount; ++Check)
    {
        if (Indices[Check] == SearchFor)
        {
            Index = Check;
            break;
        }
    }

    return Index;
}

inline bool32
AddIndexIfUnique(u32 *Indices, u32 *IndicesCount, u32 Index, u32 *OutExistingIndex = 0)
{
    s32 ExistingIndex = GetIndexOf(Indices, *IndicesCount, Index);

    bool32 IsUnique = ExistingIndex < 0;

    if (IsUnique)
    {
        Indices[(*IndicesCount)++] = Index;
    }
    else
    {
        if (OutExistingIndex)
        {
            *OutExistingIndex = SafeCastToU32(ExistingIndex);
        }        
    }

    return IsUnique;
}

inline void
RemoveIndexAtSwapWithEnd(u32 *Indices, u32 *IndicesCount, u32 Index)
{
    Indices[Index] = Indices[--(*IndicesCount)];
}

inline u32
AddFaceLoopCycleLengths(world_mesh *Mesh, face_index *Faces, u32 FacesCount)
{
    u32 TotalLoopCount = 0;

    for (u32 f = 0; f < FacesCount; ++f)
    {
        face_index FaceIndex = Faces[f];
        mesh_face *Face = &Mesh->Faces[FaceIndex];
        TotalLoopCount += Face->LoopCycleLen;
    }

    return TotalLoopCount;
}

inline void
GetFaceVertsAndEdges(world_mesh *Mesh, face_index FaceIndex,
                     freeable_loi *OutVerts, freeable_loi *OutEdges)
{
    mesh_face Face = Mesh->Faces[FaceIndex];

    for (loop_index l = Face.LoopCycleBase, i = 0;
         i < Face.LoopCycleLen;
         l = Mesh->Loops[l].Next, ++i)
    {
        topo_loop Loop = Mesh->Loops[l];

        if (OutVerts)
        {
            AddIndexIfUnique(OutVerts->List, &OutVerts->Count, Loop.Vert);
        }
        if (OutEdges)
        {
            AddIndexIfUnique(OutEdges->List, &OutEdges->Count, Loop.Edge);
        }
    }
}

internal void
ExtrudeEdges(world_mesh *Mesh, edge_index *Edges, u32 EdgesCount, 
             freeable_loi *OutCreatedVerts,
             freeable_loi *OutCreatedFaces,
             freeable_loi *OutOppositeEdges);
/* Extrude faces from edges.

    $:OutCreatedVerts   the new vertices created by the operation
    $:OutCreatedFaces   the new faces created by the operation
    $:OutOppositeEdges  (optional) the newly created edges that are not connected to the input edges, opposite from them in the created rects. Memory allocated by the operation.

    Input edges that are connected/adjacent will respect sharing of vertices.
        .-----.-----.  => :=====:=====:

*/

internal void
ExtrudeFaces(world_mesh *Mesh, face_index *InputFaces, u32 InputFacesCount,
             freeable_loi *OutCreatedFaces,
             freeable_loi *OutOppositeFaces);

// RingCut / Slice

internal vec3
GetFaceNormal(vec3 *Points, u32 PointCount);
internal vec3
GetFaceNormal(world_mesh *Mesh, face_index Face);

internal void
EarClippingTriangulationCCW(memory_blocks *Memory, vec3 *Points, u32 PointCount,
                            u32 **Triangles, u32 *IndicesCount);

struct vertex_buffer
{
    PNUV2_vertex *VB;
    u32           VC;
    u32          *IB;
    u32           IC;
};

/* Allocates memory for VB and IB and fills them with triangles for the face */
internal vertex_buffer
TriangulateEditMeshFace(memory_blocks *Memory, world_mesh *World, face_index Face);


/*

Keep it simple! For undo/redo just serialize the entire editor/mesh state to disk!
Or keep in memory in ring buffer

blat out entire state struct to memory/disk -> if small enough to blat out in
reasonable time, then probably small enough to just keep a bunch in a ring buffer
in memory

*/

#endif // SHOOTER_EDITMESH_H
