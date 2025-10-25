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
    u32 LoopCycleLength;
};

struct mesh_edge
{
    vert_index A;
    vert_index B;
    loop_index RadialCycleBase;
};

struct mesh_vert
{
    vec3 P;

    edge_index ConnectedEdges[8];

    // Editor helper data
    // poscache
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

    mesh_edge Edges[30000];
    edge_index FreeEdgeHandles[30000];
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
RadialCycleRemoveLoop(world_mesh *Mesh, loop_index ToRemove);

internal void
DiskCycleInsertEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToInsert);
internal void
DiskCycleRemoveEdge(world_mesh *Mesh, vert_index DiskCycle, edge_index ToRemove);

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
CreateFace(world_mesh *Mesh, edge_index *Edges, u32 EdgeCount);
internal void
KillFace(world_mesh *Mesh, face_index Face);

// mesh_vert *SplitEdgeMakeVert(edit_mesh *Mesh, mesh_edge *Edge, float T = 0.5f);
// mesh_edge *JoinEdgeKillVert(edit_mesh *Mesh, );
// mesh_face *SplitFaceMakeEdge(edit_mesh *Mesh, mesh_face *Face, mesh_vert *V1, mesh_vert *V2);
// mesh_face *JoinFaceKillEdge(edit_mesh *Mesh, );




/*

e.g. editor state that we want restored by undo:
    manual_heap for topo entities e.g. 64mb
    edit_mesh e.g. 1mb
    selection info
    etc.

blat out entire state struct to memory/disk -> if small enough to blat out in reasonable time, then probably
small enough to just keep a bunch in a ring buffer in memory

*/





// void RecountLoopLength(mesh_face *Face);
// void GetLoopCycle(mesh_face *Face, mesh_loop **LoopWriteTo, u32 LoopMaxCount, u32 *LoopLength);
// void GetFaceVertices(mesh_face *Face, mesh_vert **VerticesWriteTo, u32 VerticesMaxCount, u32 *VerticesCount);
// void GetFaceEdges(mesh_face *Face, mesh_edge **EdgesWriteTo, u32 EdgesMaxCount, u32 *EdgesCount);
// vec3 GetFaceNormal(mesh_face *Face);

// void RadialCycleInsertLoop(mesh_edge *Edge, Loop *ToInsert);
// void RadialCycleRemoveLoop(mesh_edge *Edge, Loop *ToRemove);
// mesh_vert *SharesVert(mesh_edge *A, mesh_edge *B);
// void GetRadialCycle(mesh_edge *Edge, mesh_loop **RadialWriteTo, u32 RadialMaxCount, u32 *RadialLength);

// void DiskCycleInsertEdge(mesh_edge *Edge);
// void DiskCycleRemoveEdge(mesh_edge *Edge);

// TODO(Kevin): I can just assign element IDs during serialization!
// u32 SessionElementID();




//
// Euler/atomic operators
//
// mesh_face *CreateFace(edit_mesh *Mesh, mesh_edge **Edges, u32 EdgeCount);
// void       KillFace(edit_mesh *Mesh, mesh_face *Face);
// mesh_edge *CreateEdge(edit_mesh *Mesh, mesh_vert *V1, mesh_vert *V2);
// void       KillEdge(edit_mesh *Mesh, mesh_edge *Edge);
// mesh_vert *CreateVert(edit_mesh *Mesh, vec3 P);
// void       KillVert(edit_mesh *Mesh, mesh_vert *Vert);
// mesh_vert *SplitEdgeMakeVert(edit_mesh *Mesh, mesh_edge *Edge, float T = 0.5f);
// // mesh_edge *JoinEdgeKillVert(edit_mesh *Mesh, );
// mesh_face *SplitFaceMakeEdge(edit_mesh *Mesh, mesh_face *Face, mesh_vert *V1, mesh_vert *V2);
// mesh_face *JoinFaceKillEdge(edit_mesh *Mesh, );

//
// Helpers
//
// void MakeRectangularVolume(Volume *vol);
// void MakeCubeVolume(Volume *vol);
// void FaceLoopReverse(Face *face);
// void TriangulateFace_QuickDumb_WithColor(const Face f, vec3 color, float *vbdata, int *out_count);
// void TriangulateFace_QuickDumb(const Face f, std::vector<float> *vb);
// void TriangulateFace_ForFaceBatch_QuickDumb(const Face f, std::vector<float> *vb);

// Keep it simple! For undo/redo just serialize the entire editor/mesh state to disk!
// Or keep in memory in ring buffer

// //
// // Serialization
// //
// struct mesh_serializer_state
// {
//     // when deserializing, maps previous session elemIds to current session pointers
//     // std::unordered_map<u32, void*> DeserElemIdToElem;
//     // ByteBuffer Buffer;
//     bool32 IsReading;
// };

// void SerializeVert(mesh_vert *Element, mesh_serializer_state State);
// void SerializeEdge(mesh_edge *Element, mesh_serializer_state State);
// void SerializeFace(mesh_face *Element, mesh_serializer_state State);

#endif // SHOOTER_EDITMESH_H
