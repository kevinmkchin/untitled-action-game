#ifndef SHOOTER_EDITMESH_H
#define SHOOTER_EDITMESH_H

#include "shooter_containers.h"

typedef i32 bool32;

struct mesh_edge;
struct mesh_vert;
struct mesh_face;
struct topo_loop;

struct extra_vertex_data
{
    vec2 lmuvcache;
    vec3 poscache;
};

// NOTE(Kevin): The connections b/w topologies entities (edges, verts, faces) are
// defined by series of loops (cycles) around edges and faces.
// NOTE(Kevin): "Loop Cycle" is a series of face edges around a polygon. Implicitly
// stores face direction.

struct topo_loop
{
    topo_loop *Next; // CCW order
    topo_loop *Prev;
    topo_loop *RadialNext;
    topo_loop *RadialPrev;

    mesh_edge *Edge;
    mesh_vert *Vert;
    mesh_face *Face;

    extra_vertex_data PerFaceVertexData;
};

struct mesh_face
{
    topo_loop *LoopCycleBase;
    u32 LoopCycleLength;

    // u32 ElementId;
};

struct mesh_edge
{
    mesh_vert *A;
    mesh_vert *B;

    topo_loop *RadialCycleBase;

    // u32 ElementId;
};

struct mesh_vert
{
    vec3 P;

    carray<mesh_edge *, 16> ConnectedEdges;

    // u32 ElementId;

    extra_vertex_data EditorData;
};

void RecountLoopLength(mesh_face *Face);
void GetLoopCycle(mesh_face *Face, mesh_loop **LoopWriteTo, u32 LoopMaxCount, u32 *LoopLength);
void GetFaceVertices(mesh_face *Face, mesh_vert **VerticesWriteTo, u32 VerticesMaxCount, u32 *VerticesCount);
void GetFaceEdges(mesh_face *Face, mesh_edge **EdgesWriteTo, u32 EdgesMaxCount, u32 *EdgesCount);
vec3 GetFaceNormal(mesh_face *Face);

void RadialCycleInsertLoop(mesh_edge *Edge, Loop *ToInsert);
void RadialCycleRemoveLoop(mesh_edge *Edge, Loop *ToRemove);
mesh_vert *SharesVert(mesh_edge *A, mesh_edge *B);
void GetRadialCycle(mesh_edge *Edge, mesh_loop **RadialWriteTo, u32 RadialMaxCount, u32 *RadialLength);

void DiskCycleInsertEdge(mesh_edge *Edge);
void DiskCycleRemoveEdge(mesh_edge *Edge);

// TODO(Kevin): I can just assign element IDs during serialization!
// u32 SessionElementID();

/*

e.g. editor state that we want restored by undo:
    manual_heap for topo entities e.g. 64mb
    edit_mesh e.g. 1mb
    selection info
    etc.

blat out entire state struct to memory/disk

*/

struct edit_mesh
{
    // NOTE(Kevin): Use this allocator for all topological entities
    manual_heap *MeshAllocator;
    // NOTE(Kevin): Entries must be contiguous - no holes!
    carray<mesh_face *, 10000> Faces;
    carray<mesh_edge *, 30000> Edges;
    carray<mesh_vert *, 40000> Vertices;
};

//
// Euler/atomic operators
//
mesh_face *CreateFace(edit_mesh *Mesh, mesh_edge **Edges, u32 EdgeCount);
void       KillFace(edit_mesh *Mesh, mesh_face *Face);
mesh_edge *CreateEdge(edit_mesh *Mesh, mesh_vert *V1, mesh_vert *V2);
void       KillEdge(edit_mesh *Mesh, mesh_edge *Edge);
mesh_vert *CreateVert(edit_mesh *Mesh, vec3 P);
void       KillVert(edit_mesh *Mesh, mesh_vert *Vert);
mesh_vert *SplitEdgeMakeVert(edit_mesh *Mesh, mesh_edge *Edge, float T = 0.5f);
// mesh_edge *JoinEdgeKillVert(edit_mesh *Mesh, );
mesh_face *SplitFaceMakeEdge(edit_mesh *Mesh, mesh_face *Face, mesh_vert *V1, mesh_vert *V2);
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

//
// Serialization
//
struct mesh_serializer_state
{
    // when deserializing, maps previous session elemIds to current session pointers
    // std::unordered_map<u32, void*> DeserElemIdToElem;
    // ByteBuffer Buffer;
    bool32 IsReading;
};

void SerializeVert(mesh_vert *Element, mesh_serializer_state State);
void SerializeEdge(mesh_edge *Element, mesh_serializer_state State);
void SerializeFace(mesh_face *Element, mesh_serializer_state State);

#endif // SHOOTER_EDITMESH_H
