#pragma once

/* https://developer.blender.org/docs/features/objects/mesh/bmesh

Persistent adjacency information is stored using DOUBLE LINKED CIRCULAR LISTS which
maintain the relationships among topological entities.

The connections between the elements are defined by loops around topological entities, 
referred to as cycles. The base of each cycle is the entity which the cycle is designed 
to answer adjacency queries for. For instance a cycle designed to answer the question, 
“what edges share this vertex” would have the vertex itself as the base and its edges 
as the nodes of the cycle. Note that it is not required to explicitly store all possible
adjacency relationships and full connectivity information can be quickly derived from 
using two or more cycles in conjunction.

*/

/*  Example usage:

    v0 = CreateVert(...)
    v1 = CreateVert(...)
    v2 = CreateVert(...)
    v3 = CreateVert(...)
    v4 = CreateVert(...)

    e0 = CreateEdge(v0, v1)
    e1 = CreateEdge(v1, v2)
    e2 = CreateEdge(v3, v2) // order of vertices must not matter
    e3 = CreateEdge(v3, v4)
    e4 = CreateEdge(v4, v0)

    loe = [e0, e1, e2, e3, e4]

    f0 = CreateFace(loe)

    v5 = CreateVert(...)

    e5 = CreateEdge(v5, v1)
    e6 = CreateEdge(v5, v2)

    loe = [e1, e5, e6]

    f1 = CreateFace(loe)

    This would create a mesh structure with two faces where e1 is the shared edge connecting the two faces.
    e1 would also have two loops in its radial cycle.

    The topological entities created from these functions are only partially populated. Vertices are only
    fully populated once they are used to create Edges, and Edges are only fully populated once they are
    used to create Faces.
*/

namespace MapEdit
{
    struct Edge;
    struct Vert;
    struct Loop;
    struct Face;

    // TODO(Kevin): make this an implicit array using stb_ds
    extern NiceArray<Face*, 100000> LevelEditorFaces;

    struct Loop
    {
        Loop *loopNext = NULL; // CCW order
        Loop *loopPrev = NULL;
        Loop *radialNext = NULL;
        Loop *radialPrev = NULL;
        Edge *e = NULL;
        Vert *v = NULL;
        Face *f = NULL;

        // per face vertex data
        // Verts are shared between faces, so vertex data that is unique to the face
        // can be put in here. e.g. UVs 
        vec2 lmuvcache;
    };

    struct Face
    {
        Loop *loopbase = NULL;
        int looplen = -1;

        u32 elemId = 0;

        // == level editor data ==
        GPUMesh facemesh; // For map editor rendering. Each face is its own mesh. Don't optimize until my PC chugs while map editing.
        db_tex_t texture;
        lm_face_t lightmap;
        // bool receivelight; mark whether to generate light map
        // bool hascollision; mark whether to generate collider
        i32 storageIndex = -1;
        bool hovered = false;
        // =====================


        int RefreshLoopLen();

        std::vector<Loop*> GetLoopCycle() const;

        std::vector<Edge*> GetEdges() const;

        std::vector<Vert*> GetVertices() const;

        vec3 QuickNormal();
    };

    struct Edge
    {
        Vert *a = NULL;
        Vert *b = NULL;
        Loop *radial = NULL;

        u32 elemId = 0;


        void InsertIntoRadialCycle(Loop *loop);

        void RemoveFromRadialCycle(Loop *loop);

        Vert *SharesVert(Edge *other) const;

        std::vector<Loop*> GetRadialCycleCopy() const;
    };

    struct Vert
    {
        vec3 pos;

        std::vector<Edge*> edges;

        u32 elemId = 0;
        // editor runtime data
        vec3 poscache;

        void DiskLinkInsertEdge(Edge *e);

        void DiskLinkRemoveEdge(Edge *e);
    };

    struct Volume
    {
        std::vector<Face*> faces;
        std::vector<Edge*> edges;
        std::vector<Vert*> verts;

        u64 persistId = 0; // 0 is invalid volume. All volumes should call FreshVolumePersistId.
    };

    extern u64 session_VolumePersistIdCounter;
    u64 FreshVolumePersistId();

    u32 FreshElemId();

    Face *CreateFace(const std::vector<Edge*>& edges, Volume *owner);
    Edge *CreateEdge(Vert *v1, Vert *v2, Volume *owner);
    Vert *CreateVert(vec3 pos, Volume *owner);

    void KillFace(Face *face, Volume *owner);
    void KillEdge(Edge *edge, Volume *owner);
    void KillVert(Vert *vert, Volume *owner);

    Vert *Euler_SplitEdgeMakeVert(Edge *edge, float abLerp = 0.5f);
    Edge *Euler_JoinEdgeKillVert();
    Face *Euler_SplitFaceMakeEdge(Face *face, Vert *v1, Vert *v2);
    Face *Euler_JoinFaceKillEdge();

    void FaceLoopReverse(Face *face);

    void MakeRectangularVolume(Volume *vol);

    void MakeCubeVolume(Volume *vol);


    void TriangulateFace_QuickDumb_WithColor(const Face f, vec3 color, float *vbdata, int *out_count);
    void TriangulateFace_QuickDumb(const Face f, std::vector<float> *vb);
    void TriangulateFace_ForFaceBatch_QuickDumb(const Face f, std::vector<float> *vb);
}