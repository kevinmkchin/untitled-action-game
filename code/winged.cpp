#include "winged.h"

namespace MapEdit
{
    NiceArray<Face*, 100000> LevelEditorFaces;

    int Face::RefreshLoopLen()
    {
        looplen = 0;
        Loop *l = loopbase;
        while (l != NULL && l != loopbase->loopPrev)
        {
            ++looplen;
            l = l->loopNext;
        }
        return looplen;
    }

    std::vector<Loop*> Face::GetLoopCycle() const
    {
        ASSERT(loopbase);
        std::vector<Loop*> loopcycle;
        Loop *l = loopbase;
        do {
            loopcycle.push_back(l);
            l = l->loopNext;
        } while (l != loopbase && l != NULL);
        return loopcycle;
    }

    std::vector<Edge*> Face::GetEdges() const
    {
        ASSERT(loopbase);
        std::vector<Edge*> edgesInOrder;
        Loop *l = loopbase;
        do {
            edgesInOrder.push_back(l->e);
            l = l->loopNext;
        } while (l != loopbase && l != NULL);
        return edgesInOrder;
    }

    std::vector<Vert*> Face::GetVertices() const
    {
        ASSERT(loopbase);
        std::vector<Vert*> verticesInOrder;
        Loop *l = loopbase;
        do {
            verticesInOrder.push_back(l->v);
            l = l->loopNext;
        } while (l != loopbase && l != NULL);
        return verticesInOrder;
    }

    vec3 Face::QuickNormal()
    {
        // just takes the first three verts and calculates a normal
        // not accurate if verts past the first three are not coplanar
        ASSERT(looplen > 2);
        vec3 a = loopbase->v->pos;
        vec3 b = loopbase->loopNext->v->pos;
        vec3 c = loopbase->loopNext->loopNext->v->pos;
        return Normalize(Cross(b-a, c-a));
    }


    void Edge::InsertIntoRadialCycle(Loop *loop)
    {
        if (radial == NULL)
        {
            loop->radialNext = loop;
            loop->radialPrev = loop;
            radial = loop;
        }
        else
        {
            loop->radialNext = radial->radialNext;
            loop->radialPrev = radial;
            radial->radialNext->radialPrev = loop;
            radial->radialNext = loop;
        }
    }

    void Edge::RemoveFromRadialCycle(Loop *loop)
    {
        ASSERT(radial);

        Loop *l = radial;
        do {
            if (l == loop)
            {
                if (l == l->radialPrev)
                {
                    radial = NULL;
                    return;
                }

                l->radialPrev->radialNext = l->radialNext;
                l->radialNext->radialPrev = l->radialPrev;
                radial = l->radialPrev;
                return;
            }
            l = l->radialNext;
        } while (l != radial && l != NULL);
    }

    Vert *Edge::SharesVert(Edge *other) const
    {
        if (other->a == a || other->b == a)
            return a;
        else if (other->a == b || other->b == b)
            return b;
        else
            return NULL;
    }

    std::vector<Loop*> Edge::GetRadialCycleCopy() const
    {
        ASSERT(radial);

        std::vector<Loop*> radialCycle;

        Loop *l = radial;
        do {
            radialCycle.push_back(l);
            l = l->radialNext;
        } while (l != radial && l != NULL);

        return radialCycle;
    }


    void Vert::DiskLinkInsertEdge(Edge *e)
    {
        edges.push_back(e);
    }

    void Vert::DiskLinkRemoveEdge(Edge *e)
    {
        for (Edge *edge : edges)
        {
            if (edge == e)
            {
                edges.erase(std::find(edges.begin(), edges.end(), e));
                return;
            }
        }
        fprintf(stderr, "Attemping to remove edge from disk link but edge doesn't exist in disk link.");
    }

    u64 session_VolumePersistIdCounter = 0;
    u64 FreshVolumePersistId()
    {
        return ++session_VolumePersistIdCounter;
    }

    u32 FreshElemId()
    {
        static u32 elemIdCounter = 0;
        return ++elemIdCounter;
    }

    Face *CreateFace(const std::vector<Edge*>& edges, Volume *owner)
    {
        if (edges.size() < 2)
            return NULL;

        // Order the edges by adjacency
        std::vector<Edge*> unprocessed = edges;
        std::vector<Edge*> orderedEdges = { edges[0] };
        Edge *lastEdgeAddedToOrderedList = edges[0];
        unprocessed.erase(std::find(unprocessed.begin(), unprocessed.end(), lastEdgeAddedToOrderedList));
        while (!unprocessed.empty())
        {
            bool processed = false;
            for (Edge *u : unprocessed)
            {
                if (lastEdgeAddedToOrderedList->SharesVert(u) != NULL)
                {
                    orderedEdges.push_back(u);
                    lastEdgeAddedToOrderedList = u;
                    unprocessed.erase(std::find(unprocessed.begin(), unprocessed.end(), u));
                    processed = true;
                    break;
                }
            }
            if (!processed)
            {
                fprintf(stderr, "Failed to create face from provided edges.");
                return NULL;
            }
        }

        // Create face and loops
        Face *face = new Face();
        face->elemId = FreshElemId();

        Loop *lastLoop = NULL;
        Loop *firstLoop = NULL;
        for (Edge *e : orderedEdges)
        {
            // Store references to base vertex, corresponding edge, and corresponding face
            // Insert loop into Radial Cycle of this edge
            // Form Loop Cycle of this face

            Loop *loop = new Loop();
            loop->v = lastLoop == NULL ? orderedEdges[orderedEdges.size()-1]->SharesVert(e) : lastLoop->e->SharesVert(e);
            loop->e = e;
            loop->f = face;

            e->InsertIntoRadialCycle(loop);

            if (lastLoop == NULL)
            {
                firstLoop = loop;
            }
            else
            {
                loop->loopPrev = lastLoop;
                lastLoop->loopNext = loop;
            }
            lastLoop = loop;
        }
        firstLoop->loopPrev = lastLoop;
        lastLoop->loopNext = firstLoop;

        face->loopbase = firstLoop;
        face->looplen = (int)edges.size();

        owner->faces.put(face);
        CreateGPUMesh(&face->facemesh, 3, 2, 3, GL_DYNAMIC_DRAW);
        face->storageIndex = LevelEditorFaces.count;
        LevelEditorFaces.PushBack(face);

        return face;
    }

    Edge *CreateEdge(Vert *v1, Vert *v2, Volume *owner)
    {
        Edge *edge = new Edge();
        edge->elemId = FreshElemId();

        edge->a = v1;
        edge->b = v2;
        v1->DiskLinkInsertEdge(edge);
        v2->DiskLinkInsertEdge(edge);

        owner->edges.put(edge);
        return edge;
    }

    Vert *CreateVert(vec3 pos, Volume *owner)
    {
        Vert *v = new Vert();
        v->elemId = FreshElemId();

        v->pos = pos;

        owner->verts.put(v);
        return v;
    }

    void KillFace(Face *face, Volume *owner)
    {
        // TODO
        // find remove face from owner->faces
        // delete face and loops...i think

        DeleteGPUMesh(face->facemesh.idVAO, face->facemesh.idVBO);
   
        // swap face->storageIndex and LevelEditorFaces.Back();
        LevelEditorFaces.At(face->storageIndex) = LevelEditorFaces.Back();
        LevelEditorFaces.At(face->storageIndex)->storageIndex = face->storageIndex;
        LevelEditorFaces.PopBack();

        // maybe im missing some things
    }

    void KillEdge(Edge *edge, Volume *owner)
    {
        // TODO
        // find remove edge from owner->edges
        // kill faces that use this edge
        // delete edge
        // maybe im missing some things
    }

    void KillVert(Vert *vert, Volume *owner)
    {
        // TODO
        // find remove vert from owner->verts
        // kill edges that use this vert
        // delete vert
        // maybe im missing some things
    }

    void FaceLoopReverse(Face *face)
    {
        ASSERT(face->looplen > 2);
        Vert *ogloopbasev = face->loopbase->v;
        Loop *l = face->loopbase;

        while (l->loopNext != face->loopbase)
        {
            l->v = l->loopNext->v;
            Loop *next = l->loopNext;
            l->loopNext = l->loopPrev;
            l->loopPrev = next;
            l = next;
        }
        l->v = ogloopbasev;
        l->loopNext = l->loopPrev;
        l->loopPrev = face->loopbase;
    }

    Vert *Euler_SplitEdgeMakeVert(Edge *edge, float abLerp)
    {
        // TODO
        return nullptr;
    }

    Edge *Euler_JoinEdgeKillVert()
    {
        // TODO
        return nullptr;
    }

    Face *Euler_SplitFaceMakeEdge(Face *face, Vert *v1, Vert *v2)
    {
        // TODO
        return nullptr;
    }

    Face *Euler_JoinFaceKillEdge()
    {
        // TODO
        return nullptr;
    }



    void MakeRectangularVolume(Volume *vol)
    {
        // for (Face *f : vol->faces)
        //     delete f;
        // vol->faces.clear();

        Vert *v0 = CreateVert(vec3(-320,0,-320), vol);
        Vert *v1 = CreateVert(vec3(-320,0,320), vol);
        Vert *v2 = CreateVert(vec3(320,0,320), vol);
        Vert *v3 = CreateVert(vec3(320,0,-320), vol);

        Edge *e0 = CreateEdge(v0, v1, vol);
        Edge *e1 = CreateEdge(v1, v2, vol);
        Edge *e2 = CreateEdge(v2, v3, vol);
        Edge *e3 = CreateEdge(v3, v0, vol);

        CreateFace({e0, e1, e2, e3}, vol);
    }

    void TriangulateFace_QuickDumb_WithColor(const Face f, vec3 color, float *vbdata, int *out_count)
    {
        std::vector<vec3> verticesInOrder;
        Loop *l = f.loopbase;
        do {
            vec3 p = l->v->pos;
            verticesInOrder.push_back(p);
            l = l->loopNext;
        } while (l != f.loopbase && l != NULL);

        int offset = 0;
        for (int i = 1; i < verticesInOrder.size() - 1; ++i)
        {
            vec3 a = verticesInOrder[0];
            vec3 b = verticesInOrder[i];
            vec3 c = verticesInOrder[i+1];

            vbdata[offset++] = a.x;
            vbdata[offset++] = a.y;
            vbdata[offset++] = a.z;
            vbdata[offset++] = color.x;
            vbdata[offset++] = color.y;
            vbdata[offset++] = color.z;
            vbdata[offset++] = b.x;
            vbdata[offset++] = b.y;
            vbdata[offset++] = b.z;
            vbdata[offset++] = color.x;
            vbdata[offset++] = color.y;
            vbdata[offset++] = color.z;
            vbdata[offset++] = c.x;
            vbdata[offset++] = c.y;
            vbdata[offset++] = c.z;
            vbdata[offset++] = color.x;
            vbdata[offset++] = color.y;
            vbdata[offset++] = color.z;
        }
        *out_count = offset;
    }

    void TriangulateFace_QuickDumb(const Face f, std::vector<float> *vb)
    {
        // TODO(Kevin): Figure out a good way for triangulating arbitrary simple polygons in 3d space which
        // may not have coplanar vertices...and then do delaunay and flipping for nicer triangles...
        // https://swaminathanj.github.io/cg/PolygonTriangulation.html#:~:text=Let%20v%20be%20a%20vertex,has%20at%20least%20two%20ears.

        ASSERT(f.looplen > 2);
        std::vector<vec3> verticesInOrder;
        Loop *l = f.loopbase;
        do {
            vec3 p = l->v->pos;
            verticesInOrder.push_back(p);
            l = l->loopNext;
        } while (l != f.loopbase && l != NULL);

        for (int i = 1; i < verticesInOrder.size() - 1; ++i)
        {
            vec3 a = verticesInOrder[0];
            vec3 b = verticesInOrder[i];
            vec3 c = verticesInOrder[i+1];
            vec3 n = Normalize(Cross(b-a, c-a));

            float uvScale = (1.f / (float)THIRTYTWO); // [0,1] being scaled to 32 units
            float xf = GM_abs(Dot(n, vec3(1.f, 0.f, 0.f)));
            float yf = GM_abs(Dot(n, vec3(0.f, 1.f, 0.f)));
            float zf = GM_abs(Dot(n, vec3(0.f, 0.f, 1.f)));
            float au;
            float av;
            float bu;
            float bv;
            float cu;
            float cv;
            if(xf >= yf && xf >= zf)
            {
                au = a.z * uvScale;
                av = a.y * uvScale;
                bu = b.z * uvScale;
                bv = b.y * uvScale;
                cu = c.z * uvScale;
                cv = c.y * uvScale;
            }
            else if(yf >= xf && yf >= zf)
            {
                au = a.x * uvScale;
                av = a.z * uvScale;
                bu = b.x * uvScale;
                bv = b.z * uvScale;
                cu = c.x * uvScale;
                cv = c.z * uvScale;
            }
            else if(zf >= xf && zf >= yf)
            {
                au = a.x * uvScale;
                av = a.y * uvScale;
                bu = b.x * uvScale;
                bv = b.y * uvScale;
                cu = c.x * uvScale;
                cv = c.y * uvScale;
            }

            vb->push_back(a.x);
            vb->push_back(a.y);
            vb->push_back(a.z);
            vb->push_back(au);
            vb->push_back(av);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(b.x);
            vb->push_back(b.y);
            vb->push_back(b.z);
            vb->push_back(bu);
            vb->push_back(bv);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(c.x);
            vb->push_back(c.y);
            vb->push_back(c.z);
            vb->push_back(cu);
            vb->push_back(cv);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
        }
    }

    void TriangulateFace_ForFaceBatch_QuickDumb(const Face f, std::vector<float> *vb)
    {
        // TODO(Kevin): Just keep triangulation simple...just force vertices to be coplanar and do earclipping

        ASSERT(f.looplen > 2);
        std::vector<vec3> verticesInOrder;
        std::vector<vec2> lightmapUVs;
        Loop *l = f.loopbase;
        do {
            vec3 p = l->v->pos;
            verticesInOrder.push_back(p);
            lightmapUVs.push_back(l->lmuvcache);
            l = l->loopNext;
        } while (l != f.loopbase && l != NULL);

        for (int i = 1; i < verticesInOrder.size() - 1; ++i)
        {
            vec3 a = verticesInOrder[0];
            vec3 b = verticesInOrder[i];
            vec3 c = verticesInOrder[i+1];
            vec3 n = Normalize(Cross(b-a, c-a));

            vec2 lmuvA = lightmapUVs[0];
            vec2 lmuvB = lightmapUVs[i];
            vec2 lmuvC = lightmapUVs[i+1];

            float uvScale = (1.f / (float)THIRTYTWO); // [0,1] being scaled to 32 units
            float xf = GM_abs(Dot(n, vec3(1.f, 0.f, 0.f)));
            float yf = GM_abs(Dot(n, vec3(0.f, 1.f, 0.f)));
            float zf = GM_abs(Dot(n, vec3(0.f, 0.f, 1.f)));
            float au;
            float av;
            float bu;
            float bv;
            float cu;
            float cv;
            if(xf >= yf && xf >= zf)
            {
                au = a.z * uvScale;
                av = a.y * uvScale;
                bu = b.z * uvScale;
                bv = b.y * uvScale;
                cu = c.z * uvScale;
                cv = c.y * uvScale;
            }
            else if(yf >= xf && yf >= zf)
            {
                au = a.x * uvScale;
                av = a.z * uvScale;
                bu = b.x * uvScale;
                bv = b.z * uvScale;
                cu = c.x * uvScale;
                cv = c.z * uvScale;
            }
            else if(zf >= xf && zf >= yf)
            {
                au = a.x * uvScale;
                av = a.y * uvScale;
                bu = b.x * uvScale;
                bv = b.y * uvScale;
                cu = c.x * uvScale;
                cv = c.y * uvScale;
            }

            vb->push_back(a.x);
            vb->push_back(a.y);
            vb->push_back(a.z);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(au);
            vb->push_back(av);
            vb->push_back(lmuvA.x);
            vb->push_back(lmuvA.y);

            vb->push_back(b.x);
            vb->push_back(b.y);
            vb->push_back(b.z);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(bu);
            vb->push_back(bv);
            vb->push_back(lmuvB.x);
            vb->push_back(lmuvB.y);

            vb->push_back(c.x);
            vb->push_back(c.y);
            vb->push_back(c.z);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(cu);
            vb->push_back(cv);
            vb->push_back(lmuvC.x);
            vb->push_back(lmuvC.y);
        }
    }

}