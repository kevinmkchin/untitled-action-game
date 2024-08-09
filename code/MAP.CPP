

namespace MapEdit
{
    struct Edge;
    struct Vert;
    struct Loop;
    struct Face;

    struct Loop
    {
        Loop *loopNext = NULL;
        Loop *loopPrev = NULL;
        Loop *radialNext = NULL;
        Loop *radialPrev = NULL;
        Edge *e = NULL;
        Vert *v = NULL;
        Face *f = NULL;
    };

    struct Face
    {
        Loop *loopbase = NULL;
        int looplen = -1;

        int RefreshLoopLen()
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

        vec3 QuickNormal();
    };

    struct Edge
    {
        Vert *a = NULL;
        Vert *b = NULL;
        Loop *radial = NULL;

        void InsertIntoRadialCycle(Loop *loop)
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

        void RemoveFromRadialCycle(Loop *loop)
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

        Vert *SharesVert(Edge *other) const
        {
            if (other->a == a || other->b == a)
                return a;
            else if (other->a == b || other->b == b)
                return b;
            else
                return NULL;
        }

        // void SwitchAB()
        // {
        //     auto v = a;
        //     b = a;
        //     a = v;
        // }
    };

    struct Vert
    {
        vec3 pos;

        std::vector<Edge*> edges;

        void DiskLinkInsertEdge(Edge *e)
        {
            edges.push_back(e);
        }

        void DiskLinkRemoveEdge(Edge *e)
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
    };

    struct Volume
    {
        std::vector<Face*> faces;
        std::vector<Edge*> edges;
        std::vector<Vert*> verts;
    };

    // vec3 Face::QuickNormal()
    // {
    //     // just takes the first three verts and calculates a normal
    //     // not accurate if verts past the first three are not coplanar
    //     ASSERT(looplen > 2);
    //     vec3 a = loopbase->v->pos;
    //     vec3 b = loopbase->loopNext->v->pos;
    //     vec3 c = loopbase->loopNext->loopNext->v->pos;
    //     return Normalize(Cross(b-a, c-a));
    // }

    /*
    Example usage:

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

        owner->faces.push_back(face);
        return face;
    }

    Edge *CreateEdge(Vert *v1, Vert *v2, Volume *owner)
    {
        Edge *edge = new Edge();
        edge->a = v1;
        edge->b = v2;
        v1->DiskLinkInsertEdge(edge);
        v2->DiskLinkInsertEdge(edge);

        owner->edges.push_back(edge);
        return edge;
    }

    Vert *CreateVert(vec3 pos, Volume *owner)
    {
        Vert *v = new Vert();
        v->pos = pos;

        owner->verts.push_back(v);
        return v;
    }

    void KillFace(Face *face, Volume *owner)
    {
        // find remove face from owner->faces
        // delete face and loops...i think
        // maybe im missing some things
    }

    void KillEdge(Edge *edge, Volume *owner)
    {
        // find remove edge from owner->edges
        // kill faces that use this edge
        // delete edge
        // maybe im missing some things
    }

    void KillVert(Vert *vert, Volume *owner)
    {
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

    Vert *Euler_SplitEdgeMakeVert(Edge *edge, float abLerp = 0.5f)
    {
        return nullptr;
    }

    Edge *Euler_JoinEdgeKillVert()
    {
        return nullptr;
    }

    Face *Euler_SplitFaceMakeEdge(Face *face, Vert *v1, Vert *v2)
    {
        return nullptr;
    }

    Face *Euler_JoinFaceKillEdge()
    {
        return nullptr;
    }



    void MakeRectangularVolume(Volume *vol)
    {
        for (Face *f : vol->faces)
            delete f;
        vol->faces.clear();

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

    void MakeCubeVolume(Volume *vol)
    {

    }

    void TriangulateFace_QuickDumb(const Face f, std::vector<float> *vb)
    {
        // TODO Figure out a good way for triangulating arbitrary simple polygons in 3d space which
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
}


struct EditableMapData
{

};




