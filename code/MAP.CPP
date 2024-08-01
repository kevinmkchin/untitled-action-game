

namespace EditableVolumes
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

    Face *CreateFace(const std::vector<Edge*>& edges)
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
        face->looplen = edges.size();

        return face;
    }

    Edge *CreateEdge(Vert *v1, Vert *v2)
    {
        Edge *edge = new Edge();
        edge->a = v1;
        edge->b = v2;
        v1->DiskLinkInsertEdge(edge);
        v2->DiskLinkInsertEdge(edge);
        return edge;
    }

    Vert *CreateVert(vec3 pos)
    {
        Vert *v = new Vert();
        v->pos = pos;
        return v;
    }

    void KillFace(Face *face)
    {

    }

    void KillEdge(Edge *edge)
    {

    }

    void KillVert(Vert *vert)
    {

    }

    void FaceLoopReverse(Face *face)
    {

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


    struct Volume
    {
        std::vector<Face*> faces;
    };

    void MakeRectangularVolume(Volume *vol)
    {
        for (Face *f : vol->faces)
            delete f;
        vol->faces.clear();

        Vert *v0 = CreateVert(vec3(-32,0,-32));
        Vert *v1 = CreateVert(vec3(-32,0,32));
        Vert *v2 = CreateVert(vec3(32,0,32));
        Vert *v3 = CreateVert(vec3(32,0,-32));

        Edge *e0 = CreateEdge(v0, v1);
        Edge *e1 = CreateEdge(v1, v2);
        Edge *e2 = CreateEdge(v2, v3);
        Edge *e3 = CreateEdge(v3, v0);

        Face *f0 = CreateFace({e0, e1, e2, e3});

        vol->faces.push_back(f0);
    }

    void MakeCubeVolume(Volume *vol)
    {

    }
}


struct EditableMapData
{

};


