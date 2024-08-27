

namespace MapEdit
{
    struct Edge;
    struct Vert;
    struct Loop;
    struct Face;
    struct Face;

    NiceArray<Face*, 100000> EDITOR_FACES;

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

        // For map editor rendering. Each face is its own mesh. Don't optimize until my PC chugs while map editing.
        FaceBatch facemesh;
        i32 EDITOR_FACES_INDEX = -1;
        bool hovered = false;

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

        std::vector<Vert*> GetVertices() const
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

        std::vector<Loop*> GetRadialCycleCopy() const
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
        CreateFaceBatchMeshOnGPU(&face->facemesh.idVAO, &face->facemesh.idVBO);
        face->EDITOR_FACES_INDEX = EDITOR_FACES.count;
        EDITOR_FACES.PushBack(face);

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
        DeleteFaceBatchMeshOnGPU(face->facemesh.idVAO, face->facemesh.idVBO); // delete gpu face batch
        EDITOR_FACES.Back()->EDITOR_FACES_INDEX = face->EDITOR_FACES_INDEX; // update index of the face that will be moved
        EDITOR_FACES.EraseAt(face->EDITOR_FACES_INDEX); // erase face and replace index with face at the back of nice array
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



NiceArray<MapEdit::Volume, 3200> EDITOR_MAP_VOLUMES;

std::vector<float> MY_VERTEX_BUFFER;

vec3 editorCameraPosition = vec3(128, 128, 128);
float GRID_INCREMENT = 8.f;
float DISC_HANDLE_RADIUS = 10.f;
NiceArray<int, 16> SELECTED_MAP_VOLUMES_INDICES;
std::vector<MapEdit::Vert*> SELECTABLE_VERTICES;
std::vector<MapEdit::Vert*> SELECTED_VERTICES;
std::vector<int> SELECTED_SELECTABLE_VERTICES_INDICES;
std::vector<MapEdit::Face*> SELECTABLE_FACES;


enum class MapEditorTools
{
    SimpleBrushTool,
    VertexManip,
    EdgeManip,
    FaceManip,
};
MapEditorTools editorActiveTool = MapEditorTools::SimpleBrushTool;

enum class SimpleBrushToolState
{
    NotAvailable,
    NotActive,
    DrawingRectangle,
    DrawingHeight
};
SimpleBrushToolState simpleBrushToolState = SimpleBrushToolState::NotActive;


vec3 ScreenPointToWorldRay(ivec2 screenspaceCoords)
{
    // https://antongerdelan.net/opengl/raycasting.html
    // assuming 3D world view is taking up the entire window
    // Reversing perspective divide not necessary because this is a vector/direction/ray with no intrinsic depth.
    float x_NDC = ((float)screenspaceCoords.x / (float)BACKBUFFER_WIDTH) * 2.f - 1.f;
    float y_NDC = (float(BACKBUFFER_HEIGHT - screenspaceCoords.y) / (float)BACKBUFFER_HEIGHT) * 2.f - 1.f;
    vec4 ray_in_clipspace = vec4(x_NDC, y_NDC, -1.f, 1.f);
    vec4 ray_in_viewspace = activePerspectiveMatrix.GetInverse() * ray_in_clipspace;
    ray_in_viewspace = vec4(ray_in_viewspace.x, ray_in_viewspace.y, -1.f, 0.f);
    vec3 ray_in_worldspace = Normalize((activeViewMatrix.GetInverse() * ray_in_viewspace).xyz);
    return ray_in_worldspace;
}

vec3 ScreenPointToWorldPoint(ivec2 screenspaceCoords, float z_NDC)
{
    float x_NDC = ((float)screenspaceCoords.x / (float)BACKBUFFER_WIDTH) * 2.f - 1.f;
    float y_NDC = (float(BACKBUFFER_HEIGHT - screenspaceCoords.y) / (float)BACKBUFFER_HEIGHT) * 2.f - 1.f;
    vec4 point_in_clipspace = vec4(x_NDC, y_NDC, z_NDC, 1.f);
    // For points, reverse perspective divide after the inverse projection matrix transformation because it's easier that way.
    vec4 point_in_viewspace_before_perspective_divide = activePerspectiveMatrix.GetInverse() * point_in_clipspace;
    vec4 point_in_viewspace = point_in_viewspace_before_perspective_divide / point_in_viewspace_before_perspective_divide.w;
    vec4 point_in_worldspace = activeViewMatrix.GetInverse() * point_in_viewspace;
    return point_in_worldspace.xyz;
}

vec3 WorldPointToScreenPoint(vec3 worldPosition)
{
    vec4 clipspaceCoordinates = activePerspectiveMatrix * activeViewMatrix * vec4(worldPosition, 1.f);
    float screenspaceRatioX = ((clipspaceCoordinates.x / clipspaceCoordinates.w) + 1.f) / 2.f;
    float screenspaceRatioY = 1.f - (((clipspaceCoordinates.y / clipspaceCoordinates.w) + 1.f) / 2.f);
    float internalResolutionWidth = (float)BACKBUFFER_WIDTH;
    float internalResolutionHeight = (float)BACKBUFFER_HEIGHT;
    float distanceFromCameraWCS = Dot(worldPosition - editorCameraPosition, cameraDirection);
    return vec3(screenspaceRatioX * internalResolutionWidth, screenspaceRatioY * internalResolutionHeight, distanceFromCameraWCS);
}

float SnapToMapEditorGrid(float beforeSnap)
{
    int adjustedBeforeSnap = int(roundf(beforeSnap) + ((beforeSnap < 0.f ? -0.5f : 0.5f) * GRID_INCREMENT));
    // TODO(Kevin): fix divide by zero when GRID_INCREMENT is less than 1
    int snapped = adjustedBeforeSnap - adjustedBeforeSnap % (int)(GRID_INCREMENT);
    return (float)snapped;
}

vec3 SnapToMapEditorGrid(vec3 beforeSnap)
{
    return vec3(SnapToMapEditorGrid(beforeSnap.x), SnapToMapEditorGrid(beforeSnap.y), SnapToMapEditorGrid(beforeSnap.z));
}

float GetEditorHandleSize(vec3 worldPosition, float sizeInPixels)
{
    float distanceFromCamera = Dot(worldPosition - editorCameraPosition, cameraDirection);
    quat camOrientation = EulerToQuat(cameraRotation * GM_DEG2RAD);
    vec3 screenPos = WorldPointToScreenPoint(editorCameraPosition + RotateVector(vec3(distanceFromCamera, 0, 0), camOrientation));
    vec3 screenPos2 = WorldPointToScreenPoint(editorCameraPosition + RotateVector(vec3(distanceFromCamera, 0, 32.f), camOrientation));
    // scaled by 32 to avoid floating point imprecision
    float screenDist = Magnitude(screenPos - screenPos2);
    return (sizeInPixels*32.f / GM_max(screenDist, 0.0001f));
}

void TickMapEditor()
{
    SDL_SetRelativeMouseMode(mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT) ? SDL_TRUE : SDL_FALSE);

    // EDITOR CAMERA MOVE
    if (mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT))
    {
        float camYawDelta = mousedelta.x*0.1f;
        float camPitchDelta = mousedelta.y*0.1f;
        cameraRotation.y -= camYawDelta;
        cameraRotation.z -= camPitchDelta;
        if (cameraRotation.z > 89.f)
            cameraRotation.z = 89.f;
        if (cameraRotation.z < -89.f)
            cameraRotation.z = -89.f;
    }
    cameraDirection = Normalize(OrientationToDirection(EulerToQuat(cameraRotation * GM_DEG2RAD)));
    cameraRight = Normalize(Cross(cameraDirection, GM_UP_VECTOR));
    cameraUp = Normalize(Cross(cameraRight, cameraDirection));
    float moveSpeed = 250.f;
    if (keyscurrent[SDL_SCANCODE_LSHIFT])
        moveSpeed *= 2.7f;
    vec3 playerPositionDelta;
    if (mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT))
    {
        if (keyscurrent[SDL_SCANCODE_W])
            playerPositionDelta += cameraDirection * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_A])
            playerPositionDelta += -cameraRight * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_S])
            playerPositionDelta += -cameraDirection * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_D])
            playerPositionDelta += cameraRight * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_Q])
            playerPositionDelta += -GM_UP_VECTOR * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_E])
            playerPositionDelta += GM_UP_VECTOR * moveSpeed * g_DeltaTime;
        editorCameraPosition += playerPositionDelta;
    }
    activeViewMatrix = ViewMatrixLookAt(editorCameraPosition, editorCameraPosition + cameraDirection, cameraUp);

    // DRAW AXIS AND GRID LINES
    vec3 intersection_with_xz_plane;
    if (IntersectPlaneAndLine(vec3(), GM_UP_VECTOR, editorCameraPosition,
                              -GM_UP_VECTOR,
                              &intersection_with_xz_plane))
    {
        float gridVisualIncrement = GRID_INCREMENT;
        intersection_with_xz_plane = SnapToMapEditorGrid(intersection_with_xz_plane);
        float gridoriginx = intersection_with_xz_plane.x;
        float gridoriginz = intersection_with_xz_plane.z;
        int linedrawcount = 4000 / (int)gridVisualIncrement;
        for (int i = -linedrawcount; i < linedrawcount; ++i)
        {
            if (float(i) * gridVisualIncrement + gridoriginz != 0.f)
                PrimitiveDrawLine(vec3(320000.f + gridoriginx, 0.f, float(i) * gridVisualIncrement + gridoriginz),
                                  vec3(-320000.f + gridoriginx, 0.f, float(i) * gridVisualIncrement + gridoriginz),
                                  vec4(RGB255TO1(155,155,155), 0.25f));
            if (float(i) * gridVisualIncrement + gridoriginx != 0.f)
                PrimitiveDrawLine(vec3(float(i) * gridVisualIncrement + gridoriginx, 0.f, 320000.f + gridoriginz),
                                  vec3(float(i) * gridVisualIncrement + gridoriginx, 0.f, -320000.f + gridoriginz),
                                  vec4(RGB255TO1(155,155,155), 0.25f));
        }
    }
    PrimitiveDrawLine(vec3(320000.f, 0.f, 0.f), vec3(-320000.f, 0.f, 0.f), vec4(RGB255TO1(205, 56, 9), 0.8f));
    // PrimitiveDrawLine(vec3(0.f, 320000.f, 0.f), vec3(0.f, -320000.f, 0.f), vec4(RGB255TO1(67, 123, 9), 0.8f));
    PrimitiveDrawLine(vec3(0.f, 0.f, 320000.f), vec3(0.f, 0.f, -320000.f), vec4(RGB255TO1(21, 129, 205), 0.8f));
    Gui::PrimitiveText(100,100, 36, Gui::Align::Left, std::to_string(GRID_INCREMENT).c_str());

    if (Gui::PrimitiveLabelledButton(Gui::UIRect(400,10,60,32), "Grid -", Gui::Align::Right))
    {
        GRID_INCREMENT /= 2.f;
    }
    if (Gui::PrimitiveLabelledButton(Gui::UIRect(470,10,60,32), "Grid +", Gui::Align::Left))
    {
        GRID_INCREMENT *= 2.f;
    }


    static u32 hotHandleId = 0;


    if (keyspressed[SDL_SCANCODE_1] ||
        Gui::ImageButton(Gui::UIRect(10,10,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::SimpleBrushTool;
        hotHandleId = 0;
    }
    if (keyspressed[SDL_SCANCODE_2] ||
        Gui::ImageButton(Gui::UIRect(10,52,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::VertexManip;
        hotHandleId = 0;
    }
    if (keyspressed[SDL_SCANCODE_3] ||
        Gui::ImageButton(Gui::UIRect(10,94,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::EdgeManip;
        hotHandleId = 0;
    }
    if (keyspressed[SDL_SCANCODE_4] ||
        Gui::ImageButton(Gui::UIRect(10,136,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::FaceManip;
        hotHandleId = 0;
    }

    Gui::PrimitiveText(100,28, 9, Gui::Align::Left, "Toggle Auto Colliders []");

    if (Gui::anyElementActive)
        return;

    // Selection handles

    // on mouse up, disc remains selected
    // on mouse down, if disc not selected, then if ctrl pressed then add to selection but don't start drag
    //                                      if ctrl not pressed then start drag
    // on mouse down, if disc was already selected, then if ctrl pressed then deselect that disc/vert, if ctrl not pressed
    // then start checking for drag

    SELECTABLE_VERTICES.clear();
    SELECTABLE_FACES.clear();
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        MapEdit::Volume volume = EDITOR_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));

        for (int j = 0; j < volume.verts.size(); ++j)
        {
            SELECTABLE_VERTICES.push_back(volume.verts[j]);
        }

        for (int j = 0; j < volume.faces.size(); ++j)
        {
            SELECTABLE_FACES.push_back(volume.faces[j]);
        }
    }

    if (editorActiveTool == MapEditorTools::FaceManip)
    {
        static vec3 intersectOffsetFromLoopbaseVert;

        bool lmbPressedThisFrame = mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT);
        bool lmbIsPressed = mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT);

        for (u32 id = 1; id <= SELECTABLE_FACES.size(); ++id)
        {
            MapEdit::Face *face = SELECTABLE_FACES[id-1];
            vec3 idrgb = HandleIdToRGB(id);
            int count;
            MapEdit::TriangulateFace_QuickDumb_WithColor(*face, idrgb, HANDLES_VB.data + HANDLES_VB.count, &count);
            HANDLES_VB.count += count;
        }
        u32 hoveredFaceId = FlushHandles(mousepos, renderTargetGame, activeViewMatrix, activePerspectiveMatrix, false);
        LogMessage("%d", hoveredFaceId);

        if (hoveredFaceId > 0)
        {
            MapEdit::Face *face = SELECTABLE_FACES[hoveredFaceId-1];
            face->hovered = true;

            if (lmbPressedThisFrame)
            {
                hotHandleId = hoveredFaceId;
                SELECTED_VERTICES = face->GetVertices();

                vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);
                vec3 intersect;
                IntersectPlaneAndLineWithDirections(face->loopbase->v->pos, face->QuickNormal(), worldpos_mouse, worldray_mouse, &intersect);
                intersectOffsetFromLoopbaseVert = intersect - face->loopbase->v->pos;
            }
        }


        if (lmbIsPressed && hotHandleId > 0)
        {
            MapEdit::Face *hotFace = SELECTABLE_FACES[hotHandleId-1];

            vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);

            vec3 translation;
            vec3 faceNormal = hotFace->QuickNormal();
            // PrimitiveDrawSolidDisc(intersect, -cameraDirection, 8.f, vec4(1.0,0.0,1.0,1.0));
            vec3 dragPlanePoint = hotFace->loopbase->v->pos + intersectOffsetFromLoopbaseVert;
            vec3 intersect;
            IntersectPlaneAndLineWithDirections(dragPlanePoint, -cameraDirection, worldpos_mouse, worldray_mouse, &intersect);
            float scaleComponent = Dot((intersect - dragPlanePoint), faceNormal);
            translation = SnapToMapEditorGrid(scaleComponent) * faceNormal;//vec3(0.f,yTranslation,0.f);

            // vec3 dragPlanePoint = hotFace->pos;
            // vec3 dragPlaneNormal = -cameraDirection;
            // vec3 intersect;
            // IntersectPlaneAndLineWithDirections(dragPlanePoint, dragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
            // intersect = vec3(intersect.x, SnapToMapEditorGrid(intersect.y), intersect.z);
            // float yTranslation = Dot((intersect - dragPlanePoint), GM_UP_VECTOR);
            // translation = vec3(0.f,yTranslation,0.f);
            // if (keyscurrent[SDL_SCANCODE_LALT])
            // {
            //     vec3 dragPlanePoint = hotVert->pos;
            //     vec3 dragPlaneNormal = -cameraDirection;
            //     vec3 intersect;
            //     IntersectPlaneAndLineWithDirections(dragPlanePoint, dragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
            //     intersect = vec3(intersect.x, SnapToMapEditorGrid(intersect.y), intersect.z);
            //     float yTranslation = Dot((intersect - dragPlanePoint), GM_UP_VECTOR);
            //     translation = vec3(0.f,yTranslation,0.f);
            // }
            // else
            // {
            //     vec3 XZDragPlanePoint = hotVert->pos;
            //     vec3 XZDragPlaneNormal = GM_UP_VECTOR;
            //     vec3 intersect;
            //     IntersectPlaneAndLine(XZDragPlanePoint, XZDragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
            //     translation = SnapToMapEditorGrid(intersect) - hotVert->pos;
            // }
            // if (translation != vec3())
            // {
            //     TranslationHappenedForSomeFrames = true;
            //     TotalTranslationThisMouseDown += translation;
            // }

            for (MapEdit::Vert *vert : SELECTED_VERTICES)
            {
                vert->pos += translation;
            }
            for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
            {
                MapEdit::Volume& selectedVol = EDITOR_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
                for (MapEdit::Face *face : selectedVol.faces)
                {
                    MY_VERTEX_BUFFER.clear();
                    MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindFaceBatchBufferObject(&face->facemesh, (u32)(sizeof(float)*MY_VERTEX_BUFFER.size()), MY_VERTEX_BUFFER.data());
                }
            }
        }
    }

    if (editorActiveTool == MapEditorTools::VertexManip)
    {
        static bool LCtrlDownOnLeftMouseDown = false;
        static bool AlreadySelectedOnLeftMouseDown = false;
        static bool TranslationHappenedForSomeFrames = false;
        static vec3 TotalTranslationThisMouseDown = vec3();

        if (mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
            TotalTranslationThisMouseDown = vec3();
            TranslationHappenedForSomeFrames = false;
            LCtrlDownOnLeftMouseDown = keyscurrent[SDL_SCANCODE_LCTRL];

            for (u32 id = 1; id <= SELECTABLE_VERTICES.size(); ++id)
            {
                MapEdit::Vert *vert = SELECTABLE_VERTICES[id-1];
                DoDiscHandle(id, vert->pos, editorCameraPosition, GetEditorHandleSize(vert->pos, DISC_HANDLE_RADIUS + 4.f));
            }
            u32 clickedId = FlushHandles(mousepos, renderTargetGame, activeViewMatrix, activePerspectiveMatrix, false);

            if (clickedId > 0)
            {
                hotHandleId = clickedId;
                auto clickedIter = std::find(SELECTED_SELECTABLE_VERTICES_INDICES.begin(), SELECTED_SELECTABLE_VERTICES_INDICES.end(), clickedId);
                AlreadySelectedOnLeftMouseDown = clickedIter != SELECTED_SELECTABLE_VERTICES_INDICES.end();
                if (!AlreadySelectedOnLeftMouseDown)
                {
                    if (!LCtrlDownOnLeftMouseDown)
                        SELECTED_SELECTABLE_VERTICES_INDICES.clear();
                    SELECTED_SELECTABLE_VERTICES_INDICES.push_back(clickedId);
                }
            }
            else if (!LCtrlDownOnLeftMouseDown)
            {
                SELECTED_SELECTABLE_VERTICES_INDICES.clear();
            }
            else
            {
                // if id == 0 (nothing clicked) then check for click against volumes
                // also draw all the other volumes with a clickable rgb id so we can switch which volume displays their vertices
                // draw the selected volume(s) with rgb(0) so it is ignored when clicked
                // draw these discs on top of all the volumes, but the discs themselves must do depth test so a disc closer to camera
                // is clicked first.
            }
        }

        if (mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT) && hotHandleId > 0)
        {
            MapEdit::Vert *hotVert = SELECTABLE_VERTICES[hotHandleId-1];

            vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);

            vec3 translation;
            if (keyscurrent[SDL_SCANCODE_LALT])
            {
                vec3 dragPlanePoint = hotVert->pos;
                vec3 dragPlaneNormal = -cameraDirection;
                vec3 intersect;
                IntersectPlaneAndLineWithDirections(dragPlanePoint, dragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
                intersect = vec3(intersect.x, SnapToMapEditorGrid(intersect.y), intersect.z);
                float yTranslation = Dot((intersect - dragPlanePoint), GM_UP_VECTOR);
                translation = vec3(0.f,yTranslation,0.f);
            }
            else
            {
                vec3 XZDragPlanePoint = hotVert->pos;
                vec3 XZDragPlaneNormal = GM_UP_VECTOR;
                vec3 intersect;
                IntersectPlaneAndLine(XZDragPlanePoint, XZDragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
                translation = SnapToMapEditorGrid(intersect) - hotVert->pos;
            }
            if (translation != vec3())
            {
                TranslationHappenedForSomeFrames = true;
                TotalTranslationThisMouseDown += translation;
            }

            for (int index : SELECTED_SELECTABLE_VERTICES_INDICES)
            {
                MapEdit::Vert *vert = SELECTABLE_VERTICES[index - 1];
                vert->pos += translation;
            }
            for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
            {
                MapEdit::Volume& selectedVol = EDITOR_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
                for (MapEdit::Face *face : selectedVol.faces)
                {
                    MY_VERTEX_BUFFER.clear();
                    MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindFaceBatchBufferObject(&face->facemesh, (u32)(sizeof(float)*MY_VERTEX_BUFFER.size()), MY_VERTEX_BUFFER.data());
                }
            }
        }

        if (keyscurrent[SDL_SCANCODE_ESCAPE])
        {
            for (int index : SELECTED_SELECTABLE_VERTICES_INDICES)
            {
                MapEdit::Vert *vert = SELECTABLE_VERTICES[index - 1];
                vert->pos -= TotalTranslationThisMouseDown;
            }
            for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
            {
                MapEdit::Volume& selectedVol = EDITOR_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
                for (MapEdit::Face *face : selectedVol.faces)
                {
                    MY_VERTEX_BUFFER.clear();
                    MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindFaceBatchBufferObject(&face->facemesh, (u32)(sizeof(float)*MY_VERTEX_BUFFER.size()), MY_VERTEX_BUFFER.data());
                }
            }

            hotHandleId = 0;
            TotalTranslationThisMouseDown = vec3();
            AlreadySelectedOnLeftMouseDown = false;
            LCtrlDownOnLeftMouseDown = false;
            TranslationHappenedForSomeFrames = false;
        }

        if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
            if (hotHandleId > 0)
            {
                if (LCtrlDownOnLeftMouseDown)
                {
                    auto clickedIter = std::find(SELECTED_SELECTABLE_VERTICES_INDICES.begin(), SELECTED_SELECTABLE_VERTICES_INDICES.end(), hotHandleId);
                    if (clickedIter != SELECTED_SELECTABLE_VERTICES_INDICES.end())
                    {
                        if (AlreadySelectedOnLeftMouseDown && !TranslationHappenedForSomeFrames)
                            SELECTED_SELECTABLE_VERTICES_INDICES.erase(clickedIter);
                    }
                    else
                    {
                        LogError("wtf");
                    }
                }
                else if (!TranslationHappenedForSomeFrames)
                {
                    SELECTED_SELECTABLE_VERTICES_INDICES.clear();
                    SELECTED_SELECTABLE_VERTICES_INDICES.push_back(hotHandleId);
                }
            }

            hotHandleId = 0;
            TotalTranslationThisMouseDown = vec3();
            AlreadySelectedOnLeftMouseDown = false;
            LCtrlDownOnLeftMouseDown = false;
            TranslationHappenedForSomeFrames = false;
        }
    }

    if (editorActiveTool == MapEditorTools::SimpleBrushTool)
    {
        static vec3 rectstartpoint;
        static vec3 rectendpoint;
        static float desiredBrushY_in_world;
        switch (simpleBrushToolState)
        {
            case SimpleBrushToolState::NotActive:
                if (mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    // todo with drawing brush ON other volumes, do picking first to see on which volume we are drawing,
                    // then do the mouse ray intersect...although this optimization might not be needed

                    // We only want to draw within a certain radius from the camera. never too far from the camera.
                    desiredBrushY_in_world = 0.f;
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    float f = (desiredBrushY_in_world - ws.y) / wr.y;
                    if (f > 0.f) // should accept any f... since we want to draw brush in any direction
                    {
                        simpleBrushToolState = SimpleBrushToolState::DrawingRectangle;
                        rectstartpoint = SnapToMapEditorGrid(ws + wr * f);
                    }
                }
                break;
            case SimpleBrushToolState::DrawingRectangle:
                if (mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    float f = (desiredBrushY_in_world - ws.y) / wr.y;
                    vec3 endpoint = SnapToMapEditorGrid(ws + wr * f);
                    vec3 startToEndVector = vec3(endpoint - rectstartpoint);
                    float d = Magnitude(startToEndVector);
                    vec3 startToEndProjOnToZDirection = Dot(startToEndVector, GM_RIGHT_VECTOR) * GM_RIGHT_VECTOR;
                    vec3 startToEndProjOnToXDirection = Dot(startToEndVector, GM_FORWARD_VECTOR) * GM_FORWARD_VECTOR;
                    PrimitiveDrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToXDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                    PrimitiveDrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToZDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                    PrimitiveDrawLine(endpoint, endpoint - startToEndProjOnToXDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                    PrimitiveDrawLine(endpoint, endpoint - startToEndProjOnToZDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                }
                if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    float f = (desiredBrushY_in_world - ws.y) / wr.y;
                    rectendpoint = SnapToMapEditorGrid(ws + wr * f);
                    vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);
                    float d = Magnitude(startToEndVector);
                    if (true) // TODO check X and Z values to see if theyre bigger than grid size
                        simpleBrushToolState = SimpleBrushToolState::DrawingHeight;
                    else
                        simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                if (keyspressed[SDL_SCANCODE_ESCAPE])
                    simpleBrushToolState = SimpleBrushToolState::NotActive;

                break;
            case SimpleBrushToolState::DrawingHeight:
                vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);
                float zcomponent = Dot(startToEndVector, GM_RIGHT_VECTOR);
                float xcomponent = Dot(startToEndVector, GM_FORWARD_VECTOR);
                vec3 startToEndProjOnToZDirection = zcomponent * GM_RIGHT_VECTOR;
                vec3 startToEndProjOnToXDirection = xcomponent * GM_FORWARD_VECTOR;

                // let plane be defined at point rectendpoint with normal -cameraDirection
                // then the point of intersection between mouse ray and the plane - endrectpoint and project it onto
                // the direction of translation e.g. GM_UP_VECTOR
                vec3 drawingSurfaceNormal = GM_UP_VECTOR;
                static vec3 height = vec3();
                static vec3 heightBeforeSnap = vec3();
                vec3 pn = -cameraDirection;
                vec3 pp = rectendpoint + heightBeforeSnap;
                vec3 wp = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 wr = ScreenPointToWorldRay(mousepos);
                vec3 intersection;
                IntersectPlaneAndLineWithDirections(pp, pn, wp, wr, &intersection);
                float trueHeightComp = Dot((intersection - rectendpoint), drawingSurfaceNormal);
                float heightcomponent = SnapToMapEditorGrid(trueHeightComp);
                height = heightcomponent * drawingSurfaceNormal;
                heightBeforeSnap = trueHeightComp * drawingSurfaceNormal;

                vec3 floorPointA = rectstartpoint;
                vec3 floorPointB = rectstartpoint + startToEndProjOnToXDirection;
                vec3 floorPointC = rectendpoint;
                vec3 floorPointD = rectendpoint - startToEndProjOnToXDirection;
                // given how i've set up points ABCD:
                //      if xcomponent < 0, swap A and B, swap C and D
                //      if zcomponent < 0, swap B and C, swap A and D
                if (xcomponent < 0)
                {
                    auto temp = floorPointA;
                    floorPointA = floorPointB;
                    floorPointB = temp;
                    temp = floorPointC;
                    floorPointC = floorPointD;
                    floorPointD = temp;
                }
                if (zcomponent < 0)
                {
                    auto temp = floorPointB;
                    floorPointB = floorPointC;
                    floorPointC = temp;
                    temp = floorPointA;
                    floorPointA = floorPointD;
                    floorPointD = temp;
                }
                vec3 ceilPointA = floorPointA + height;
                vec3 ceilPointB = floorPointB + height;
                vec3 ceilPointC = floorPointC + height;
                vec3 ceilPointD = floorPointD + height;

                PrimitiveDrawLine(floorPointA, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointA, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointC, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointC, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0));

                PrimitiveDrawLine(floorPointA, ceilPointA, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointB, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointC, ceilPointC, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointD, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0));

                PrimitiveDrawLine(ceilPointA, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(ceilPointA, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(ceilPointC, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(ceilPointC, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0));

                if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    // if heigth component is zero or too small then just default to current grid increment

                    MapEdit::Volume createdVolume;

                    // complete with drawn height
                    MapEdit::Vert *fv0 = MapEdit::CreateVert(floorPointA, &createdVolume);
                    MapEdit::Vert *fv1 = MapEdit::CreateVert(floorPointB, &createdVolume);
                    MapEdit::Vert *fv2 = MapEdit::CreateVert(floorPointC, &createdVolume);
                    MapEdit::Vert *fv3 = MapEdit::CreateVert(floorPointD, &createdVolume);
                    MapEdit::Vert *cv0 = MapEdit::CreateVert(ceilPointA, &createdVolume);
                    MapEdit::Vert *cv1 = MapEdit::CreateVert(ceilPointB, &createdVolume);
                    MapEdit::Vert *cv2 = MapEdit::CreateVert(ceilPointC, &createdVolume);
                    MapEdit::Vert *cv3 = MapEdit::CreateVert(ceilPointD, &createdVolume);
                    if (heightcomponent < 0)
                    {
                        auto temp0 = fv0;
                        auto temp1 = fv1;
                        auto temp2 = fv2;
                        auto temp3 = fv3;
                        fv0 = cv0;
                        fv1 = cv1;
                        fv2 = cv2;
                        fv3 = cv3;
                        cv0 = temp0;
                        cv1 = temp1;
                        cv2 = temp2;
                        cv3 = temp3;
                    }

                    MapEdit::Edge *f0_to_f1 = MapEdit::CreateEdge(fv0, fv1, &createdVolume);
                    MapEdit::Edge *f1_to_f2 = MapEdit::CreateEdge(fv1, fv2, &createdVolume);
                    MapEdit::Edge *f2_to_f3 = MapEdit::CreateEdge(fv2, fv3, &createdVolume);
                    MapEdit::Edge *f3_to_f0 = MapEdit::CreateEdge(fv3, fv0, &createdVolume);
                    MapEdit::Edge *c0_to_c1 = MapEdit::CreateEdge(cv0, cv1, &createdVolume);
                    MapEdit::Edge *c1_to_c2 = MapEdit::CreateEdge(cv1, cv2, &createdVolume);
                    MapEdit::Edge *c2_to_c3 = MapEdit::CreateEdge(cv2, cv3, &createdVolume);
                    MapEdit::Edge *c3_to_c0 = MapEdit::CreateEdge(cv3, cv0, &createdVolume);
                    MapEdit::Edge *f0_to_c0 = MapEdit::CreateEdge(fv0, cv0, &createdVolume);
                    MapEdit::Edge *f1_to_c1 = MapEdit::CreateEdge(fv1, cv1, &createdVolume);
                    MapEdit::Edge *f2_to_c2 = MapEdit::CreateEdge(fv2, cv2, &createdVolume);
                    MapEdit::Edge *f3_to_c3 = MapEdit::CreateEdge(fv3, cv3, &createdVolume);

                    MapEdit::CreateFace({f0_to_f1, f1_to_f2, f2_to_f3, f3_to_f0}, &createdVolume);
                    MapEdit::Face *revthis = MapEdit::CreateFace({c0_to_c1, c1_to_c2, c2_to_c3, c3_to_c0},
                                                                 &createdVolume);
                    MapEdit::FaceLoopReverse(revthis);
                    MapEdit::CreateFace({f0_to_f1, f0_to_c0, f1_to_c1, c0_to_c1}, &createdVolume);
                    MapEdit::CreateFace({f1_to_f2, f1_to_c1, c1_to_c2, f2_to_c2}, &createdVolume);
                    MapEdit::CreateFace({f2_to_f3, f2_to_c2, c2_to_c3, f3_to_c3}, &createdVolume);
                    MapEdit::CreateFace({f3_to_f0, f3_to_c3, c3_to_c0, f0_to_c0}, &createdVolume);

                    EDITOR_MAP_VOLUMES.PushBack(createdVolume);
                    SELECTED_MAP_VOLUMES_INDICES.PushBack(EDITOR_MAP_VOLUMES.count - 1);

                    for (auto face: createdVolume.faces)
                    {
                        MY_VERTEX_BUFFER.clear();
                        TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                        RebindFaceBatchBufferObject(&face->facemesh, (u32) MY_VERTEX_BUFFER.size() * sizeof(float),
                            MY_VERTEX_BUFFER.data());
                        face->facemesh.sharedTexture = tex_Default;
                    }

                    simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                if (keyspressed[SDL_SCANCODE_ESCAPE])
                {
                    simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                break;
        }
    }
}

// void EditorSerializeMap()
// {
//     for (int i = 0; i < EDITOR_MAP_VOLUMES.count; ++i)
//     {
//         const MapEdit::Volume& volume = EDITOR_MAP_VOLUMES.At(i);
        
//         for (Vert *v : volume.verts)
//         {

//         }

//         for (Edge *e : volume.edges)
//         {

//         }

//     }

//     // serialize EDITOR_MAP_VOLUMES 
//     // i think i want to reuse MapEdit::CreateFace for Deserialize, 
//     // which means i need to preserve the order of each vertex

//     // when serializing, every element's address becomes its unique id
//     // when deserializing, 
//     //      every vertex is created and hashmap maps unique id to the created vertex ptr
//     //      every edge is created 

//     // hash the address to the element
//     // to 

//     // all of the vertices need to exist in memory
//     // all of the edges then get created from those vertices

// }

void EditorDeserializeMap()
{
    // deserialize into EDITOR_MAP_VOLUMES
}

void BuildGameMap()
{
    // need to just save a bunch of FaceBatches (which are basically just meshes)
}

void LoadGameMap()
{
    // just load the face batches into memory
}


/*
ALL_VERTICES
start at address 0x0000
when create vertex, it has address = base + offset
to find its index, it is simply base - address 


*/





