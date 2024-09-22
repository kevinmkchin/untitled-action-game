
struct db_tex_t
{
    u32 persistId = 0;
    BitmapHandle bitmap;
    GPUTexture gputex;
};


struct lm_face_t
{
    vec3 *pos = NULL;
    vec3 *norm = NULL;
    u8   *light = NULL;
    i32 w = -1;
    i32 h = -1;
};

vec3 *all_lm_pos = NULL;
vec3 *all_lm_norm = NULL;
u8   *all_lm_light = NULL;



namespace MapEdit
{
    struct Edge;
    struct Vert;
    struct Loop;
    struct Face;
    struct Face;

    NiceArray<Face*, 100000> EDITOR_FACES; // TODO(Kevin): make this an implicit array using stb_ds

    struct Loop
    {
        Loop *loopNext = NULL;
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

        // For map editor rendering. Each face is its own mesh. Don't optimize until my PC chugs while map editing.
        FaceBatch facemesh;
        db_tex_t texture;
        lm_face_t lightmap;
        // bool receivelight; mark whether to generate light map
        // bool hascollision; mark whether to generate collider
        u32 elemId = 0;

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

        std::vector<Loop*> GetLoopCycle() const
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

        std::vector<Edge*> GetEdges() const
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

        u32 elemId = 0;

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

        u32 elemId = 0;
        // editor runtime data
        vec3 poscache;

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

        u64 persistId = 0; // 0 is invalid volume. All volumes should call FreshVolumePersistId.
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

    static u64 session_VolumePersistIdCounter = 0;
    u64 FreshVolumePersistId()
    {
        return ++session_VolumePersistIdCounter;
    }

    u32 FreshElemId()
    {
        static u32 elemIdCounter = 0;
        return ++elemIdCounter;
    }

    // this is the OMEGA function
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

        owner->faces.push_back(face);
        CreateFaceBatchMeshOnGPU(&face->facemesh.idVAO, &face->facemesh.idVBO);
        face->EDITOR_FACES_INDEX = EDITOR_FACES.count;
        EDITOR_FACES.PushBack(face);

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

        owner->edges.push_back(edge);
        return edge;
    }

    Vert *CreateVert(vec3 pos, Volume *owner)
    {
        Vert *v = new Vert();
        v->elemId = FreshElemId();

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

    void TriangulateFace_QuickDumb_LMUV(const Face f, std::vector<float> *vb)
    {
        // TODO(Kevin): Figure out a good way for triangulating arbitrary simple polygons in 3d space which
        // may not have coplanar vertices...and then do delaunay and flipping for nicer triangles...
        // https://swaminathanj.github.io/cg/PolygonTriangulation.html#:~:text=Let%20v%20be%20a%20vertex,has%20at%20least%20two%20ears.

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
            vb->push_back(lmuvA.x);
            vb->push_back(lmuvA.y);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(b.x);
            vb->push_back(b.y);
            vb->push_back(b.z);
            vb->push_back(lmuvB.x);
            vb->push_back(lmuvB.y);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
            vb->push_back(c.x);
            vb->push_back(c.y);
            vb->push_back(c.z);
            vb->push_back(lmuvC.x);
            vb->push_back(lmuvC.y);
            vb->push_back(n.x);
            vb->push_back(n.y);
            vb->push_back(n.z);
        }
    }

}

// Asset DB

struct asset_db_t
{
    std::map<u64, db_tex_t> Textures;
};
asset_db_t Assets;

struct editor_session_data_t
{
    MapEdit::Face *SelectedFace = NULL;

    db_tex_t SelectedTexture;

    u32 TexturePersistIdCounter = 0;

    // move hotHandleId here
    // move all the editor session specific data here
};
editor_session_data_t Session;

db_tex_t GetTextureByPersistId(u32 persistId)
{
    auto textureiter = Assets.Textures.find(persistId);
    if (textureiter != Assets.Textures.end())
        return textureiter->second;
    
    db_tex_t missingTex;
    missingTex.gputex = tex_Missing;
    return missingTex;
}

db_tex_t LoadNewTextureToDB(const char *path)
{
    db_tex_t tex;
    tex.persistId = ++Session.TexturePersistIdCounter;
    ReadImage(tex.bitmap, path);
    if (tex.bitmap.memory != NULL)
    {        
        CreateGPUTextureFromBitmap(&tex.gputex, (u8*)tex.bitmap.memory,
                        tex.bitmap.width,tex.bitmap.height,
                        GL_RGBA, (tex.bitmap.bitDepth == 3 ? GL_RGB : GL_RGBA));

        Assets.Textures.insert({tex.persistId, tex});
        LogMessage("Loaded %s with texture persist id %d", path, tex.persistId);
    }

    return tex;
}

void SessionResetFaceToolData()
{
    Session.SelectedFace = NULL;
}

NiceArray<MapEdit::Volume, 3200> EDITOR_MAP_VOLUMES;
std::vector<float> MY_VERTEX_BUFFER;

void LoadMapEditor()
{
    db_tex_t defaultTexureAsset = LoadNewTextureToDB(wd_path("default.png").c_str());
    LoadNewTextureToDB(texture_path("t_bpav2.bmp").c_str());
    LoadNewTextureToDB(texture_path("t_gf56464.bmp").c_str());
    LoadNewTextureToDB(texture_path("t_hzdg.bmp").c_str());
    LoadNewTextureToDB(texture_path("t_kgr2_p.bmp").c_str());
    LoadNewTextureToDB(texture_path("t_mbrk2_1.bmp").c_str());
    LoadNewTextureToDB(texture_path("t_vstnfcv.bmp").c_str());

    LoadNewTextureToDB(texture_path("example_5.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_7.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_9.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_10.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_14.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_16.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_17.jpg").c_str());
    LoadNewTextureToDB(texture_path("sld_gegfblock02b_64.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_19.jpg").c_str());
    LoadNewTextureToDB(texture_path("example_20.jpg").c_str());
    Session.SelectedTexture = defaultTexureAsset;
}

void CloseMapEditor()
{
    // TODO(Kevin)
}

#define MAPSER_VOLUMES_U64_DIVIDER_START 0x49779b0ef139ca39
#define MAPSER_VOLUMES_U64_DIVIDER_END   0x9fcf207673e66e63 
#define DESER_FACE_ADDITIONAL_SIZE 60

// TODO(Kevin): add binary format file data
// [4 bytes] : file type identifier
// [4 bytes] : 0x00000000
// [4 bytes] : file serial version 
// [4 bytes] : payload size n
// [n bytes] : payload
// [4 bytes] : EOF identifier

bool EditorSerializeMap(const char *mapFilePath)
{
#if INTERNAL_BUILD
    std::unordered_set<u32> SerPtrSet;
#endif

    ByteBuffer mapbuf;
    ByteBufferInit(&mapbuf);

    ByteBufferWrite(&mapbuf, u64, MAPSER_VOLUMES_U64_DIVIDER_START);
    const int volumesCount = EDITOR_MAP_VOLUMES.count;
    ByteBufferWrite(&mapbuf, int, volumesCount);

    for (int i = 0; i < volumesCount; ++i)
    {
        const MapEdit::Volume& volume = EDITOR_MAP_VOLUMES.At(i);
        
        ByteBufferWrite(&mapbuf, u64, volume.persistId);

        ByteBufferWrite(&mapbuf, size_t, volume.verts.size());
        for (MapEdit::Vert *v : volume.verts)
        {
#if INTERNAL_BUILD
            SerPtrSet.emplace(v->elemId);
#endif
            ByteBufferWrite(&mapbuf, u32, v->elemId);
            ByteBufferWrite(&mapbuf, float, v->pos.x);
            ByteBufferWrite(&mapbuf, float, v->pos.y);
            ByteBufferWrite(&mapbuf, float, v->pos.z);
        }

        ByteBufferWrite(&mapbuf, size_t, volume.edges.size());
        for (MapEdit::Edge *e : volume.edges)
        {
            u32 ep = e->elemId;
            u32 ea = e->a->elemId;
            u32 eb = e->b->elemId;
#if INTERNAL_BUILD
            ASSERT(SerPtrSet.find(ea) != SerPtrSet.end());
            ASSERT(SerPtrSet.find(eb) != SerPtrSet.end());
            SerPtrSet.emplace(ep);
#endif
            ByteBufferWrite(&mapbuf, u32, ep);
            ByteBufferWrite(&mapbuf, u32, ea);
            ByteBufferWrite(&mapbuf, u32, eb);
        }

        ByteBufferWrite(&mapbuf, size_t, volume.faces.size());
        for (MapEdit::Face *f : volume.faces)
        {
            // save the sequence of edges in the face's loop cycle by their pointers
            const std::vector<MapEdit::Edge*>& faceEdges = f->GetEdges();
            ByteBufferWrite(&mapbuf, size_t, faceEdges.size());
            for (MapEdit::Edge *fe : faceEdges)
            {
#if INTERNAL_BUILD
                ASSERT(SerPtrSet.find(fe->elemId) != SerPtrSet.end());
#endif
                ByteBufferWrite(&mapbuf, u32, fe->elemId);
            }
            u32 faceTexturePersistId = f->texture.persistId;
            ByteBufferWrite(&mapbuf, u32, faceTexturePersistId);
            // leaving some buffer room for additional Face data in the future
            // make sure to decrement as needed and mirror same value on deserialize
            u8 nullbytes[DESER_FACE_ADDITIONAL_SIZE];
            ByteBufferWriteBulk(&mapbuf, &nullbytes, DESER_FACE_ADDITIONAL_SIZE);
        }
    }

    ByteBufferWrite(&mapbuf, u64, MAPSER_VOLUMES_U64_DIVIDER_END);

    bool writtenToFile = ByteBufferWriteToFile(&mapbuf, mapFilePath) == 1;

    ByteBufferFree(&mapbuf);

    return writtenToFile;
}

bool EditorDeserializeMap(const char *mapFilePath)
{
    std::unordered_map<u32, void*> DeserElemIdToElem;

    ByteBuffer mapbuf;
    if (ByteBufferReadFromFile(&mapbuf, mapFilePath) == 0)
        return false;

    u64 u64slot;
    ByteBufferRead(&mapbuf, u64, &u64slot);
    ASSERT(u64slot == MAPSER_VOLUMES_U64_DIVIDER_START);

    int volumesCount = -1;
    ByteBufferRead(&mapbuf, int, &volumesCount);

    for (int i = 0; i < volumesCount; ++i)
    {
        MapEdit::Volume owner;
        ByteBufferRead(&mapbuf, u64, &owner.persistId);
        MapEdit::session_VolumePersistIdCounter = GM_max(owner.persistId, MapEdit::session_VolumePersistIdCounter);

        size_t vertexCount, edgeCount, faceCount = 0;

        ByteBufferRead(&mapbuf, size_t, &vertexCount);
        for (size_t j = 0; j < vertexCount; ++j)
        {
            u32 vElemId;
            vec3 vpos;
            ByteBufferRead(&mapbuf, u32, &vElemId);
            ByteBufferRead(&mapbuf, float, &vpos.x);
            ByteBufferRead(&mapbuf, float, &vpos.y);
            ByteBufferRead(&mapbuf, float, &vpos.z);

            void *elemptr = (void*)CreateVert(vpos, &owner); // MEM ALLOC
            DeserElemIdToElem.emplace(vElemId, elemptr);
        }

        ByteBufferRead(&mapbuf, size_t, &edgeCount);
        for (size_t j = 0; j < edgeCount; ++j)
        {
            u32 eElemId;
            u32 aElemId;
            u32 bElemId;
            ByteBufferRead(&mapbuf, u32, &eElemId);
            ByteBufferRead(&mapbuf, u32, &aElemId);
            ByteBufferRead(&mapbuf, u32, &bElemId);

            MapEdit::Vert *av = (MapEdit::Vert*)DeserElemIdToElem.at(aElemId);
            MapEdit::Vert *bv = (MapEdit::Vert*)DeserElemIdToElem.at(bElemId);
            void *elemptr = (void*)CreateEdge(av, bv, &owner); // MEM ALLOC
            DeserElemIdToElem.emplace(eElemId, elemptr);
        }

        ByteBufferRead(&mapbuf, size_t, &faceCount);
        for (size_t j = 0; j < faceCount; ++j)
        {
            size_t faceEdgesCount = 0;
            std::vector<MapEdit::Edge*> faceEdges;
            
            ByteBufferRead(&mapbuf, size_t, &faceEdgesCount);
            for (int k = 0; k < faceEdgesCount; ++k)
            {
                u32 feElemId;
                ByteBufferRead(&mapbuf, u32, &feElemId);
                faceEdges.push_back((MapEdit::Edge*)DeserElemIdToElem.at(feElemId));
            }

            u32 faceTexturePersistId;
            ByteBufferRead(&mapbuf, u32, &faceTexturePersistId);
            ByteBufferAdvancePosition(&mapbuf, DESER_FACE_ADDITIONAL_SIZE);

            MapEdit::Face *face = CreateFace(faceEdges, &owner); // MEM ALLOC

            MY_VERTEX_BUFFER.clear();
            TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER); // TODO(Kevin): do smarter triangulation...
            RebindFaceBatchBufferObject(&face->facemesh, (u32) MY_VERTEX_BUFFER.size() * sizeof(float), MY_VERTEX_BUFFER.data());
            
            face->texture = GetTextureByPersistId(faceTexturePersistId);
            face->facemesh.sharedTexture = face->texture.gputex;
        }

        EDITOR_MAP_VOLUMES.PushBack(owner);
    }

    ByteBufferRead(&mapbuf, u64, &u64slot);
    ASSERT(u64slot == MAPSER_VOLUMES_U64_DIVIDER_END);

    ByteBufferFree(&mapbuf);

    return true;
}


bool BuildGameMap(const char *path)
{
    ByteBuffer mapbuf;
    ByteBufferInit(&mapbuf);


    // need to save a bunch of FaceBatches (which are basically just meshes)
    // sort by texture, etc. every face using same texture goes into same FaceBatch
    // collider data -> just save the point clouds for mesh colliders and add to Octree while loading into game
    // need to do something smarter for texture data (want to use database so that textures are just enums)

    int totalfacecount = MapEdit::EDITOR_FACES.count;
    
    std::unordered_map<u32, std::vector<float>> VertexBuffers;
    std::vector<vec3> ColliderWorldPoints;
    std::vector<u32> ColliderSpans;

    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::EDITOR_FACES.At(i);

        std::vector<MapEdit::Vert*> faceVerts = face->GetVertices();

        u32 ColliderSpan = 0;
        for (MapEdit::Vert *v : faceVerts)
        {
            vec3 worldpos = v->pos;
            ColliderWorldPoints.push_back(worldpos);
            ++ColliderSpan;
        }
        ColliderSpans.push_back(ColliderSpan);
    }


    // LIGHT MAP BAKING
    Bounds MapBounds = Bounds(vec3(-0.17f, -0.17f, -0.17f), vec3(8000, 8000, 8000));
    Octree LightMapOcclusionTree = Octree(MapBounds, 8, 32);
    std::vector<StaticMeshCollider> MapSurfaceColliders(ColliderSpans.size());
    int iter = 0;
    // later, when only some surfaces have colliders, can't use ColliderSpan, need to traverse all faces again
    for (u32 i = 0; i < ColliderSpans.size(); ++i)
    {
        u32 span = ColliderSpans[i];
        StaticMeshCollider& surface = MapSurfaceColliders[i];
        surface.pointCloudPtr = &ColliderWorldPoints[iter];
        surface.pointCount = span;
        surface.debugId = i;
        iter += span;
        LightMapOcclusionTree.Insert(&surface);
    }
    // LightMapOcclusionTree is complete

    // Single bounce radiosity
    // Start with emitting patches, of which there are N.
    // Each emitting patch goes every other patch, of which there are M. O(NM)
    // Then, each patch (M) goes to every other patch (M) and transfers energy exactly once. O(NMM) 

    stbrp_rect *lm_rects = NULL;
    // calculate bounds, and divide into patches
    arrsetcap(all_lm_pos, 312500);
    arrsetcap(all_lm_norm, 312500);
    arrsetcap(all_lm_light, 312500);
    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::EDITOR_FACES.At(i);
        
        lm_face_t lm;
        // project 3D verts unto 2D plane using basis vectors U and V
        // find min uv and max uv
        vec3 v0 = face->loopbase->v->pos;
        vec3 v1 = face->loopbase->loopNext->v->pos;
        vec3 v2 = face->loopbase->loopNext->loopNext->v->pos;
        vec3 fn = Normalize(Cross(v1-v0, v2-v0));
        vec3 basisU = Normalize(v1-v0);
        vec3 basisV = Normalize(Cross(fn, basisU));

        vec2 minuv = vec2(FLT_MAX, FLT_MAX);
        vec2 maxuv = vec2(-FLT_MAX, -FLT_MAX);
        std::vector<MapEdit::Loop*> loopcycle = face->GetLoopCycle();

        for (MapEdit::Loop *loop : loopcycle)
        {
            vec3 p = loop->v->pos;
            float u = Dot(p-v0, basisU);
            float v = Dot(p-v0, basisV);
            minuv.x = fminf(minuv.x, u);
            minuv.y = fminf(minuv.y, v);
            maxuv.x = fmaxf(maxuv.x, u);
            maxuv.y = fmaxf(maxuv.y, v);
            vec2 basisUV = vec2(u,v);
            loop->lmuvcache = basisUV; // store basisUV for processing
        }
        
        const float lm_patch_size = 16.f;
        vec2 dim = maxuv - minuv;
        dim.x = ceilf(dim.x / lm_patch_size) * lm_patch_size;
        dim.y = ceilf(dim.y / lm_patch_size) * lm_patch_size;

        // lm uv
        for (MapEdit::Loop *loop : loopcycle)
        {
            // map lm uv to a local [0,1] value
            // after packing lm rects into larger light map atlas, these will be remapped
            loop->lmuvcache -= minuv;
            loop->lmuvcache.x /= dim.x;
            loop->lmuvcache.y /= dim.y;
            ASSERT(0.f <= loop->lmuvcache.x && loop->lmuvcache.x <= 1.f);
            ASSERT(0.f <= loop->lmuvcache.y && loop->lmuvcache.y <= 1.f);
        }

        // lm data
        lm.w = (i32)(dim.x / lm_patch_size);
        lm.h = (i32)(dim.y / lm_patch_size);
        i32 lmsz = lm.w*lm.h;
        lm.pos = arraddnptr(all_lm_pos, lmsz);
        lm.norm = arraddnptr(all_lm_norm, lmsz);
        lm.light = arraddnptr(all_lm_light, lmsz);
        for (i32 pi = 0; pi < lmsz; ++pi)
        {
            i32 x = pi % lm.w;
            i32 y = pi / lm.w;
            float uSampleCenter = minuv.x + lm_patch_size*x + lm_patch_size*0.5f;
            float vSampleCenter = minuv.y + lm_patch_size*y + lm_patch_size*0.5f;
            // reverse the projection
            vec3 p = v0 + uSampleCenter * basisU + vSampleCenter * basisV;
            lm.pos[pi] = p;
            lm.norm[pi] = fn;
            lm.light[pi] = pi*255/lmsz;
        }

        face->lightmap = lm;
        stbrp_rect rect;
        rect.id = face->EDITOR_FACES_INDEX;
        rect.w = lm.w;
        rect.h = lm.h;
        arrput(lm_rects, rect);
    }

    // direct lighting from emissive points and patches

    // indirect lighting between patches

    // pack light maps
    i32 lightMapAtlasW = 400;
    i32 lightMapAtlasH = 400;
    u8 *LIGHT_MAP_ATLAS = NULL;
    arrsetcap(LIGHT_MAP_ATLAS, lightMapAtlasW*lightMapAtlasH);
    stbrp_node *LMPackerNodes = NULL;
    arrsetlen(LMPackerNodes, 1000);
    stbrp_context LightMapPacker;
    stbrp_init_target(&LightMapPacker, lightMapAtlasW, lightMapAtlasH, LMPackerNodes, (int)arrlenu(LMPackerNodes));
    stbrp_pack_rects(&LightMapPacker, lm_rects, (int)arrlenu(lm_rects));
    arrfree(LMPackerNodes);
    size_t rectcount = arrlenu(lm_rects);
    for (size_t i = 0; i < rectcount; ++i)
    {
        stbrp_rect rect = lm_rects[i];
        if (rect.was_packed == 0) continue;
        //ASSERT(rect.was_packed != 0); // TODO(Kevin): additional light map atlases if couldn't fit into one

        vec2 minuv = vec2((float)rect.x / (float)lightMapAtlasW, (float)rect.y / (float)lightMapAtlasH);
        vec2 maxuv = vec2((float)(rect.x + rect.w) / (float)lightMapAtlasW, (float)(rect.y + rect.h) / (float)lightMapAtlasH);

        int editorFaceIndex = rect.id;
        MapEdit::Face *face = MapEdit::EDITOR_FACES.At(editorFaceIndex);
        std::vector<MapEdit::Loop*> loopcycle = face->GetLoopCycle();
        for (MapEdit::Loop *loop : loopcycle)
        {
            // map lm uv from local to global in light map atlas
            loop->lmuvcache.x = Lerp(minuv.x, maxuv.x, loop->lmuvcache.x); 
            loop->lmuvcache.y = Lerp(minuv.y, maxuv.y, loop->lmuvcache.y); 
        }
        lm_face_t lmface = face->lightmap;
        BlitRect((u8*)LIGHT_MAP_ATLAS, lightMapAtlasW, lightMapAtlasH, (u8*)lmface.light, lmface.w, lmface.h, rect.x, rect.y, sizeof(u8));
    }

    ByteBufferWrite(&mapbuf, i32, lightMapAtlasW);
    ByteBufferWrite(&mapbuf, i32, lightMapAtlasH);
    ByteBufferWriteBulk(&mapbuf, LIGHT_MAP_ATLAS, lightMapAtlasW*lightMapAtlasH*sizeof(u8));

    // save out light map atlases as part of .MAP
    arrfree(LIGHT_MAP_ATLAS);
    arrfree(lm_rects);
    arrfree(all_lm_pos);
    arrfree(all_lm_norm);
    arrfree(all_lm_light);


    // Multithread later
    // for the indirect calculation, the direct lightmap stays constant, so divide up the indirect lightmap
    // patches across threads.


    // LineCollider ray;
    // ray.a = vec3(0,4,0);
    // ray.b = vec3(0,-500,0);
    // std::unordered_set<Collider*> query = LightMapOcclusionTree.Query(ray);
    // for (Collider *surface : query)
    // {
    //     ColResult result = GJK(surface, &ray);
    //     if (result.hasCollision)
    //     {
    //     }
    // }


    // sort faces by their textures
    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::EDITOR_FACES.At(i);
        db_tex_t tex = face->texture;
        if (VertexBuffers.find(tex.persistId) == VertexBuffers.end())
        {
            VertexBuffers.emplace(tex.persistId, std::vector<float>());
        }

        std::vector<float>& vb = VertexBuffers.at(tex.persistId);
        TriangulateFace_QuickDumb_LMUV(*face, &vb);
    }

    // colliders
    size_t numColliderPoints = ColliderWorldPoints.size();
    size_t numColliderSpans = ColliderSpans.size();
    ByteBufferWrite(&mapbuf, size_t, numColliderPoints);
    ByteBufferWrite(&mapbuf, size_t, numColliderSpans);
    ByteBufferWriteBulk(&mapbuf, ColliderWorldPoints.data(), sizeof(vec3)*numColliderPoints);
    ByteBufferWriteBulk(&mapbuf, ColliderSpans.data(), sizeof(u32)*numColliderSpans);

    // vertex buffers
    size_t numVertexBufs = VertexBuffers.size();
    ByteBufferWrite(&mapbuf, size_t, numVertexBufs);
    for (auto& vbpair : VertexBuffers)
    {
        u32 texturePersistId = vbpair.first;
        const std::vector<float>& vb = vbpair.second;

        ByteBufferWrite(&mapbuf, u32, texturePersistId);
        ByteBufferWrite(&mapbuf, size_t, vb.size());
        ByteBufferWriteBulk(&mapbuf, (void*)vb.data(), sizeof(float)*vb.size());
    }

    bool writtenToFile = ByteBufferWriteToFile(&mapbuf, path) == 1;
    ByteBufferFree(&mapbuf);
    return writtenToFile;
}

bool LoadGameMap(const char *path)
{
    std::unordered_map<u32, void*> DeserElemIdToElem;

    // deserialize
    ByteBuffer mapbuf;
    if (ByteBufferReadFromFile(&mapbuf, path) == 0)
        return false;

    i32 lmw, lmh;
    ByteBufferRead(&mapbuf, i32, &lmw);
    ByteBufferRead(&mapbuf, i32, &lmh);
    u8 *lightMapData = NULL; 
    arrsetcap(lightMapData, lmw*lmh);
    ByteBufferReadBulk(&mapbuf, lightMapData, lmw*lmh*sizeof(u8));
    CreateGPUTextureFromBitmap(&POSCOLORTESTTEX, (u8*)lightMapData, lmw, lmh,
        GL_RGB, GL_RED, GL_NEAREST, GL_NEAREST, GL_UNSIGNED_BYTE);
    arrfree(lightMapData);

    // colliders
    size_t numColliderPoints, numColliderSpans;
    ByteBufferRead(&mapbuf, size_t, &numColliderPoints);
    ByteBufferRead(&mapbuf, size_t, &numColliderSpans);

    ASSERT(GameLevelColliders.size() == 0); // just crash for now
    GameLevelColliderPoints.resize(numColliderPoints);
    GameLevelColliders.resize(numColliderSpans);

    std::vector<u32> ColliderSpans(numColliderSpans); 
    ByteBufferReadBulk(&mapbuf, GameLevelColliderPoints.data(), sizeof(vec3)*numColliderPoints);
    ByteBufferReadBulk(&mapbuf, ColliderSpans.data(), sizeof(u32)*numColliderSpans);
    
    int GameLevelColliderPointsIterator = 0;
    for (u32 colliderIndex = 0; colliderIndex < numColliderSpans; ++colliderIndex)
    {
        u32 span = ColliderSpans[colliderIndex];
        StaticMeshCollider& collider = GameLevelColliders[colliderIndex];
        collider.pointCloudPtr = &GameLevelColliderPoints[GameLevelColliderPointsIterator];
        collider.pointCount = span;
        GameLevelColliderPointsIterator += span;
    }
    ASSERT(GameLevelCollisionTree.root.get() == NULL); // i don't care. crash for now.
    Bounds GameLevelWorldBound = Bounds(vec3(-0.17f, -0.17f, -0.17f), vec3(8000, 8000, 8000));
    GameLevelCollisionTree = Octree(GameLevelWorldBound, 7, 24); // NOTE(Kevin): maybe I can be smarter about how I'm deciding the values for max tree depth and max count per octant
    for (StaticMeshCollider& collider : GameLevelColliders)
    {
        GameLevelCollisionTree.Insert(&collider);
    }

    // vertex buffers
    size_t numVertexBufs;
    ByteBufferRead(&mapbuf, size_t, &numVertexBufs);
    for (int i = 0; i < numVertexBufs; ++i)
    {
        u32 texturePersistId;
        ByteBufferRead(&mapbuf, u32, &texturePersistId);

        size_t vertexcount;
        ByteBufferRead(&mapbuf, size_t, &vertexcount);

        std::vector<float> vb;
        vb.resize(vertexcount);
        ByteBufferReadBulk(&mapbuf, vb.data(), sizeof(float)*vertexcount);
        ASSERT(vb.size() == vertexcount);
        
        FaceBatch batch;
        batch.sharedTexture = POSCOLORTESTTEX;//GetTextureByPersistId(texturePersistId).gputex;
        CreateFaceBatchMeshOnGPU(&batch.idVAO, &batch.idVBO);
        RebindFaceBatchBufferObject(&batch, u32(sizeof(float)*vertexcount), vb.data());
        GameLevelFaceBatches.push_back(batch);
    }

    ByteBufferFree(&mapbuf);

    return true;
}




vec3 editorCameraPosition = vec3(600, 500, 600);
float DISC_HANDLE_RADIUS = 10.f;
// These indices are not guaranteed to persist frame to frame. For now (2024-09-04) they index into
// the huge array of all Volumes EDITOR_MAP_VOLUMES. Things will/should move around within the array. 
NiceArray<int, 16> SELECTED_MAP_VOLUMES_INDICES;
std::vector<MapEdit::Vert*> SELECTABLE_VERTICES;
std::vector<MapEdit::Vert*> SELECTED_VERTICES;
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

/// === GRID DEFINITION ===
// I want the grid to be defined by a "UP" and a "RIGHT"
// based on these, I can figure out the orientation of the grid
float GRID_INCREMENT = 32.f;
vec3 GRID_ORIGIN = vec3();
vec3 GRID_UP_VECTOR = GM_UP_VECTOR;
vec3 GRID_RIGHT_VECTOR = GM_RIGHT_VECTOR;

void ResetGridOriginAndOrientation()
{
    GRID_ORIGIN = vec3();
    GRID_UP_VECTOR = GM_UP_VECTOR;
    GRID_RIGHT_VECTOR = GM_RIGHT_VECTOR;
}

mat3 GetGridRotationMatrix()
{
    mat3 rot;
    rot.columns[0] = Normalize(Cross(GRID_RIGHT_VECTOR, GRID_UP_VECTOR)); // forward
    rot.columns[1] = Normalize(GRID_UP_VECTOR); // up
    rot.columns[2] = Normalize(GRID_RIGHT_VECTOR); // right
    return rot;
}

float SnapToGrid(float beforeSnap)
{
    return roundf(beforeSnap / GRID_INCREMENT) * GRID_INCREMENT;
}

vec3 SnapToGrid(vec3 beforeSnap)
{
    vec3 snaptemp = beforeSnap;
    mat3 rot = GetGridRotationMatrix();
    mat3 invRot = rot.GetInverse();
    snaptemp = snaptemp - GRID_ORIGIN;
    snaptemp = invRot * snaptemp;
    snaptemp = vec3(SnapToGrid(snaptemp.x), SnapToGrid(snaptemp.y), SnapToGrid(snaptemp.z));
    snaptemp = rot * snaptemp;
    snaptemp = snaptemp + GRID_ORIGIN;
    return snaptemp;
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


Octree CollisionTree;

void DrawOctreeNode(OctreeNode *node)
{
    vec3 center = node->bounds.center;
    vec3 size = node->bounds.size;
    vec3 corners[] = { 
        vec3(center.x - size.x/2.f, center.y - size.y/2.f, center.z - size.z/2.f),
        vec3(center.x + size.x/2.f, center.y - size.y/2.f, center.z - size.z/2.f),
        vec3(center.x - size.x/2.f, center.y + size.y/2.f, center.z - size.z/2.f),
        vec3(center.x - size.x/2.f, center.y - size.y/2.f, center.z + size.z/2.f),
        vec3(center.x - size.x/2.f, center.y + size.y/2.f, center.z + size.z/2.f),
        vec3(center.x + size.x/2.f, center.y + size.y/2.f, center.z - size.z/2.f),
        vec3(center.x + size.x/2.f, center.y - size.y/2.f, center.z + size.z/2.f),
        vec3(center.x + size.x/2.f, center.y + size.y/2.f, center.z + size.z/2.f)
    }; 

    vec4 octantColor = vec4(0, 1, 0, 1);
    // if (node->objects.size() > 0)
    // {
    //     octantColor = vec4(1,0,0,1);
    // }

    PrimitiveDrawLine(corners[0], corners[1], octantColor);
    PrimitiveDrawLine(corners[1], corners[6], octantColor);
    PrimitiveDrawLine(corners[6], corners[3], octantColor);
    PrimitiveDrawLine(corners[3], corners[0], octantColor);
    
    PrimitiveDrawLine(corners[2], corners[4], octantColor);
    PrimitiveDrawLine(corners[4], corners[7], octantColor);
    PrimitiveDrawLine(corners[7], corners[5], octantColor);
    PrimitiveDrawLine(corners[5], corners[2], octantColor);
    
    PrimitiveDrawLine(corners[0], corners[2], octantColor);
    PrimitiveDrawLine(corners[1], corners[5], octantColor);
    PrimitiveDrawLine(corners[6], corners[7], octantColor);
    PrimitiveDrawLine(corners[3], corners[4], octantColor);


    if (node->isLeaf)
        return;
    else
    {
        for (int i = 0; i < 8; ++i)
            DrawOctreeNode(&node->children[i]);
    }
}

void VisualizeOctreeHack()
{
    Bounds worldBounds;
    worldBounds.center = vec3(-0.17f, -0.17f, -0.17f); // little bit off origin so octant bounds aren't strictly along grid
    worldBounds.size = vec3(3200,3200,3200);
    CollisionTree = Octree(worldBounds,32,2);

    int totalfacecount = MapEdit::EDITOR_FACES.count;
    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::EDITOR_FACES.At(i);

        MeshCollider *col = new MeshCollider();
        std::vector<MapEdit::Vert*> points = face->GetVertices();
        for (MapEdit::Vert *v : points)
        {
            col->pointCloud.push_back(v->pos);
        }

        CollisionTree.Insert(col);
    }
}


u32 PickVolume(MapEdit::Volume *volumes, u32 arraycount)
{
    // returns the 1 + index of the volume in the provided array. 0 is nothing picked.
    for (u32 volumeFrameId = 1; volumeFrameId <= arraycount; ++volumeFrameId)
    {
        vec3 idrgb = HandleIdToRGB(volumeFrameId);
        MapEdit::Volume& vol = volumes[volumeFrameId - 1];
        for (MapEdit::Face *f : vol.faces)
        {
            int count;
            // TODO(Kevin): instead of triangulate every face every time, probably cache this somehow?
            MapEdit::TriangulateFace_QuickDumb_WithColor(*f, idrgb, HANDLES_VB.data + HANDLES_VB.count, &count);
            HANDLES_VB.count += count;
        }
    }
    u32 pickedVolumeFrameId = FlushHandles(mousepos, renderTargetGame, activeViewMatrix, activePerspectiveMatrix, false);
    return pickedVolumeFrameId;
}

u32 PickFace(MapEdit::Face **faces, u32 arraycount)
{
    // returns 1 + index of the face in the provided array. 0 is nothing picked.
    for (u32 id = 1; id <= arraycount; ++id)
    {
        MapEdit::Face *face = faces[id-1];
        vec3 idrgb = HandleIdToRGB(id);
        int count;
        // TODO(Kevin): instead of triangulate every face every time, probably cache this somehow?
        MapEdit::TriangulateFace_QuickDumb_WithColor(*face, idrgb, HANDLES_VB.data + HANDLES_VB.count, &count);
        HANDLES_VB.count += count;
    }
    u32 faceId = FlushHandles(mousepos, renderTargetGame, activeViewMatrix, activePerspectiveMatrix, false);
    return faceId;
}

void TickMapEditor()
{
    IsEditorActive = true;
    SDL_SetRelativeMouseMode(mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT) ? SDL_TRUE : SDL_FALSE);

    // if (keyspressed[SDL_SCANCODE_J])
    // {
    //     VisualizeOctreeHack();
    // }
    // if (CollisionTree.root.get())
    //     DrawOctreeNode(CollisionTree.root.get());

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

    // DRAW AXIS LINES
    PrimitiveDrawLine(vec3(320000.f, 0.f, 0.f), vec3(-320000.f, 0.f, 0.f), vec4(RGB255TO1(205, 56, 9), 0.8f));
    // PrimitiveDrawLine(vec3(0.f, 320000.f, 0.f), vec3(0.f, -320000.f, 0.f), vec4(RGB255TO1(67, 123, 9), 0.8f));
    PrimitiveDrawLine(vec3(0.f, 0.f, 320000.f), vec3(0.f, 0.f, -320000.f), vec4(RGB255TO1(21, 129, 205), 0.8f));


    static u32 hotHandleId = 0;

    Gui::BeginWindow(Gui::UIRect(0,0,renderTargetGUI.width,19));
    Gui::EditorBeginHorizontal();
    if (Gui::EditorLabelledButton("SAVE"))
    {
        std::string path = SaveEditableMapFormatDialog();
        if(!path.empty())
        {
            // TODO ...
            if (EditorSerializeMap(path.c_str()))
                LogMessage("Saved %s", path.c_str());
            else
                LogError("Failed to save to %s", path.c_str());
            // TODO ...
        }
    }
    if (Gui::EditorLabelledButton("OPEN"))
    {
        std::string path = OpenEditableMapFormatDialog();
        if(!path.empty())
        {
            // TODO ...
            if (EditorDeserializeMap(path.c_str()))
                LogMessage("Opened %s", path.c_str());
            else
                LogError("Failed to open %s", path.c_str());
            // TODO ...
        }
    }
    if (Gui::EditorLabelledButton("BUILD"))
    {
        std::string path = SaveGameMapDialog();
        if(!path.empty())
        {
            // TODO ...
            if (BuildGameMap(path.c_str()))
                LogMessage("Built %s", path.c_str());
            else
                LogError("Failed to build to %s", path.c_str());
            // TODO ...
        }
    }
    Gui::EditorSpacer(16,0);

    bool brushToolActive = editorActiveTool == MapEditorTools::SimpleBrushTool;
    bool vertToolActive = editorActiveTool == MapEditorTools::VertexManip;
    bool edgeToolActive = editorActiveTool == MapEditorTools::EdgeManip;
    bool faceToolActive = editorActiveTool == MapEditorTools::FaceManip;
    if (keyspressed[SDL_SCANCODE_1] || Gui::EditorSelectable_2("BRUSH", &brushToolActive))
    {
        editorActiveTool = MapEditorTools::SimpleBrushTool;
        hotHandleId = 0;
        SELECTED_MAP_VOLUMES_INDICES.ResetCount();
        SessionResetFaceToolData();
    }
    if (keyspressed[SDL_SCANCODE_2] || Gui::EditorSelectable_2("VERT", &vertToolActive))
    {
        editorActiveTool = MapEditorTools::VertexManip;
        hotHandleId = 0;
        SessionResetFaceToolData();
    }
    if (keyspressed[SDL_SCANCODE_3] || Gui::EditorSelectable_2("EDGE", &edgeToolActive))
    {
        editorActiveTool = MapEditorTools::EdgeManip;
        hotHandleId = 0;
        SessionResetFaceToolData();
    }
    if (keyspressed[SDL_SCANCODE_4] || Gui::EditorSelectable_2("FACE", &faceToolActive))
    {
        editorActiveTool = MapEditorTools::FaceManip;
        hotHandleId = 0;
        SessionResetFaceToolData();
    }

    Gui::EditorSpacer(16,0);
    if (Gui::EditorLabelledButton("-", 18))
    {
        GRID_INCREMENT /= 2.f;
    }
    Gui::EditorSpacer(1,0);
    Gui::EditorText(std::to_string(GRID_INCREMENT).c_str());
    Gui::EditorSpacer(1,0);
    if (Gui::EditorLabelledButton("+", 18))
    {
        GRID_INCREMENT *= 2.f;
    }

    Gui::EditorSpacer(16,0);
    Gui::EditorLabelledButton("Toggle Auto Colliders []");
    Gui::EditorLabelledButton("Snap Vertex or Translation to Grid?");

    Gui::EditorEndHorizontal();
    Gui::EndWindow();


    int volc = SELECTED_MAP_VOLUMES_INDICES.count;
    Gui::BeginWindow(Gui::UIRect(renderTargetGUI.width - 160, 28, 150, 15 + volc * 15));
    Gui::EditorText((std::string("selected volumes (") + std::to_string(volc) + std::string(")")).c_str());
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        int editorMapVolumeIndex = SELECTED_MAP_VOLUMES_INDICES.At(i);
        const MapEdit::Volume& volume = EDITOR_MAP_VOLUMES.At(editorMapVolumeIndex);
        std::string volPersIdStr = std::to_string(volume.persistId);
        Gui::EditorBeginHorizontal();
        // Gui::EditorLabelledButton("S");
        if (Gui::EditorLabelledButton("X"))
        {
            // TODO remove volume from selection
        }
        Gui::EditorSpacer(4, 0);
        Gui::EditorText(volPersIdStr.c_str());
        Gui::EditorEndHorizontal();
    }
    // Gui::EditorText((std::string("            max ") + std::to_string(SELECTED_MAP_VOLUMES_INDICES.capacity)).c_str());
    Gui::EndWindow();


    Gui::BeginWindow(Gui::UIRect(renderTargetGUI.width - 210, renderTargetGUI.height - 310, 200, 300));
    Gui::EditorText("Texture");
    
    static bool ShowTextureBrowser = false;
    if (Gui::EditorImageButton(Session.SelectedTexture.gputex.id, ivec2(160,160)))
    {
        ShowTextureBrowser = true;
    }

    if (ShowTextureBrowser)
    {
        Gui::BeginWindow(Gui::UIRect(0, 0, renderTargetGUI.width, renderTargetGUI.height));
        
        Gui::EditorBeginHorizontal();
        if (Gui::EditorLabelledButton("Close"))
            ShowTextureBrowser = false;
        Gui::EditorSpacer(4,0);
        if (Gui::EditorLabelledButton("Load Textures"))
        {
            std::vector<std::string> paths = OpenImageFilesDialog();
            for (const auto& fp : paths)
            {
                db_tex_t loadedtex = LoadNewTextureToDB(fp.c_str());
            }
        }
        Gui::EditorEndHorizontal();

        Gui::EditorSpacer(0,8);
        Gui::EditorBeginGrid(renderTargetGUI.width, 8000);
        for (auto& pair : Assets.Textures)
        {
            Gui::EditorBeginGridItem(162,186);

            db_tex_t t = pair.second;
            if (Gui::EditorImageButton(t.gputex.id, ivec2(160,160)))
            {
                Session.SelectedTexture = t;
                ShowTextureBrowser = false;
            }
            char displaystr[128] = {0};
            stbsp_sprintf(displaystr, "ID: %d\t\t%dx%d", t.persistId, t.gputex.width, t.gputex.height);
            Gui::EditorText(displaystr);

            Gui::EditorEndGridItem();
        }
        Gui::EditorEndGrid();

        Gui::EndWindow();
    }

    // TODO(Kevin): Have shortcuts for these actions so you can go click click shortcut click click shortcut
    Gui::EditorLabelledButton("Apply Texture to Volume");
    if (Session.SelectedFace != NULL && Gui::EditorLabelledButton("Apply Texture to Face"))
    {
        Session.SelectedFace->texture = Session.SelectedTexture; 
        Session.SelectedFace->facemesh.sharedTexture = Session.SelectedTexture.gputex; 
    }
    Gui::EndWindow();


    if (Gui::anyElementHovered || Gui::anyWindowHovered)
    {
        // TODO tools reset? if LMB relased? not always.
        return;
    }


    ResetGridOriginAndOrientation();


    // === Volume picking ===
    // Triangulation on the fly is efficient but I don't care right now
    // can pick volumes while in most modes
    // can pick nothing to deselect in most modes
    bool volumePickable = 
        editorActiveTool == MapEditorTools::VertexManip ||
        editorActiveTool == MapEditorTools::EdgeManip ||
        editorActiveTool == MapEditorTools::FaceManip;
    if (volumePickable && mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT) && hotHandleId == 0)
    {
        u32 pickedVolumeFrameId = PickVolume(EDITOR_MAP_VOLUMES.data, EDITOR_MAP_VOLUMES.count);
        if (pickedVolumeFrameId <= 0)
        {
            SELECTED_MAP_VOLUMES_INDICES.ResetCount();
            SessionResetFaceToolData();
        }
        else
        {
            if (!keyscurrent[SDL_SCANCODE_LCTRL])
            {
                SELECTED_MAP_VOLUMES_INDICES.ResetCount();
                SessionResetFaceToolData();
            }

            int pickedVolumeIndexInMegaArray = pickedVolumeFrameId-1;
            bool exists = false;
            for (int j = 0; j < SELECTED_MAP_VOLUMES_INDICES.count; ++j)
                exists |= SELECTED_MAP_VOLUMES_INDICES.At(j) == pickedVolumeIndexInMegaArray;
            if (!exists && SELECTED_MAP_VOLUMES_INDICES.count < SELECTED_MAP_VOLUMES_INDICES.capacity)
                SELECTED_MAP_VOLUMES_INDICES.PushBack(pickedVolumeIndexInMegaArray);
        }

    }


    // === Selection handles ===

    // on mouse up, disc remains selected
    // on mouse down, if disc not selected, then if ctrl pressed then add to selection but don't start drag
    //                                      if ctrl not pressed then start drag
    // on mouse down, if disc was already selected, then if ctrl pressed then deselect that disc/vert, if ctrl not pressed
    // then start checking for drag

    SELECTABLE_VERTICES.clear();
    SELECTABLE_FACES.clear();
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        const MapEdit::Volume& volume = EDITOR_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));

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
        static vec3 DragPlanePoint;

        bool lmbPressedThisFrame = mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT);
        bool lmbReleasedThisFrame = mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT);
        bool lmbIsPressed = mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT);

        u32 hoveredFaceId = PickFace(SELECTABLE_FACES.data(), (u32)SELECTABLE_FACES.size());

        if (hoveredFaceId > 0)
        {
            MapEdit::Face *face = SELECTABLE_FACES[hoveredFaceId-1];
            face->hovered = true;

            if (lmbPressedThisFrame)
            {
                hotHandleId = hoveredFaceId;
                Session.SelectedFace = face;

                SELECTED_VERTICES = face->GetVertices();

                for (MapEdit::Vert *vert : SELECTED_VERTICES)
                    vert->poscache = vert->pos;

                vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);
                IntersectPlaneAndLineWithDirections(face->loopbase->v->pos, face->QuickNormal(), worldpos_mouse, worldray_mouse, &DragPlanePoint);
            }
        }

        if (lmbIsPressed && hotHandleId > 0)
        {
            MapEdit::Face *hotFace = SELECTABLE_FACES[hotHandleId-1];

            vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);

            vec3 TotalTranslation;
            if (keyscurrent[SDL_SCANCODE_LALT])
            {
                vec3 intersect;
                IntersectPlaneAndLineWithDirections(DragPlanePoint, vec3(-cameraDirection.x, 0.f, -cameraDirection.z), worldpos_mouse, worldray_mouse, &intersect);
                float yTranslation = Dot((intersect - DragPlanePoint), GM_UP_VECTOR);
                TotalTranslation = vec3(0.f,yTranslation,0.f);
                TotalTranslation = SnapToGrid(TotalTranslation);
            }
            else
            {
                vec3 intersect;
                IntersectPlaneAndLine(DragPlanePoint, GM_UP_VECTOR, worldpos_mouse, worldray_mouse, &intersect);
                TotalTranslation = intersect - DragPlanePoint;
                TotalTranslation = SnapToGrid(TotalTranslation);
            }

            if (keyscurrent[SDL_SCANCODE_ESCAPE])
            {
                for (MapEdit::Vert *vert : SELECTED_VERTICES)
                    vert->pos = vert->poscache;
                hotHandleId = 0;
            }
            else
            {
                for (MapEdit::Vert *vert : SELECTED_VERTICES)
                {
                    vert->pos = vert->poscache + TotalTranslation;
                }
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

        if (lmbReleasedThisFrame && hotHandleId > 0)
        {
            hotHandleId = 0;
        }
    }

    if (editorActiveTool == MapEditorTools::VertexManip)
    {
        static vec3 DragPlanePoint;
        static bool LCtrlDownOnLeftMouseDown = false;
        static bool AlreadySelectedOnLeftMouseDown = false;
        static vec3 TotalTranslation;

        if (mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
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
                MapEdit::Vert *hotVert = SELECTABLE_VERTICES[hotHandleId-1];
                auto clickedIter = std::find(SELECTED_VERTICES.begin(), SELECTED_VERTICES.end(), hotVert);
                AlreadySelectedOnLeftMouseDown = clickedIter != SELECTED_VERTICES.end();
                if (!AlreadySelectedOnLeftMouseDown)
                {
                    if (!LCtrlDownOnLeftMouseDown)
                        SELECTED_VERTICES.clear();
                    SELECTED_VERTICES.push_back(hotVert);
                    hotVert->poscache = hotVert->pos;
                }

                vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);
                IntersectPlaneAndLineWithDirections(hotVert->pos, -cameraDirection, worldpos_mouse, worldray_mouse, &DragPlanePoint);
            }
            else if (!LCtrlDownOnLeftMouseDown)
            {
                SELECTED_VERTICES.clear();
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

            TotalTranslation = vec3();
            if (keyscurrent[SDL_SCANCODE_LALT])
            {
                vec3 intersect;
                IntersectPlaneAndLineWithDirections(DragPlanePoint, vec3(-cameraDirection.x, 0.f, -cameraDirection.z), worldpos_mouse, worldray_mouse, &intersect);
                float yTranslation = Dot((intersect - DragPlanePoint), GM_UP_VECTOR);
                TotalTranslation = vec3(0.f,yTranslation,0.f);
                TotalTranslation = SnapToGrid(TotalTranslation);
            }
            else
            {
                vec3 intersect;
                IntersectPlaneAndLine(DragPlanePoint, GM_UP_VECTOR, worldpos_mouse, worldray_mouse, &intersect);
                TotalTranslation = intersect - DragPlanePoint;
                TotalTranslation = SnapToGrid(TotalTranslation);
            }

            if (keyscurrent[SDL_SCANCODE_ESCAPE])
            {
                for (MapEdit::Vert *vert : SELECTED_VERTICES)
                    vert->pos = vert->poscache;
                hotHandleId = 0;
                LCtrlDownOnLeftMouseDown = false;
                AlreadySelectedOnLeftMouseDown = false;
            }
            else
            {
                for (MapEdit::Vert *vert : SELECTED_VERTICES)
                {
                    vert->pos = vert->poscache + TotalTranslation;
                }
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

        if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
            if (hotHandleId > 0)
            {
                MapEdit::Vert *hotVert = SELECTABLE_VERTICES[hotHandleId-1];
                if (LCtrlDownOnLeftMouseDown)
                {
                    auto hotVertIter = std::find(SELECTED_VERTICES.begin(), SELECTED_VERTICES.end(), hotVert);
                    if (hotVertIter != SELECTED_VERTICES.end())
                    {
                        if (AlreadySelectedOnLeftMouseDown && TotalTranslation != vec3())
                            SELECTED_VERTICES.erase(hotVertIter);
                    }
                    else
                    {
                        LogError("wtf");
                    }
                }
                else if (TotalTranslation != vec3())
                {
                    SELECTED_VERTICES.clear();
                    SELECTED_VERTICES.push_back(hotVert);
                }
            }

            hotHandleId = 0;
            LCtrlDownOnLeftMouseDown = false;
            AlreadySelectedOnLeftMouseDown = false;
        }
    }

    if (editorActiveTool == MapEditorTools::SimpleBrushTool)
    {
        static vec3 rectstartpoint;
        static vec3 drawingplanenormal;
        static vec3 drawinghorizontal;
        static vec3 drawingvertical;
        static vec3 rectendpoint;
        switch (simpleBrushToolState)
        {
            case SimpleBrushToolState::NotActive:
                if (mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    u32 pickedVolumeFrameId = PickVolume(EDITOR_MAP_VOLUMES.data, EDITOR_MAP_VOLUMES.count);
                    if (pickedVolumeFrameId > 0)
                    {
                        MapEdit::Volume& vol = EDITOR_MAP_VOLUMES.At(pickedVolumeFrameId-1);
                        std::vector<MapEdit::Face*>& faces = vol.faces;
                        u32 faceIndex = PickFace(faces.data(), (u32)faces.size());
                        MapEdit::Face *drawingFace = faces[faceIndex-1];
                        drawingplanenormal = drawingFace->QuickNormal();
                        vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                        vec3 wr = ScreenPointToWorldRay(mousepos);
                        vec3 intersectionPoint;
                        IntersectPlaneAndLineWithDirections(drawingFace->loopbase->v->pos, drawingplanenormal, ws, wr, &intersectionPoint);
                        simpleBrushToolState = SimpleBrushToolState::DrawingRectangle;
                        rectstartpoint = intersectionPoint;
                    }
                    else
                    {
                        vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                        vec3 wr = ScreenPointToWorldRay(mousepos);
                        float f = (0.f - ws.y) / wr.y;
                        simpleBrushToolState = SimpleBrushToolState::DrawingRectangle;
                        rectstartpoint = ws + wr * f;
                        drawingplanenormal = GM_UP_VECTOR;
                    }
                    rectstartpoint = SnapToGrid(rectstartpoint);
                    // no matter what, at least one of these vectors is GM_RIGHT_VECTOR rotated around y-axis.
                    vec3 flattenedNormal = vec3(drawingplanenormal.x, 0.f, drawingplanenormal.z);
                    if (Magnitude(flattenedNormal) < 0.001f)
                        flattenedNormal = GM_FORWARD_VECTOR;
                    else if (GM_abs(drawingplanenormal.y) < 0.001f)
                        flattenedNormal = GM_UP_VECTOR;
                    else
                        flattenedNormal = Normalize(flattenedNormal);
                    drawinghorizontal = Normalize(Cross(drawingplanenormal, flattenedNormal));
                    drawingvertical = Normalize(Cross(drawingplanenormal, drawinghorizontal));
                }
                break;
            case SimpleBrushToolState::DrawingRectangle:
                GRID_ORIGIN = rectstartpoint;
                GRID_RIGHT_VECTOR = drawinghorizontal;
                GRID_UP_VECTOR = drawingplanenormal;
                if (mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    vec3 intersection;
                    IntersectPlaneAndLineWithDirections(rectstartpoint, drawingplanenormal, ws, wr, &intersection);
                    vec3 endpoint = intersection;
                    endpoint = SnapToGrid(endpoint);
                    vec3 startToEndVector = vec3(endpoint - rectstartpoint);

                    // PrimitiveDrawSolidDisc(rectstartpoint, drawingplanenormal, 4, vec4(0,0,1,1));
                    // PrimitiveDrawSolidDisc(endpoint, drawingplanenormal, 4, vec4(0,0,1,1));
                    // PrimitiveDrawLine(rectstartpoint, rectstartpoint + drawinghorizontal * 16.f, vec4(0,0,1,1), 2.f); 
                    // PrimitiveDrawLine(rectstartpoint, rectstartpoint + drawingvertical * 16.f, vec4(0,0,1,1), 2.f); 
                    vec3 startToEndProjOnToHorizontal = Dot(startToEndVector, drawinghorizontal) * drawinghorizontal;
                    vec3 startToEndProjOnToVertical = Dot(startToEndVector, drawingvertical) * drawingvertical;
                    PrimitiveDrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToVertical,
                                      vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                    PrimitiveDrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToHorizontal,
                                      vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                    PrimitiveDrawLine(endpoint, endpoint - startToEndProjOnToVertical,
                                      vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                    PrimitiveDrawLine(endpoint, endpoint - startToEndProjOnToHorizontal,
                                      vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                }
                if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    vec3 intersection;
                    IntersectPlaneAndLineWithDirections(rectstartpoint, drawingplanenormal, ws, wr, &intersection);
                    rectendpoint = intersection;
                    rectendpoint = SnapToGrid(rectendpoint);
                    vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);
                    vec3 startToEndProjOnToHorizontal = Dot(startToEndVector, drawinghorizontal) * drawinghorizontal;
                    vec3 startToEndProjOnToVertical = Dot(startToEndVector, drawingvertical) * drawingvertical;
                    if (Magnitude(startToEndProjOnToHorizontal)+0.05f >= GRID_INCREMENT && Magnitude(startToEndProjOnToVertical)+0.05f >= GRID_INCREMENT)
                        simpleBrushToolState = SimpleBrushToolState::DrawingHeight;
                    else
                        simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                if (keyspressed[SDL_SCANCODE_ESCAPE])
                    simpleBrushToolState = SimpleBrushToolState::NotActive;

                break;
            case SimpleBrushToolState::DrawingHeight:

                vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);

                float zcomponent = Dot(startToEndVector, drawinghorizontal);
                float xcomponent = Dot(startToEndVector, drawingvertical);
                vec3 startToEndProjOnToHorizontal = zcomponent * drawinghorizontal;
                vec3 startToEndProjOnToVertical = xcomponent * drawingvertical;

                GRID_ORIGIN = rectendpoint;
                GRID_UP_VECTOR = startToEndProjOnToHorizontal;
                GRID_RIGHT_VECTOR = startToEndProjOnToVertical;

                // let plane be defined at point rectendpoint with normal -cameraDirection
                // then the point of intersection between mouse ray and the plane - endrectpoint and project it onto
                // the direction of translation e.g. GM_UP_VECTOR
                vec3 drawingSurfaceNormal = drawingplanenormal;
                static vec3 height = vec3();
                static vec3 heightBeforeSnap = vec3();
                vec3 pn = -cameraDirection;
                vec3 pp = rectendpoint + heightBeforeSnap;
                vec3 wp = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 wr = ScreenPointToWorldRay(mousepos);
                vec3 intersection;
                IntersectPlaneAndLineWithDirections(pp, pn, wp, wr, &intersection);
                float trueHeightComp = Dot((intersection - rectendpoint), drawingSurfaceNormal);
                float heightcomponent = SnapToGrid(trueHeightComp);
                height = heightcomponent * drawingSurfaceNormal;
                heightBeforeSnap = trueHeightComp * drawingSurfaceNormal;

                vec3 floorPointA = rectstartpoint;
                vec3 floorPointB = rectstartpoint + startToEndProjOnToVertical;
                vec3 floorPointC = rectendpoint;
                vec3 floorPointD = rectendpoint - startToEndProjOnToVertical;
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

                PrimitiveDrawLine(floorPointA, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(floorPointA, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(floorPointC, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(floorPointC, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);

                PrimitiveDrawLine(floorPointA, ceilPointA, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(floorPointB, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(floorPointC, ceilPointC, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(floorPointD, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);

                PrimitiveDrawLine(ceilPointA, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(ceilPointA, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(ceilPointC, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                PrimitiveDrawLine(ceilPointC, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);

                if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    // if heigth component is zero or too small then just default to current grid increment

                    MapEdit::Volume createdVolume;
                    createdVolume.persistId = MapEdit::FreshVolumePersistId();

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

                    for (auto face : createdVolume.faces)
                    {
                        MY_VERTEX_BUFFER.clear();
                        TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                        RebindFaceBatchBufferObject(&face->facemesh, (u32) MY_VERTEX_BUFFER.size() * sizeof(float),
                            MY_VERTEX_BUFFER.data());
                        face->texture = Session.SelectedTexture; 
                        face->facemesh.sharedTexture = Session.SelectedTexture.gputex; 
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






