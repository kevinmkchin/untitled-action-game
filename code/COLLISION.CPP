
/// GJK + EPA

struct Collider
{
    /* Returns the furthest point on the collider along a given direction */
    virtual vec3 FindFurthestPoint(vec3 direction) const = 0;
};

struct MeshCollider : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
        vec3 maxVertexSoFar;
        float maxDistanceSoFar = -FLT_MAX;
        for (auto pointWCS : pointCloud)
        {
            float distance = Dot(pointWCS, direction);
            if (distance > maxDistanceSoFar)
            {
                maxVertexSoFar = pointWCS;
                maxDistanceSoFar = distance;
            }
        }
        return maxVertexSoFar;
    }

    std::vector<vec3> pointCloud;
};

struct SphereCollider : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
        vec3 offset = Normalize(direction) * radius;
        return position + offset;
    }

    vec3 position;
    float radius;
};

struct LineCollider : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
        float distanceA = Dot(a, direction);
        float distanceB = Dot(b, direction);
        return (distanceA > distanceB ? a : b);
    }

    vec3 a;
    vec3 b;
};


struct GJKSimplex
{
private:
    std::array<vec3, 4> points = {};
    u8 count = 0;

public:
    GJKSimplex& operator=(std::initializer_list<vec3> list)
    {
        count = 0;
        for (auto v : list)
        {
            points[count] = v;
            ++count;
        }
        return *this;
    }

    vec3& operator[](int i)
    {
        return points[i];
    }

    void pushFront(vec3 point)
    {
        points = { point, points[0], points[1], points[2] };
        count = GM_min(count + 1, 4u);
    }

    u8 size() const
    {
        return count;
    }
};

static vec3 GJKSupport(const Collider *colA, const Collider *colB, const vec3& direction)
{
    // https://en.wikipedia.org/wiki/Support_function
    return colA->FindFurthestPoint(direction) - colB->FindFurthestPoint(-direction);
}

#define GJKSameDirection(dir, ao) (Dot(dir, ao) > 0.f)

static bool GJKLine(GJKSimplex& capture, vec3& direction)
{
    vec3 a = capture[0];
    vec3 b = capture[1];
    vec3 ab = b - a;
    // Note that vec3 ao (a to origin) is just -a

    if (GJKSameDirection(ab, -a))
    {
        // NOTE(Kevin): If initial search direction aligns with the collision vector between two spheres, then it doesn't work.
        direction = Cross(Cross(ab, -a), ab);
    }
    else
    {
        capture = { a };
        direction = -a;
        // THIS PATH WILL SIGNAL THAT THERE IS NO COLLISION, but it might waste a support call
        // Is there a way to stop the algorithm here?
        // if origin is past point A, then it's impossible to be included in the simplex.
        // I need to figure out how to stop the algorithm here and return that they are not intersecting.
    }

    return false;
}

static bool GJKTriangle(GJKSimplex& capture, vec3& direction)
{
    vec3 a = capture[0];
    vec3 b = capture[1];
    vec3 c = capture[2];
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 triangleNormal = Cross(ab, ac);

    if (GJKSameDirection(Cross(triangleNormal, ac), -a))
    {
        if (GJKSameDirection(ac, -a))
        {
            capture = { a, c };
            direction = Cross(Cross(ac, -a), ac);
        }
        else
        {
            capture = { a };
            direction = -a;
        }
    }
    else
    {
        if (GJKSameDirection(Cross(ab, triangleNormal), -a))
        {
            if (GJKSameDirection(ab, -a))
            {
                capture = { a, b };
                direction = Cross(Cross(ab, -a), ab);
            }
            else
            {
                capture = { a };
                direction = -a;
            }
        }
        else
        {
            // The origin must be above or below this triangle
            if (GJKSameDirection(triangleNormal, -a))
            {
                capture = { a, b, c };
                direction = triangleNormal;
            }
            else
            {
                capture = { a, c, b };
                direction = -triangleNormal;
            }
        }
    }

    return false;
}

static bool GJKTetrahedron(GJKSimplex& capture, vec3& direction)
{
    vec3 a = capture[0];
    vec3 b = capture[1];
    vec3 c = capture[2];
    vec3 d = capture[3];
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ad = d - a;
    vec3 abcNormal = Cross(ab, ac);
    vec3 acdNormal = Cross(ac, ad);
    vec3 adbNormal = Cross(ad, ab);

    /*
     * ORDER NEEDS TO BE ABC first, ACD second, ADB third OR IT MIGHT INFINITELY
     * LOOP THE TETRAHEDRON AND TRIANGLE CASES BUT I DON'T UNDERSTAND WHY
     * */

    if (GJKSameDirection(abcNormal, -a))
    {
        capture = { a, b, c };
        return GJKTriangle(capture, direction);
    }
    if (GJKSameDirection(acdNormal, -a))
    {
        capture = { a, c, d };
        return GJKTriangle(capture, direction);
    }
    if (GJKSameDirection(adbNormal, -a))
    {
        capture = { a, d, b };
        return GJKTriangle(capture, direction);
    }

    // IN THIS CASE: The origin is not in the direction of the normals of the three triangles formed with a.
    // AND we know that the origin is not "below" the triangle (if bcd forms the base) because we just added
    // the point a i.e. the origin is in direction of direction given to this function.
    // THEREFORE, the origin MUST be inside the tetrahedron
    return true;
}

static bool GJKNextSimplex(GJKSimplex& capture, vec3& direction)
{
    /* We need a series of different checks for each shape of simplex
     * to see what the new simplex should be and what direction we'll
     * search in next */
    switch (capture.size()) {
    case 2: return GJKLine(capture, direction);
    case 3: return GJKTriangle(capture, direction);
    case 4: return GJKTetrahedron(capture, direction);
    }

    ASSERT(false) // should never happen
        return false;
}

typedef struct GJKEPACollisionInfo
{
    vec3 normal;
    float penetrationDepth = 0.f;
    bool hasCollision = false;
} ColResult;

static std::pair<std::vector<vec4>, int> EPAGetFaceNormals(
    const std::vector<vec3>& polytope,
    const std::vector<int>& faces)
{
    /* Returns the normal of every face in the polytope, their distances
     * from the origin, and the index of the face closest to the origin. */

    std::vector<vec4> normals;
    int minTriangle = 0;
    float minDistance = FLT_MAX;

    for (int i = 0; i < faces.size(); i += 3)
    {
        vec3 a = polytope[faces[i]];
        vec3 b = polytope[faces[(size_t) i + 1]];
        vec3 c = polytope[faces[(size_t) i + 2]];
        vec3 normal = Normalize(Cross((b - a), (c - a)));
        float distance = Dot(normal, a); // the distance is the component of vector a projected onto the normal

        if (distance < 0.f)
        {
            normal *= -1.f;
            distance *= -1.f;
        }

        normals.emplace_back(normal, distance);

        if (distance < minDistance)
        {
            minDistance = distance;
            minTriangle = i / 3;
        }
    }

    return { normals, minTriangle };
}

static void EPAAddIfUniqueEdge(
    std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& faces,
    int a,
    int b)
{
    auto reverse = std::find(edges.begin(), edges.end(),
        std::make_pair(faces[b], faces[a]));
    if (reverse != edges.end())
    {
        edges.erase(reverse);
    }
    else
    {
        edges.emplace_back(faces[a], faces[b]);
    }
}

#ifndef EPA_MAX_ITERATIONS
#define EPA_MAX_ITERATIONS 10
#endif
static GJKEPACollisionInfo EPA(GJKSimplex& simplex, const Collider* colA, const Collider* colB)
{
    std::vector<vec3> polytope;
    ASSERT(simplex.size() == 4)
        for (int i = 0; i < simplex.size(); ++i)
        {
            polytope.push_back(simplex[i]);
        }
    std::vector<int> faces = {
            0, 1, 2,
            0, 3, 1,
            0, 2, 3,
            1, 3, 2
    };

    auto normalsResult = EPAGetFaceNormals(polytope, faces);
    auto normals = normalsResult.first;
    auto minFace = normalsResult.second;

    vec3 minNormal;
    float minDistance = FLT_MAX;

    u8 epaIterations = 0; // the EPA loop can be infinite if we have curved edges - just return whatever it has after X iterations.
    while (minDistance == FLT_MAX)
    {
        minNormal = normals[minFace].xyz;
        minDistance = normals[minFace].w;

        if (epaIterations++ > EPA_MAX_ITERATIONS)
        {
            break;
        }

        vec3 support = GJKSupport(colA, colB, minNormal); // i.e. is there a point to be added in this direction?
        float sDistance = Dot(minNormal, support);

        if (GM_abs(sDistance - minDistance) > 0.001f)
        {
            // then we must expand the polytope
            std::vector<std::pair<int, int>> uniqueEdges;
            for (int i = 0; i < normals.size(); ++i)
            {
                if (GJKSameDirection(normals[i].xyz, support))
                {
                    // for each face, if the face is in the same direction as support, remove
                    int f = i * 3;
                    EPAAddIfUniqueEdge(uniqueEdges, faces, f, f + 1);
                    EPAAddIfUniqueEdge(uniqueEdges, faces, f + 1, f + 2);
                    EPAAddIfUniqueEdge(uniqueEdges, faces, f + 2, f);
                    // reconstruction
                    faces[(size_t) f + 2] = faces.back();
                    faces.pop_back();
                    faces[(size_t) f + 1] = faces.back();
                    faces.pop_back();
                    faces[f] = faces.back();
                    faces.pop_back();
                    normals[i] = normals.back();
                    normals.pop_back();
                    --i;
                }
            }

            if (uniqueEdges.size() == 0)
            {
                minNormal = vec3(0.f, 0.f, 0.f);
                minDistance = 0.f;
                break;
            }

            std::vector<int> newFaces;
            for (auto edge : uniqueEdges)
            {
                newFaces.push_back(edge.first);
                newFaces.push_back(edge.second);
                newFaces.push_back((int)polytope.size());
            }
            polytope.push_back(support);

            normalsResult = EPAGetFaceNormals(polytope, newFaces);
            auto newNormals = normalsResult.first;
            auto newMinFace = normalsResult.second;

            float oldMinDistance = FLT_MAX;
            for (int i = 0; i < normals.size(); ++i)
            {
                if (normals[i].w < oldMinDistance)
                {
                    oldMinDistance = normals[i].w;
                    minFace = i;
                }
            }

            if (newNormals[newMinFace].w < oldMinDistance)
            {
                minFace = newMinFace + (int)normals.size();
            }

            faces.insert(faces.end(), newFaces.begin(), newFaces.end());
            normals.insert(normals.end(), newNormals.begin(), newNormals.end());

            minDistance = FLT_MAX;
        }
    }

    GJKEPACollisionInfo points;
    points.normal = minNormal;
    points.penetrationDepth = minDistance + 0.001f;
    points.hasCollision = true;
    return points;
}

// static GJKEPACollisionInfo CheckSphereCollision(SphereCollider* a, SphereCollider* b)
// {
//     vec3 aPos = a->GetPosition();
//     vec3 bPos = b->GetPosition();
//     float aRad = a->GetScaledRadius();
//     float bRad = b->GetScaledRadius();
//     float baLen = Magnitude(bPos - aPos);
//     if (baLen < aRad + bRad)
//     {
//         return { Normalize(bPos - aPos), aRad + bRad - baLen, true };
//     }
//     else
//     {
//         return { vec3(), 0.f, false };
//     }
// }

static GJKEPACollisionInfo GJK(Collider* colA, Collider* colB)
{
    // colA->TransformCollider();
    // colB->TransformCollider();

    // if (colA->colliderType == ColliderType::Sphere && colB->colliderType == ColliderType::Sphere)
    // {
    //     return CheckSphereCollision((SphereCollider*)colA, (SphereCollider*)colB);
    // }

    //NOTE(Kevin): start direction could be origin - (colA owning object position - colB owning object position)
    vec3 support = GJKSupport(colA, colB, { 1.f,1.f,1.f });
    GJKSimplex capture;
    capture.pushFront(support);
    vec3 direction = -support;

    for (;;)
    {
        support = GJKSupport(colA, colB, direction);
        if (Dot(support, direction) <= 0)
        {
            // previous support point was already the furthest along direction
            GJKEPACollisionInfo collisionInfo;
            collisionInfo.hasCollision = false;
            return collisionInfo;
        }

        capture.pushFront(support);
        if (GJKNextSimplex(capture, direction))
        {
            GJKEPACollisionInfo collisionInfo = EPA(capture, colA, colB);
            collisionInfo.hasCollision = true;
            return collisionInfo;
        }
    }
}



/// Acceleration with Octree

struct Bounds : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
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

        vec3 maxVertexSoFar;
        float maxDistanceSoFar = -FLT_MAX;
        for (int i = 0; i < 8; ++i)
        {
            vec3 pointWCS = corners[i];
            float distance = Dot(pointWCS, direction);
            if (distance > maxDistanceSoFar)
            {
                maxVertexSoFar = pointWCS;
                maxDistanceSoFar = distance;
            }
        }
        return maxVertexSoFar;
    }

    vec3 center;
    vec3 size;

    bool Intersects(const Bounds& other) const 
    {
        return (GM_abs(center.x - other.center.x) * 2.f < (size.x + other.size.x)) &&
               (GM_abs(center.y - other.center.y) * 2.f < (size.y + other.size.y)) &&
               (GM_abs(center.z - other.center.z) * 2.f < (size.z + other.size.z));
    }

    bool Intersects(const LineCollider& line) // Vector3 A, Vector3 B, Vector3 boxMin, Vector3 boxMax) 
    {
        vec3 A = line.a;
        vec3 B = line.b;
        vec3 boxMin = center - size/2.f;
        vec3 boxMax = center + size/2.f;

        vec3 dir = B - A;
        float tmin = (boxMin.x - A.x) / dir.x;
        float tmax = (boxMax.x - A.x) / dir.x;

        if (tmin > tmax) std::swap(tmin, tmax);

        float tymin = (boxMin.y - A.y) / dir.y;
        float tymax = (boxMax.y - A.y) / dir.y;

        if (tymin > tymax) std::swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (boxMin.z - A.z) / dir.z;
        float tzmax = (boxMax.z - A.z) / dir.z;

        if (tzmin > tzmax) std::swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        if (tzmin > tmin)
            tmin = tzmin;
        if (tzmax < tmax)
            tmax = tzmax;

        // Check if intersection occurs within the segment [0, 1]
        return (tmin <= tmax) && (tmin >= 0) && (tmin <= 1);
    }

    // bool Contains(const Bounds& other) const 
    // {
    //     return (other.center.x - other.size.x / 2.f >= center.x - size.x / 2.f) &&
    //            (other.center.x + other.size.x / 2.f <= center.x + size.x / 2.f) &&
    //            (other.center.y - other.size.y / 2.f >= center.y - size.y / 2.f) &&
    //            (other.center.y + other.size.y / 2.f <= center.y + size.y / 2.f) &&
    //            (other.center.z - other.size.z / 2.f >= center.z - size.z / 2.f) &&
    //            (other.center.z + other.size.z / 2.f <= center.z + size.z / 2.f);
    // }
};

Bounds BoundsFromMeshCollider(MeshCollider& meshCollider)
{
    vec3 min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (vec3 point : meshCollider.pointCloud) 
    {
        min.x = GM_min(min.x, point.x);
        min.y = GM_min(min.y, point.y);
        min.z = GM_min(min.z, point.z);
        
        max.x = GM_max(max.x, point.x);
        max.y = GM_max(max.y, point.y);
        max.z = GM_max(max.z, point.z);
    }

    Bounds bounds;
    bounds.size = max - min;
    bounds.center = min + (bounds.size/2.f);
    return bounds;
}

struct OctreeNode
{
    Bounds bounds;
    std::vector<Collider*> objects;
    std::unique_ptr<OctreeNode[]> children;
    bool isLeaf = true;

    OctreeNode()
    {}

    OctreeNode(const Bounds& bounds)
        : bounds(bounds)
    {}
};

struct Octree
{
    Octree() {}
    Octree(const Bounds& worldBounds, int maxDepth, int maxObjectsPerNode)
        : root(new OctreeNode(worldBounds))
        , maxDepth(maxDepth)
        , maxObjectsPerNode(maxObjectsPerNode) 
    {}

    // Places Collider into octree
    void Insert(Collider *obj) 
    {
        Insert(root.get(), obj, 0);
    }

    // Retrieve objects that may intersect with given bounding box
    std::unordered_set<Collider*> Query(const Bounds& queryBounds) 
    {
        return QueryNode(root.get(), queryBounds);
    }

    // Retrieve objects that may intersect with given line
    std::unordered_set<Collider*> Query(const LineCollider& queryRay) 
    {
        return QueryNode(root.get(), queryRay);
    }

    std::unique_ptr<OctreeNode> root;
    int maxDepth;
    int maxObjectsPerNode; // TODO (Kevin): Compare performance of large scene with this at 100000 and this at something reasonable like 32

    void Insert(OctreeNode *node, Collider *obj, int depth);
    u8 GetChildIndices(const OctreeNode *node, Collider *obj) ;
    void Subdivide(OctreeNode *node);
    void RedistributeObjects(OctreeNode *node);
    std::unordered_set<Collider*> QueryNode(OctreeNode *node, const Bounds& queryBounds);
    std::unordered_set<Collider*> QueryNode(OctreeNode *node, const LineCollider& queryRay);
};

// To handle large colliders that can span multiple octants, I'm resorting to storing them in every leaf node that
// contains them. Alternative approach is to store it in the deepest node that contains the collider fully (i.e. none
// of my sibling octants contain this collider but more than 1 of my children octants contain the collider).

// Objects are inserted into the appropriate nodes, and if a node exceeds the maximum
// object count, it is subdivided into eight smaller nodes.
void Octree::Insert(OctreeNode *node, Collider *obj, int depth) 
{
    if (node->isLeaf) 
    {
        node->objects.push_back(obj);

        if (node->objects.size() > maxObjectsPerNode && depth < maxDepth) 
        {
            // does not guarantee resolve max obj but will run again if node on next insert if node is still problematic
            Subdivide(node);
            RedistributeObjects(node);
        }
    } 
    else 
    {
        // can be inserted into multiple children

        u8 indices = GetChildIndices(node, obj);
        if (indices & 0b00000001)
            Insert(&node->children[0], obj, depth + 1);
        if (indices & 0b00000010)
            Insert(&node->children[1], obj, depth + 1);
        if (indices & 0b00000100)
            Insert(&node->children[2], obj, depth + 1);
        if (indices & 0b00001000)
            Insert(&node->children[3], obj, depth + 1);
        if (indices & 0b00010000)
            Insert(&node->children[4], obj, depth + 1);
        if (indices & 0b00100000)
            Insert(&node->children[5], obj, depth + 1);
        if (indices & 0b01000000)
            Insert(&node->children[6], obj, depth + 1);
        if (indices & 0b10000000)
            Insert(&node->children[7], obj, depth + 1);
    }
}

// When a node exceeds its object capacity, it splits into eight smaller regions, 
// each representing a different octant. 
void Octree::Subdivide(OctreeNode *node) 
{
    vec3 octantSize = node->bounds.size / 2.0f;

    node->children = std::make_unique<OctreeNode[]>(8);
    node->isLeaf = false;

    for (int i = 0; i < 8; ++i) 
    {
        vec3 newCenter = node->bounds.center;
        newCenter.x += octantSize.x * ((i & 1) ? 0.5f : -0.5f);
        newCenter.y += octantSize.y * ((i & 2) ? 0.5f : -0.5f);
        newCenter.z += octantSize.z * ((i & 4) ? 0.5f : -0.5f);

        Bounds octant;
        octant.center = newCenter;
        octant.size = octantSize;
        node->children[i] = OctreeNode(octant);
    }
}

void Octree::RedistributeObjects(OctreeNode *node) 
{
    std::vector<Collider*> objectsToReinsert = std::move(node->objects);
    node->objects.clear();

    for (Collider *obj : objectsToReinsert) 
    {
        u8 indices = GetChildIndices(node, obj);
        if (indices & 0b00000001)
            node->children[0].objects.push_back(obj);
        if (indices & 0b00000010)
            node->children[1].objects.push_back(obj);
        if (indices & 0b00000100)
            node->children[2].objects.push_back(obj);
        if (indices & 0b00001000)
            node->children[3].objects.push_back(obj);
        if (indices & 0b00010000)
            node->children[4].objects.push_back(obj);
        if (indices & 0b00100000)
            node->children[5].objects.push_back(obj);
        if (indices & 0b01000000)
            node->children[6].objects.push_back(obj);
        if (indices & 0b10000000)
            node->children[7].objects.push_back(obj);
    }
}

// given a collider and a node, which of its octants does it belong to? up to all of them
u8 Octree::GetChildIndices(const OctreeNode *node, Collider *obj) 
{
    Collider *octant0 = &node->children[0].bounds;
    Collider *octant1 = &node->children[1].bounds;
    Collider *octant2 = &node->children[2].bounds;
    Collider *octant3 = &node->children[3].bounds;
    Collider *octant4 = &node->children[4].bounds;
    Collider *octant5 = &node->children[5].bounds;
    Collider *octant6 = &node->children[6].bounds;
    Collider *octant7 = &node->children[7].bounds;

    // Note (Kevin): GJK returns true when a collider is lying on the surface of an octant bound
    ColResult result0 = GJK(octant0, obj);
    ColResult result1 = GJK(octant1, obj);
    ColResult result2 = GJK(octant2, obj);
    ColResult result3 = GJK(octant3, obj);
    ColResult result4 = GJK(octant4, obj);
    ColResult result5 = GJK(octant5, obj);
    ColResult result6 = GJK(octant6, obj);
    ColResult result7 = GJK(octant7, obj);

    u8 childIndices = 0;
    childIndices |= result0.hasCollision ? 0b00000001 : 0;
    childIndices |= result1.hasCollision ? 0b00000010 : 0;
    childIndices |= result2.hasCollision ? 0b00000100 : 0;
    childIndices |= result3.hasCollision ? 0b00001000 : 0;
    childIndices |= result4.hasCollision ? 0b00010000 : 0;
    childIndices |= result5.hasCollision ? 0b00100000 : 0;
    childIndices |= result6.hasCollision ? 0b01000000 : 0;
    childIndices |= result7.hasCollision ? 0b10000000 : 0;

    return childIndices;
}

// Use AABB of query collider to speed up look up
// NOTE(Kevin): these insertions into unordered_set may take up a lot of time. If so, probably better to do approach where there are no duplicates (large colliders stored in parent)
std::unordered_set<Collider*> Octree::QueryNode(OctreeNode *node, const Bounds& queryBounds) 
{
    std::unordered_set<Collider*> result;

    if (!node->bounds.Intersects(queryBounds)) 
        return result;

    if (node->isLeaf) 
    {
        for (Collider *obj : node->objects) 
        {
            result.insert(obj);
        }
    } 
    else
    {
        for (int i = 0; i < 8; ++i)
        {
            std::unordered_set<Collider*> childResults = QueryNode(&node->children[i], queryBounds);
            result.insert(childResults.begin(), childResults.end());
        }
    }

    return result;
}

// I need a QueryNode override for ray checking
std::unordered_set<Collider*> Octree::QueryNode(OctreeNode *node, const LineCollider& queryRay) 
{
    std::unordered_set<Collider*> result;

    if (!node->bounds.Intersects(queryRay)) 
        return result;

    if (node->isLeaf) 
    {
        for (Collider *obj : node->objects) 
        {
            result.insert(obj);
        }
    } 
    else
    {
        for (int i = 0; i < 8; ++i)
        {
            std::unordered_set<Collider*> childResults = QueryNode(&node->children[i], queryRay);
            result.insert(childResults.begin(), childResults.end());
        }
    }

    return result;
}
