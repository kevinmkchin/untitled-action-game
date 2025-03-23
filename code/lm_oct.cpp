


struct FlatPolygonCollider
{
    // using pointer into big static array of collider points
    vec3 *pointCloudPtr = NULL;
    u32 pointCount = 0;
};

struct LineCollider
{
    vec3 a;
    vec3 b;
};

struct CollisionResult
{
    vec3 normal;
    float penetrationDepth = 0.f;
    bool hasCollision = false;
};

// assume polygon is flat 2D polygon defined in 3D space
// assume line is going from A to B, and so resolution normal is from B to A, 
// and resolution depth is Magnitude(intersection point - B)
// TODO(Kevin): flag for ignore collision when either a or b of line lies on the polygon
CollisionResult CollideFlatPolygonXLine(FlatPolygonCollider *polygon, LineCollider *line)
{
    ASSERT(polygon->pointCount > 2);

    vec3 pv0 = polygon->pointCloudPtr[0];
    vec3 pv1 = polygon->pointCloudPtr[1];
    vec3 pv2 = polygon->pointCloudPtr[2];
    vec3 edge0 = Normalize(pv1 - pv0);
    vec3 edge1 = Normalize(pv2 - pv0);
    for (int i = 3; edge0 == edge1; ++i)
    {
        // Make sure the three points are not colinear
        pv2 = polygon->pointCloudPtr[i]; 
        edge1 = Normalize(pv2 - pv0);
    }
    vec3 normal = Normalize(Cross(edge0, edge1));

    ASSERT(GM_abs(Dot(normal, edge0)) <= 0.0001f);

    // check if finite line intersects the plane on which the polygon lies
    float t = (Dot(normal, pv1) - Dot(normal, line->a)) / Dot(normal, line->b - line->a);
    if (t < 0.f || t > 1.f)
    {
        return CollisionResult(); // no collision
    }

    // there may be a collision
    vec3 IntersectionPoint = line->a + t * (line->b - line->a);

    // project polygon and intersection point into 2D space
    c_array<vec2, 32> projectedVertices; // TODO(Kevin): more than 32 verts
    projectedVertices.reset_count();

    vec3 basisU = edge0;
    vec3 basisV = Normalize(Cross(normal, basisU));

    for (u32 i = 0; i < polygon->pointCount; ++i)
    {
        vec3 p = polygon->pointCloudPtr[i];
        float u = Dot(p-pv0, basisU);
        float v = Dot(p-pv0, basisV);
        projectedVertices.put(vec2(u, v));
    }

    vec2 pip; //projectedIntersectionPoint
    pip.x = Dot(IntersectionPoint-pv0, basisU);
    pip.y = Dot(IntersectionPoint-pv0, basisV);

    // check if line intersection point with that plane lies within the polygon
    // https://en.wikipedia.org/wiki/Point_in_polygon
    // I'm gonna do simple raycasting (even-odd rule) check. It will be good enough
    // until I ever want to add collision checks against polygons with holes.
    u32 crossings = 0;
    for (int i = 0; i < projectedVertices.count; ++i)
    {
        vec2 edgeA = projectedVertices[i];
        vec2 edgeB = projectedVertices[i != projectedVertices.count-1 ? i+1 : 0];

        if (edgeA == pip) // intersection point is a corner/vertex
        {
            crossings = 1;
            break;
        }

        if (edgeA.y == edgeB.y && edgeA.y == pip.y)
        {
            std::swap(edgeA.x, edgeA.y);
            std::swap(edgeB.x, edgeB.y);
            std::swap(pip.x, pip.y);
        }

        bool withiny = edgeA.y >= pip.y != edgeB.y > pip.y;
        if (withiny)
        {
            float crossingx = (edgeB.x - edgeA.x) * (pip.y - edgeA.y) / (edgeB.y - edgeA.y) + edgeA.x;

            if (pip.x == crossingx) // intersection point is along an edge
            {
                crossings = 1;
                break;
            }

            if (pip.x < crossingx)
            {
                ++crossings;
            }
            else
            {
                continue;
            }
        }
    }
    if (crossings & 1)
    {
        // then intersection point is inside the polygon (collision)
        CollisionResult result;
        result.hasCollision = true;
        result.normal = Normalize(line->a - line->b);
        result.penetrationDepth = Magnitude(IntersectionPoint - line->b);
        return result;
    }
    else
    {
        // intersection poiont is outside the polygon (no collision)
        return CollisionResult();
    }
}


/// Acceleration with Octree

struct Bounds
{
    vec3 center;
    vec3 size;

    Bounds()
    {
    };
    Bounds(vec3 center, vec3 size)
        : center(center)
        , size(size)
    {
    };

    bool Contains(const vec3 &p) const
    {
        vec3 boxMin = center - size / 2.f;
        vec3 boxMax = center + size / 2.f;

        return 
            boxMin.x <= p.x && p.x <= boxMax.x &&
            boxMin.y <= p.y && p.y <= boxMax.y &&
            boxMin.z <= p.z && p.z <= boxMax.z;
    }

    bool Intersects(const Bounds& other) const 
    {
        return (GM_abs(center.x - other.center.x) * 2.f < (size.x + other.size.x)) &&
               (GM_abs(center.y - other.center.y) * 2.f < (size.y + other.size.y)) &&
               (GM_abs(center.z - other.center.z) * 2.f < (size.z + other.size.z));
    }

    bool Intersects(const FlatPolygonCollider& other) const 
    {
        for (u32 i = 0; i < other.pointCount; ++i)
        {
            if (Contains(other.pointCloudPtr[i]))
                return true;
        }
        return false;
    }

    // returns true if AABB intersects or wholly contains the line
    bool LineIntersectsAABB(const vec3 &A, const vec3 &B) 
    {
        vec3 dir = B - A;
        vec3 invDir;

        float tMin = 0.0f;
        float tMax = 1.0f;

        vec3 boxMin = center - size / 2.f;
        vec3 boxMax = center + size / 2.f;

        for (int i = 0; i < 3; i++) {
            if (dir[i] == 0.0f) {
                // Line is parallel to the axis, check if the point A is within the bounds on this axis
                if (A[i] < boxMin[i] || A[i] > boxMax[i]) {
                    return false;  // Line lies outside the bounds, no intersection
                }
            }
            else {
                // Line is not parallel, calculate intersection using the slab method
                invDir[i] = 1.0f / dir[i];
                float t1 = (boxMin[i] - A[i]) * invDir[i];
                float t2 = (boxMax[i] - A[i]) * invDir[i];

                if (invDir[i] < 0.0f) std::swap(t1, t2);

                // Update the intersection interval
                tMin = GM_max(tMin, t1);
                tMax = GM_min(tMax, t2);

                // If tMin exceeds tMax, the line segment does not intersect the AABB
                if (tMin > tMax) return false;
            }
        }

        return true;  // The line segment intersects the AABB
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


struct LevelPolygonOctreeNode
{
    Bounds bounds;
    std::vector<FlatPolygonCollider*> objects;
    std::unique_ptr<LevelPolygonOctreeNode[]> children;
    bool isLeaf = true;

    LevelPolygonOctreeNode()
    {}

    LevelPolygonOctreeNode(const Bounds& bounds)
        : bounds(bounds)
    {}
};

// octree for flat polygons in the level
struct LevelPolygonOctree
{
    LevelPolygonOctree() {}
    LevelPolygonOctree(const Bounds& worldBounds, int maxDepth, int maxObjectsPerNode)
        : root(new LevelPolygonOctreeNode(worldBounds))
        , maxDepth(maxDepth)
        , maxObjectsPerNode(maxObjectsPerNode) 
    {}

    void TearDown();

    void Insert(FlatPolygonCollider *obj) 
    {
        Insert(root.get(), obj, 0);
    }

    bool Query(LineCollider& queryRay) 
    {
        return QueryNode(root.get(), queryRay);
    }

    std::unique_ptr<LevelPolygonOctreeNode> root;
    int maxDepth = -1;
    int maxObjectsPerNode = -1;

    void Insert(LevelPolygonOctreeNode *node, FlatPolygonCollider *obj, int depth);
    u8 GetChildIndices(const LevelPolygonOctreeNode *node, FlatPolygonCollider *obj) ;
    void Subdivide(LevelPolygonOctreeNode *node);
    void RedistributeObjects(LevelPolygonOctreeNode *node);
    bool QueryNode(LevelPolygonOctreeNode *node, LineCollider& queryRay);
};

void LevelPolygonOctree::TearDown()
{
    // todo
}

void LevelPolygonOctree::Insert(LevelPolygonOctreeNode *node, FlatPolygonCollider *obj, int depth) 
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

        if (
            indices != 0b00000001 &&
            indices != 0b00000010 &&
            indices != 0b00000100 &&
            indices != 0b00001000 &&
            indices != 0b00010000 &&
            indices != 0b00100000 &&
            indices != 0b01000000 &&
            indices != 0b10000000
            )
        {
            node->objects.push_back(obj);
            return;
        }

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

void LevelPolygonOctree::Subdivide(LevelPolygonOctreeNode *node)
{
    vec3 octantSize = node->bounds.size / 2.0f;

    node->children = std::make_unique<LevelPolygonOctreeNode[]>(8);
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
        node->children[i] = LevelPolygonOctreeNode(octant);
    }
}

void LevelPolygonOctree::RedistributeObjects(LevelPolygonOctreeNode *node) 
{
    std::vector<FlatPolygonCollider*> objectsToReinsert = std::move(node->objects);
    node->objects.clear();

    for (FlatPolygonCollider *obj : objectsToReinsert) 
    {
        u8 indices = GetChildIndices(node, obj);

        // keep here if doesn't fit in a single child
        if (
            indices != 0b00000001 &&
            indices != 0b00000010 &&
            indices != 0b00000100 &&
            indices != 0b00001000 &&
            indices != 0b00010000 &&
            indices != 0b00100000 &&
            indices != 0b01000000 &&
            indices != 0b10000000
            )
        {
            node->objects.push_back(obj);
            continue;
        }

        ASSERT(indices != 0);
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
u8 LevelPolygonOctree::GetChildIndices(const LevelPolygonOctreeNode *node, FlatPolygonCollider *obj) 
{
    u8 childIndices = 
        node->children[0].bounds.Intersects(*obj) << 0 |
        node->children[1].bounds.Intersects(*obj) << 1 |
        node->children[2].bounds.Intersects(*obj) << 2 |
        node->children[3].bounds.Intersects(*obj) << 3 |
        node->children[4].bounds.Intersects(*obj) << 4 |
        node->children[5].bounds.Intersects(*obj) << 5 |
        node->children[6].bounds.Intersects(*obj) << 6 |
        node->children[7].bounds.Intersects(*obj) << 7;

    return childIndices;
}

bool LevelPolygonOctree::QueryNode(LevelPolygonOctreeNode *node, LineCollider& queryRay) 
{
    if (!node->bounds.LineIntersectsAABB(queryRay.a, queryRay.b))
        return false;

    for (FlatPolygonCollider *polygon : node->objects)
    {
        CollisionResult result = CollideFlatPolygonXLine(polygon, &queryRay);
        if (result.hasCollision)
            return true;
    }

    if (!node->isLeaf)
    {
        for (int i = 0; i < 8; ++i)
        {
            bool collided = QueryNode(&node->children[i], queryRay);
            if (collided) 
                return true;
        }
    }

    return false;
}
