
/// GJK + EPA

enum class ColliderType
{
    MESH,
    POLYGON,
    SPHERE,
    LINE,
    BOUNDS,
    INVALID,
};

struct Collider
{
    /* Returns the furthest point on the collider along a given direction */
    virtual vec3 FindFurthestPoint(vec3 direction) const = 0;

    ColliderType type = ColliderType::INVALID;

#if INTERNAL_BUILD
    int debugId = -1;
#endif
};

struct MeshCollider : Collider // kinda wanna remove this... I don't want to use std::vector for pointCloud
{
    MeshCollider()
    {
        type = ColliderType::MESH;
    }

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

struct StaticMeshCollider : Collider
{
    StaticMeshCollider()
    {
        type = ColliderType::POLYGON;
    }

    vec3 FindFurthestPoint(vec3 direction) const override
    {
        vec3 maxVertexSoFar;
        float maxDistanceSoFar = -FLT_MAX;
#if INTERNAL_BUILD
        if (pointCount == 0) LogWarning("StaticMeshCollider has zero points in it...");
#endif
        for (u32 i = 0; i < pointCount; ++i)
        {
            vec3 pointWCS = pointCloudPtr[i];
            float distance = Dot(pointWCS, direction);
            if (distance > maxDistanceSoFar)
            {
                maxVertexSoFar = pointWCS;
                maxDistanceSoFar = distance;
            }
        }
        return maxVertexSoFar;
    }

    // using pointer into big static array of collider points
    vec3 *pointCloudPtr = NULL;
    u32 pointCount = 0;
};

struct FlatPolygonCollider : Collider
{
    FlatPolygonCollider()
    {
        type = ColliderType::POLYGON;
    }

    vec3 FindFurthestPoint(vec3 direction) const override
    {
        vec3 maxVertexSoFar;
        float maxDistanceSoFar = -FLT_MAX;
#if INTERNAL_BUILD
        if (pointCount == 0) LogWarning("FlatPolygonCollider has zero points in it...");
#endif
        for (u32 i = 0; i < pointCount; ++i)
        {
            vec3 pointWCS = pointCloudPtr[i];
            float distance = Dot(pointWCS, direction);
            if (distance > maxDistanceSoFar)
            {
                maxVertexSoFar = pointWCS;
                maxDistanceSoFar = distance;
            }
        }
        return maxVertexSoFar;
    }

    // using pointer into big static array of collider points
    vec3 *pointCloudPtr = NULL;
    u32 pointCount = 0;
};

struct SphereCollider : Collider
{
    SphereCollider()
    {
        type = ColliderType::SPHERE;
    }

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
    LineCollider()
    {
        type = ColliderType::LINE;
    }

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

struct CollisionResult
{
    vec3 normal;
    float penetrationDepth = 0.f;
    bool hasCollision = false;
};

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
static CollisionResult EPA(GJKSimplex& simplex, const Collider* colA, const Collider* colB)
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

    CollisionResult points;
    points.normal = minNormal;
    points.penetrationDepth = minDistance + 0.001f;
    points.hasCollision = true;
    return points;
}

// static CollisionResult CheckSphereCollision(SphereCollider* a, SphereCollider* b)
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

static CollisionResult GJK(Collider* colA, Collider* colB)
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
            CollisionResult collisionInfo;
            collisionInfo.hasCollision = false;
            return collisionInfo;
        }

        capture.pushFront(support);
        if (GJKNextSimplex(capture, direction))
        {
            CollisionResult collisionInfo = EPA(capture, colA, colB);
            collisionInfo.hasCollision = true;
            return collisionInfo;
        }
    }
}

// UGH I hate how I'm trying to dispatch based on different collider types
// fuck this object oriented shit. I know exactly what needs colliding against what,
// so just code THAT.


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
    static NiceArray<vec2, 32> projectedVertices;
    projectedVertices.ResetCount();

    vec3 basisU = edge0;
    vec3 basisV = Normalize(Cross(normal, basisU));

    for (u32 i = 0; i < polygon->pointCount; ++i)
    {
        vec3 p = polygon->pointCloudPtr[i];
        float u = Dot(p-pv0, basisU);
        float v = Dot(p-pv0, basisV);
        projectedVertices.PushBack(vec2(u, v));
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
        vec2 edgeA = projectedVertices.At(i);
        vec2 edgeB = projectedVertices.At(i != projectedVertices.count-1 ? i+1 : 0);

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

#if INTERNAL_BUILD

void TestAssert(const char *desc, bool condition)
{
    if (condition)
        LogMessage("TEST PASSED - '%s'", desc);
    else
        LogError("TEST FAILED - '%s'", desc);
}

#define TEST_INITLIST(...) __VA_ARGS__

#define TEST_COLLIDEFLATPOLYGONXLINE(desc, pointcloud, pointcount, linea, lineb, expectedHasCollision) \
{\
    vec3 cloud[] = pointcloud;\
    FlatPolygonCollider polygon;\
    polygon.pointCloudPtr = cloud;\
    polygon.pointCount = pointcount;\
    LineCollider line;\
    line.a = linea;\
    line.b = lineb;\
    CollisionResult result = CollideFlatPolygonXLine(&polygon, &line);\
    TestAssert(desc, result.hasCollision == expectedHasCollision);\
}

void Tests_CollideFlatPolygonXLine()
{
    LogMessage("Running Tests_CollideFlatPolygonXLine");

    TEST_COLLIDEFLATPOLYGONXLINE("convex polygon with intersection 1",
        TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, 0.0f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
        TEST_INITLIST({0.5f, 0.5f, -1.0f}), TEST_INITLIST({0.5f, 0.5f, 3.0f}), 
        true);

    TEST_COLLIDEFLATPOLYGONXLINE("line does not intersect polygon 1",
        TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, 0.0f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
        TEST_INITLIST({0.7f, 0.0f, -1.0f}), TEST_INITLIST({0.7f, -1.0f, 3.0f}), 
        false);

    TEST_COLLIDEFLATPOLYGONXLINE("line does not intersect polygon 2",
        TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, 0.0f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
        TEST_INITLIST({0.7f, 0.0f, -1.0f}), TEST_INITLIST({0.7f, 3.6f, 2.0f}), 
        false);

    TEST_COLLIDEFLATPOLYGONXLINE("line intersects edge of polygon",
        TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, 0.0f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
        TEST_INITLIST({0.7f, 0.0f, -1.0f}), TEST_INITLIST({0.7f, 0.0f, 3.0f}), 
        true);
    
    TEST_COLLIDEFLATPOLYGONXLINE("line intersects edge of polygon",
        TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, 0.0f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
        TEST_INITLIST({ 0.0f, 1.0f, -1.0f }), TEST_INITLIST({ 0.0f, 1.0f, 3.0f }),
        true);

    TEST_COLLIDEFLATPOLYGONXLINE("line intersects edge of polygon",
        TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, 0.0f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
        TEST_INITLIST({ 0.9f, 2.0f, -1.0f }), TEST_INITLIST({ 0.9f, 2.0f, 3.0f }),
        true);

    TEST_COLLIDEFLATPOLYGONXLINE("line intersects corner of polygon",
       TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, -0.4f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
       TEST_INITLIST({1.5f, -0.4f, -1.0f}), TEST_INITLIST({1.5f, -0.4f,  3.0f}),
       true);

    TEST_COLLIDEFLATPOLYGONXLINE("line intersects corner of polygon",
       TEST_INITLIST({ {0.0f, 0.0f, 1.0f}, {1.5f, -0.4f, 1.0f}, {2.0f, 2.0f, 1.0f}, {0.0f, 2.0f, 1.0f} }), 4,
       TEST_INITLIST({ 0.0f, 0.0f, -1.0f }), TEST_INITLIST({ 0.0f, 0.0f,  3.0f }),
       true);
}

#endif

// typedef CollisionResult (*CollisionTestFunc)(Collider*,Collider*);

// CollisionTestFunc CollisionCheckDispatchRules[ColliderType::INVALID][ColliderType::INVALID] = {NULL};

// void SetupColliderTypeDispatchRules()
// {
//     CollisionCheckDispatchRules[(u8)ColliderType::MESH][(u8)ColliderType::MESH] = GJK_;
//     CollisionCheckDispatchRules[(u8)ColliderType::POLYGON][(u8)ColliderType::LINE] = CollideFlatPolygonXLine;
//     CollisionCheckDispatchRules[(u8)ColliderType::LINE][(u8)ColliderType::POLYGON] = CollideFlatPolygonXLine;
//     CollisionCheckDispatchRules[(u8)ColliderType::MESH][(u8)ColliderType::POLYGON] = GJK_;
//     CollisionCheckDispatchRules[(u8)ColliderType::POLYGON][(u8)ColliderType::MESH] = GJK_;
// }

// CollisionResult CheckCollision(Collider *A, Collider *B)
// {
//     CollisionTestFunc checkFunc = CollisionCheckDispatchRules[(u8)A->type][(u8)B->type];
//     ASSERT(checkFunc != NULL);
//     return checkFunc(A, B);
// }


/// Acceleration with Octree

struct Bounds : Collider
{
    vec3 center;
    vec3 size;

    Bounds()
    {
        type = ColliderType::BOUNDS;
    };
    Bounds(vec3 center, vec3 size)
        : center(center)
        , size(size)
    {
        type = ColliderType::BOUNDS;
    };

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

    bool IntersectsOrContains(const LineCollider& line)
    {
        return Contains(line.a) || Contains(line.b);
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

Bounds BoundsFromFlatPolygonCollider(FlatPolygonCollider& collider)
{
    vec3 min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (u32 i = 0; i < collider.pointCount; ++i) 
    {
        vec3 point = collider.pointCloudPtr[i];

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
    int maxObjectsPerNode; // TODO(Kevin): Fun thing to do - compare performance of large scene with this at 100000 and this at something reasonable like 32

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
    CollisionResult result0 = GJK(octant0, obj);
    CollisionResult result1 = GJK(octant1, obj);
    CollisionResult result2 = GJK(octant2, obj);
    CollisionResult result3 = GJK(octant3, obj);
    CollisionResult result4 = GJK(octant4, obj);
    CollisionResult result5 = GJK(octant5, obj);
    CollisionResult result6 = GJK(octant6, obj);
    CollisionResult result7 = GJK(octant7, obj);

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

    if (!node->bounds.IntersectsOrContains(queryRay))
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

// struct LevelPolygonOctreeNode
// {
//     Bounds bounds;
//     std::vector<Collider*> objects;
//     std::unique_ptr<LevelPolygonOctreeNode[]> children;
//     bool isLeaf = true;

//     LevelPolygonOctreeNode()
//     {}

//     LevelPolygonOctreeNode(const Bounds& bounds)
//         : bounds(bounds)
//     {}
// };

// // octree for flat polygons in the level
// struct LevelPolygonOctree
// {
//     LevelPolygonOctree() {}
//     LevelPolygonOctree(const Bounds& worldBounds, int maxDepth, int maxObjectsPerNode)
//         : root(new LevelPolygonOctreeNode(worldBounds))
//         , maxDepth(maxDepth)
//         , maxObjectsPerNode(maxObjectsPerNode) 
//     {}

//     void Insert(FlatPolygonCollider *obj) 
//     {
//         Insert(root.get(), obj, 0);
//     }

//     CollisionResult Query(const LineCollider& queryRay) 
//     {
//         return QueryNode(root.get(), queryRay);
//     }

//     std::unique_ptr<LevelPolygonOctreeNode> root;
//     int maxDepth;
//     int maxObjectsPerNode;

//     void Insert(LevelPolygonOctreeNode *node, FlatPolygonCollider *obj, int depth);
//     u8 GetChildIndices(const LevelPolygonOctreeNode *node, FlatPolygonCollider *obj) ;
//     void Subdivide(LevelPolygonOctreeNode *node);
//     void RedistributeObjects(LevelPolygonOctreeNode *node);
//     CollisionResult QueryNode(LevelPolygonOctreeNode *node, const LineCollider& queryRay);
// };

