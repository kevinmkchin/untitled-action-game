
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




