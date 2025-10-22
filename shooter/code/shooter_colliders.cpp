#include "shooter_colliders.h"
#include "shooter_shared.h"

/// GJK

internal vec3 FindFurthestPointOnConvexShell(collider_base *Collider, vec3 Direction)
{
    vec3 FurthestPoint = {};

    switch (Collider->Type)
    {
        case collider_type::mesh:
        {
            mesh_collider *Mesh = (mesh_collider *)Collider;

            Assert(Mesh->PointCloud);
            Assert(Mesh->PointCount > 0);

            float MaxDistanceSoFar = -FLT_MAX;
            for (uint32 i = 0; i < Mesh->PointCount; ++i)
            {
                vec3 WorldPoint = Mesh->PointCloud[i];
                float Dist = Dot(WorldPoint, Direction);
                if (Dist > MaxDistanceSoFar)
                {
                    FurthestPoint = WorldPoint;
                    MaxDistanceSoFar = Dist;
                }
            }
        } break;

        case collider_type::flatpoly:
        {
            flatpoly_collider *FlatPoly = (flatpoly_collider *)Collider;

            Assert(FlatPoly->PointCloud);
            Assert(FlatPoly->PointCount > 0);

            float MaxDistanceSoFar = -FLT_MAX;
#if 0
            vec3 a = FlatPoly->PointCloud[0];
            vec3 b = FlatPoly->PointCloud[1];
            vec3 c = FlatPoly->PointCloud[2];
            vec3 n = Normalize(Cross(b-a, c-a));
#endif
            for (uint32 i = 0; i < FlatPoly->PointCount; ++i)
            {
                vec3 WorldPoint = FlatPoly->PointCloud[i];
                float Dist = Dot(WorldPoint, Direction);
                if (Dist > MaxDistanceSoFar)
                {
                    FurthestPoint = WorldPoint;
                    MaxDistanceSoFar = Dist;
                }
#if 0
                // NOTE(Kevin): Artificial thickening of the wall like this helps a lot
                //              to avoid tunneling, but:
                //              1. I won't be able to only resolve along wall normal
                //              2. It will be an invisible collider
                WorldPoint = WorldPoint - n * 16.f;
                Dist = Dot(WorldPoint, Direction);
                if (Dist > MaxDistanceSoFar)
                {
                    FurthestPoint = WorldPoint;
                    MaxDistanceSoFar = Dist;
                }
#endif
            }
        } break;

        case collider_type::invalid:
        default: Assert(0); break;
    }

    return FurthestPoint;
}

#define GJKMin(A, B) ((A) < (B) ? (A) : (B))
#define GJKSameDirection(Dir, A_to_Origin) (Dot(Dir, A_to_Origin) > 0.f)
#define GJKAbsoluteFloat(A) ((A) < 0.f ? (-(A)) : (A))

internal void GJKSimplexPushFront(gjk_simplex *Simplex, vec3 P)
{
    Simplex->Vertices[3] = Simplex->Vertices[2];
    Simplex->Vertices[2] = Simplex->Vertices[1];
    Simplex->Vertices[1] = Simplex->Vertices[0];
    Simplex->Vertices[0] = P;
    Simplex->VertexCount = GJKMin(Simplex->VertexCount + 1, 4);
}

internal vec3 GJKSupportFunction(collider_base *A, collider_base *B, vec3 Direction)
{
    // https://en.wikipedia.org/wiki/Support_function
    return FindFurthestPointOnConvexShell(A, Direction) - 
           FindFurthestPointOnConvexShell(B, -Direction);
}

internal bool32 GJKLine(gjk_simplex *Capture, vec3 *Direction)
{
    Assert(Capture && Direction);
    Assert(Capture->VertexCount == 2);
    vec3 A = Capture->Vertices[0];
    vec3 B = Capture->Vertices[1];
    vec3 AB = B - A;

    if (GJKSameDirection(AB, -A))
    {
        // TODO(Kevin): Handle case where initial search direction aligns with
        // the collision vector between two spheres. The following doesn't work
        // in that case.
        *Direction = Cross(Cross(AB, -A), AB);
    }
    else
    {
        Capture->Vertices[0] = A;
        Capture->VertexCount = 1;
        // NOTE(Kevin): vector A to origin is just -A
        *Direction = -A;

        // OLD NOTE:
        //     THIS PATH WILL SIGNAL THAT THERE IS NO COLLISION, but it might waste a support call
        //     Is there a way to stop the algorithm here?
        //     if origin is past point A, then it's impossible to be included in the simplex.
        //     I need to figure out how to stop the algorithm here and return that they are not intersecting.
    }

    return false;
}

internal bool32 GJKTriangle(gjk_simplex *Capture, vec3 *Direction)
{
    Assert(Capture && Direction);
    Assert(Capture->VertexCount == 3);
    vec3 A = Capture->Vertices[0];
    vec3 B = Capture->Vertices[1];
    vec3 C = Capture->Vertices[2];
    vec3 AB = B - A;
    vec3 AC = C - A;
    vec3 TriangleNormal = Cross(AB, AC);

    if (GJKSameDirection(Cross(TriangleNormal, AC), -A))
    {
        if (GJKSameDirection(AC, -A))
        {
            Capture->Vertices[0] = A;
            Capture->Vertices[1] = C;
            Capture->VertexCount = 2;
            *Direction = Cross(Cross(AC, -A), AC);
        }
        else
        {
            Capture->Vertices[0] = A;
            Capture->VertexCount = 1;
            *Direction = -A;
        }
    }
    else
    {
        if (GJKSameDirection(Cross(AB, TriangleNormal), -A))
        {
            if (GJKSameDirection(AB, -A))
            {
                Capture->Vertices[0] = A;
                Capture->Vertices[1] = B;
                Capture->VertexCount = 2;
                *Direction = Cross(Cross(AB, -A), AB);
            }
            else
            {
                Capture->Vertices[0] = A;
                Capture->VertexCount = 1;
                *Direction = -A;
            }
        }
        else
        {
            // The origin must be above or below this triangle
            if (GJKSameDirection(TriangleNormal, -A))
            {
                Capture->Vertices[0] = A;
                Capture->Vertices[1] = B;
                Capture->Vertices[2] = C;
                Capture->VertexCount = 3;
                *Direction = TriangleNormal;
            }
            else
            {
                Capture->Vertices[0] = A;
                Capture->Vertices[1] = C;
                Capture->Vertices[2] = B;
                Capture->VertexCount = 3;
                *Direction = -TriangleNormal;
            }
        }
    }

    return false;
}

internal bool32 GJKTetrahedron(gjk_simplex *Capture, vec3 *Direction)
{
    Assert(Capture && Direction);
    Assert(Capture->VertexCount == 4);
    vec3 A = Capture->Vertices[0];
    vec3 B = Capture->Vertices[1];
    vec3 C = Capture->Vertices[2];
    vec3 D = Capture->Vertices[3];
    vec3 AB = B - A;
    vec3 AC = C - A;
    vec3 AD = D - A;

    /* ??
     * ORDER NEEDS TO BE ABC first, ACD second, ADB third OR IT MIGHT INFINITELY
     * LOOP THE TETRAHEDRON AND TRIANGLE CASES BUT I DON'T UNDERSTAND WHY
     */

    vec3 ABCNormal = Cross(AB, AC);
    if (GJKSameDirection(ABCNormal, -A))
    {
        Capture->Vertices[0] = A;
        Capture->Vertices[1] = B;
        Capture->Vertices[2] = C;
        Capture->VertexCount = 3;
        return GJKTriangle(Capture, Direction);
    }

    vec3 ACDNormal = Cross(AC, AD);
    if (GJKSameDirection(ACDNormal, -A))
    {
        Capture->Vertices[0] = A;
        Capture->Vertices[1] = C;
        Capture->Vertices[2] = D;
        Capture->VertexCount = 3;
        return GJKTriangle(Capture, Direction);
    }

    vec3 ADBNormal = Cross(AD, AB);
    if (GJKSameDirection(ADBNormal, -A))
    {
        Capture->Vertices[0] = A;
        Capture->Vertices[1] = D;
        Capture->Vertices[2] = B;
        Capture->VertexCount = 3;
        return GJKTriangle(Capture, Direction);
    }

    // NOTE(Kevin):
    // In this case, the origin is not in any of the directions of the normals
    // of the three triangles formed with A and we know that the origin is not
    // "below" the triangle (imagine BCD as the base) because we just added
    // the point A. Since the origin is in the direction of Direction given to 
    // this function, the origin must be inside/captured by the tetrahedron.
    return true;
}

internal bool32 GJKNextSimplex(gjk_simplex *Capture, vec3 *OutNextDirection)
{
    // Checks to see what the next simplex should be and what direction
    // to search in next.
    // Only GJKTetrahedron returns true if it captures the origin.
    switch (Capture->VertexCount) 
    {
        case 2: return GJKLine(Capture, OutNextDirection);
        case 3: return GJKTriangle(Capture, OutNextDirection);
        case 4: return GJKTetrahedron(Capture, OutNextDirection);
        default: Assert(0); return false;
    }
}

#if HANDMADE_INTERNAL
global u32 GJKCallCounter = 0;
global u32 EPACallCounter = 0;
#endif
bool32 GJK(collider_base *A, collider_base *B, gjk_simplex *OutSimplex)
{
    // NOTE(Kevin): Returns true if the origin is captured by the simplex
    // during GJK algorithm. OutSimplex is populated during the algorithm.
    // In the case that GJK returns true, the caller may run EPA with the 
    // capture Simplex.

#if HANDMADE_INTERNAL
    ++GJKCallCounter;
#endif

    gjk_simplex BackupSimplex;
    if (OutSimplex)
    {
        *OutSimplex = {};
    }
    else
    {
        OutSimplex = &BackupSimplex;
    }

    // TODO(Kevin): make initial search direction `world origin - (A origin - B origin)`
    vec3 Support = GJKSupportFunction(A, B, { 1.f,1.f,1.f });
    GJKSimplexPushFront(OutSimplex, Support);
    vec3 SearchDirection = -Support;

    for (;;)
    {
        Support = GJKSupportFunction(A, B, SearchDirection);

        if (Dot(Support, SearchDirection) <= 0)
        {
            // Previous support point was already the furthest along direction
            return false;
        }

        GJKSimplexPushFront(OutSimplex, Support);

        if (GJKNextSimplex(OutSimplex, &SearchDirection))
        {
            return true;
        }

        // TODO(Kevin): Add fallback in case infinite loop
    }
}


/// EPA

#define EPA_MAX_ITERATIONS 10
#define EPA_DISTANCE_EPSILON 0.01f

#define EPA_MAX_VERTICES (4 + EPA_MAX_ITERATIONS)
// NOTE(Kevin): We are guaranteed a convex polyhedra therefore
// the number of faces cannot exceed more than twice the vertices.
#define EPA_MAX_FACES (EPA_MAX_VERTICES * 2 - 4)
#define EPA_MAX_EDGES (EPA_MAX_FACES * 3)

struct epa_face
{
    int Indices[3];
    vec3 Normal;
    float DistFromOrigin;
};

struct epa_edge
{
    int IndexA;
    int IndexB;
};

internal epa_face EPABuildFace(vec3 *Vertices, int Index0, int Index1, int Index2)
{
    epa_face Result;
    Result.Indices[0] = Index0;
    Result.Indices[1] = Index1;
    Result.Indices[2] = Index2;

    vec3 A = Vertices[Index0];
    vec3 B = Vertices[Index1];
    vec3 C = Vertices[Index2];
    Result.Normal = Normalize(Cross(B - A, C - A));

    // Distance from origin is the Component of vector(origin To A) projected onto the normal
    Result.DistFromOrigin = Dot(Result.Normal, A);
    if (Result.DistFromOrigin < 0.f)
    {
        Result.Normal *= -1.f;
        Result.DistFromOrigin *= -1.f;
    }

    return Result;
}

internal bool32 EPAEdgeEquals(epa_edge Lhs, epa_edge Rhs)
{
    if (Lhs.IndexA == Rhs.IndexA)
    {
        return Lhs.IndexB == Rhs.IndexB;
    }
    else
    {
        return Lhs.IndexA == Rhs.IndexB && Lhs.IndexB == Rhs.IndexA;
    }
}

internal void EPAKeepEdgeIfUnique(epa_edge *UniqueEdges, uint32 *UniqueEdgesCount, int IndexA, int IndexB)
{
    epa_edge Edge;
    // Index order matters for reconstruction later. For edge comparison,
    // the new edge to check should be in reverse order of the edge that
    // may exist, but I wrote EPAEdgeEquals to work either way.
    Edge.IndexA = IndexA;
    Edge.IndexB = IndexB;

    uint32 ExistingEdgeIndex = 0;
    for (; ExistingEdgeIndex < *UniqueEdgesCount; ++ExistingEdgeIndex)
    {
        if (EPAEdgeEquals(Edge, UniqueEdges[ExistingEdgeIndex]))
        {
            break;
        }
    }

    if (ExistingEdgeIndex == *UniqueEdgesCount)
    {
        Assert(*UniqueEdgesCount < EPA_MAX_EDGES);
        UniqueEdges[(*UniqueEdgesCount)++] = Edge;
    }
    else
    {
        // shared edge so it shouldn't exist; remove the previously added edge
        UniqueEdges[ExistingEdgeIndex] = UniqueEdges[--(*UniqueEdgesCount)];
    }
}

epa_result EPA(collider_base *A, collider_base *B, gjk_simplex *Simplex)
{
    Assert(Simplex->VertexCount == 4);
#if HANDMADE_INTERNAL
    ++EPACallCounter;
#endif

    // The initial polytope
    vec3 Vertices[EPA_MAX_VERTICES] = {};
    uint32 VerticesCount = Simplex->VertexCount;
    Copy(Vertices, Simplex->Vertices, sizeof(Simplex->Vertices));

    epa_face Faces[EPA_MAX_FACES] = {};
    uint32 FacesCount = 4;
    Faces[0] = EPABuildFace(Vertices, 0, 1, 2);
    Faces[1] = EPABuildFace(Vertices, 0, 3, 1);
    Faces[2] = EPABuildFace(Vertices, 0, 2, 3);
    Faces[3] = EPABuildFace(Vertices, 1, 3, 2);

    vec3 NearestFaceNormal;
    float NearestFaceDist = FLT_MAX;

    epa_edge UniqueEdges[EPA_MAX_EDGES] = {};
    uint32 UniqueEdgesCount = 0;

    for (uint32 EPAIterations = 0;; ++EPAIterations)
    {
        // Find the nearest face
        NearestFaceDist = FLT_MAX;
        for (uint32 FaceIndex = 0; FaceIndex < FacesCount; ++FaceIndex)
        {
            epa_face Face = Faces[FaceIndex];
            if (Face.DistFromOrigin < NearestFaceDist)
            {
                NearestFaceNormal = Face.Normal;
                NearestFaceDist = Face.DistFromOrigin;
            }
        }

        // EPA loop can be infinite if we have curved edges - just return
        // whatever it has after X iterations. Also, it can go for a really long
        // time even if there aren't curved edges.
        // TODO(Kevin): This may cause the alg to erroneously end early for very
        // detailed convex hulls. Not really a huge concern right now, but maybe
        // just check if the NearestFaceNormal and NearestFaceDist start to converge
        // really closely => when there is little variation in the NearestFaceNormal
        // and NearestFaceDist between each polytope expansion iterations, we can be
        // confident to return.
        if (EPAIterations == EPA_MAX_ITERATIONS)
        {
            break;
        }

        // Is there a point to be added in this direction? If the point we
        // find is already the furthest along the search direction (the normal
        // of the nearest face) then the point already exists in the polytope
        // on the nearest face.
        vec3 SupportP = GJKSupportFunction(A, B, NearestFaceNormal);
        float DistAlongSearchDirection = Dot(NearestFaceNormal, SupportP);

        if (GJKAbsoluteFloat(DistAlongSearchDirection - NearestFaceDist) > EPA_DISTANCE_EPSILON)
        {
            UniqueEdgesCount = 0;
            // Delete faces; Remember edges that are shared with undeleted faces
            for (uint32 FaceIndex = 0; FaceIndex < FacesCount; ++FaceIndex)
            {
                epa_face Face = Faces[FaceIndex];

                // Faces in the same direction as SupportP should be removed
                if (GJKSameDirection(Face.Normal, SupportP))
                {
                    int Index0 = Face.Indices[0];
                    int Index1 = Face.Indices[1];
                    int Index2 = Face.Indices[2];

                    EPAKeepEdgeIfUnique(UniqueEdges, &UniqueEdgesCount, Index0, Index1);
                    EPAKeepEdgeIfUnique(UniqueEdges, &UniqueEdgesCount, Index1, Index2);
                    EPAKeepEdgeIfUnique(UniqueEdges, &UniqueEdgesCount, Index2, Index0);

                    Faces[FaceIndex] = Faces[--FacesCount];
                    --FaceIndex;
                }
            }
            // TODO(Kevin): Do I need this fallback? or can I just Assert here
            if (UniqueEdgesCount == 0)
            {
                NearestFaceNormal = vec3(0.f, 0.f, 0.f);
                NearestFaceDist = 0.f;
                break;
            }

            Assert(VerticesCount < EPA_MAX_VERTICES);
            Vertices[VerticesCount++] = SupportP;

            // Reconstruction
            for (uint32 EdgeIndex = 0; EdgeIndex < UniqueEdgesCount; ++EdgeIndex)
            {
                int Index0 = UniqueEdges[EdgeIndex].IndexA;
                int Index1 = UniqueEdges[EdgeIndex].IndexB;
                int Index2 = VerticesCount-1;
                Assert(FacesCount < EPA_MAX_FACES);
                Faces[FacesCount++] = EPABuildFace(Vertices, Index0, Index1, Index2);
            }
        }
        else
        {
            break;
        }
    }

    epa_result Result;
    Result.NearestFaceNormal = NearestFaceNormal;
    Result.NearestFaceDist = NearestFaceDist + EPA_DISTANCE_EPSILON;
    return Result;
}

