#include "shooter_colliders.h"


vec3 FindFurthestPointOnConvexShell(collider_base *Collider, vec3 Direction)
{
    vec3 FurthestPoint = {};

    switch (Collider->Type)
    {
        case collider_type::mesh:
        {
            mesh_collider *Mesh = (mesh_collider *)Collider;

            ASSERT(Mesh->PointCloud);
            ASSERT(Mesh->PointCount > 0);

            float MaxDistanceSoFar = -FLT_MAX;
            for (u32 i = 0; i < Mesh->PointCount; ++i)
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

            ASSERT(FlatPoly->PointCloud);
            ASSERT(FlatPoly->PointCount > 0);

            float MaxDistanceSoFar = -FLT_MAX;
#if 0
            vec3 a = FlatPoly->PointCloud[0];
            vec3 b = FlatPoly->PointCloud[1];
            vec3 c = FlatPoly->PointCloud[2];
            vec3 n = Normalize(Cross(b-a, c-a));
#endif
            for (u32 i = 0; i < FlatPoly->PointCount; ++i)
            {
                vec3 WorldPoint = FlatPoly->PointCloud[i];
                float Dist = Dot(WorldPoint, Direction);
                if (Dist > MaxDistanceSoFar)
                {
                    FurthestPoint = WorldPoint;
                    MaxDistanceSoFar = Dist;
                }
#if 0
                // I don't think this artificial thickening is helping much
                Dist = Dot(WorldPoint - n * 1.f, Direction);
                if (Dist > MaxDistanceSoFar)
                {
                    FurthestPoint = WorldPoint;
                    MaxDistanceSoFar = Dist;
                }
#endif
            }
        } break;

        case collider_type::invalid:
        default: ASSERT(0); break;
    }

    return FurthestPoint;
}

#define GJKMin(a, b) ((a) < (b) ? (a) : (b))
#define GJKSameDirection(Dir, A_to_Origin) (Dot(Dir, A_to_Origin) > 0.f)

void GJKSimplexPushFront(gjk_simplex *Simplex, vec3 P)
{
    Simplex->Vertices[3] = Simplex->Vertices[2];
    Simplex->Vertices[2] = Simplex->Vertices[1];
    Simplex->Vertices[1] = Simplex->Vertices[0];
    Simplex->Vertices[0] = P;
    Simplex->VertexCount = GJKMin(Simplex->VertexCount + 1, 4);
}

vec3 GJKSupportFunction(collider_base *A, collider_base *B, vec3 Direction)
{
    // https://en.wikipedia.org/wiki/Support_function
    return FindFurthestPointOnConvexShell(A, Direction) - 
           FindFurthestPointOnConvexShell(B, -Direction);
}

bool GJKLine(gjk_simplex *Capture, vec3 *Direction)
{
#if SHOOTER_SLOW
    ASSERT(Capture && Direction);
    ASSERT(Capture->VertexCount == 2);
#endif
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

bool GJKTriangle(gjk_simplex *Capture, vec3 *Direction)
{
#if SHOOTER_SLOW
    ASSERT(Capture && Direction);
    ASSERT(Capture->VertexCount == 3);
#endif
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

bool GJKTetrahedron(gjk_simplex *Capture, vec3 *Direction)
{
#if SHOOTER_SLOW
    ASSERT(Capture && Direction);
    ASSERT(Capture->VertexCount == 4);
#endif
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

bool GJKNextSimplex(gjk_simplex *Capture, vec3 *OutNextDirection)
{
    // Checks to see what the next simplex should be and what direction
    // to search in next.
    // Only GJKTetrahedron returns true if it captures the origin.
    switch (Capture->VertexCount) 
    {
        case 2: return GJKLine(Capture, OutNextDirection);
        case 3: return GJKTriangle(Capture, OutNextDirection);
        case 4: return GJKTetrahedron(Capture, OutNextDirection);
        default: ASSERT(0); return false;
    }
}

bool GJK(collider_base *A, collider_base *B, gjk_simplex *OutSimplex)
{
    // NOTE(Kevin): Returns true if the origin is captured by the simplex
    // during GJK algorithm. OutSimplex is populated during the algorithm.
    // In the case that GJK returns true, the caller may run EPA with the 
    // capture Simplex.

    *OutSimplex = {};

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

