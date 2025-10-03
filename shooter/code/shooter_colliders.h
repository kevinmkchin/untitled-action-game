#pragma once

enum class collider_type
{
    mesh,
    flatpoly,
    // TODO(Kevin): aabb should be GJK acceptible collider type too
    // because when we are baking the spatial hash, we can get exactly
    // the aabb cells that intersect with the flatpoly rather than all
    // the aabb cells that overlap with the aabb of the flatpoly. during
    // baking, i should prob narrow down the aabb cells first (broadphase)
    // before checking the narrowed down cells against the flatpoly using GJK
    invalid
};

struct collider_base
{
    collider_type Type;
    // TODO(Kevin): perhaps an "Origin" this collider is relative to?
    // This is important to preserve better precision for world space colliders
    // far away from world origin.
    // The GJK alg could even, before starting, transform both origins s.t. the two
    // colliders are relative to some deterministic middle point between them we treat
    // as the origin. This way, the GJK alg itself is also more precise.
};

struct mesh_collider
{
    mesh_collider()
    {
        Base.Type = collider_type::mesh;
    }

    collider_base Base;
    vec3 *PointCloud;
    uint32 PointCount;
};

struct flatpoly_collider
{
    flatpoly_collider()
    {
        Base.Type = collider_type::flatpoly;
    }

    collider_base Base;
    vec3 *PointCloud;
    uint32 PointCount;
};

struct gjk_simplex
{
    vec3 Vertices[4];
    uint32 VertexCount; // its dimension + 1
};

bool GJK(collider_base *A, collider_base *B, gjk_simplex *OutSimplex);

struct epa_result
{
    // NOTE(Kevin): This can be 0,0,0 for the normal
    vec3 NearestFaceNormal;
    float NearestFaceDist;
};

epa_result EPA(collider_base *A, collider_base *B, gjk_simplex *Simplex);
