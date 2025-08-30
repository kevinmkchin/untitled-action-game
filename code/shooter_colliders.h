#pragma once

enum class collider_type
{
    mesh,
    flatpoly,
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
    u32 PointCount;
};

struct flatpoly_collider
{
    flatpoly_collider()
    {
        Base.Type = collider_type::flatpoly;
    }

    collider_base Base;
    vec3 *PointCloud;
    u32 PointCount;
};

struct gjk_simplex
{
    vec3 Vertices[4];
    u32 VertexCount; // its dimension + 1
};

bool GJK(collider_base *A, collider_base *B, gjk_simplex *OutSimplex);
