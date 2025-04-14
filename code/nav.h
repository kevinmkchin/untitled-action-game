#pragma once

#include "common.h"
#include "game.h"

bool CreateRecastNavMesh(game_state *GameState);
void DestroyRecastNavMesh();

#define MAX_SMOOTH 300
bool FindSmoothPathTo(vec3 Origin, vec3 Target, float *SmoothPath, int *SmoothPathCount);

void GetRandomPointOnNavMesh(float *Point);

#if INTERNAL_BUILD
#include "mem.h"
#include <DebugDraw.h>

void DebugDrawRecast(duDebugDraw *DebugDrawer, enum recast_debug_drawmode DrawMode);
void DebugDrawFollowPath(struct support_renderer_t *SupportRenderer);

enum recast_debug_drawmode
{
    DRAWMODE_NAVMESH,
    DRAWMODE_NAVMESH_TRANS,
    DRAWMODE_NAVMESH_BVTREE,
    DRAWMODE_NAVMESH_NODES,
    DRAWMODE_NAVMESH_INVIS,
    DRAWMODE_MESH,
    DRAWMODE_VOXELS,
    DRAWMODE_VOXELS_WALKABLE,
    DRAWMODE_COMPACT,
    DRAWMODE_COMPACT_DISTANCE,
    DRAWMODE_COMPACT_REGIONS,
    DRAWMODE_REGION_CONNECTIONS,
    DRAWMODE_RAW_CONTOURS,
    DRAWMODE_BOTH_CONTOURS,
    DRAWMODE_CONTOURS,
    DRAWMODE_POLYMESH,
    DRAWMODE_POLYMESH_DETAIL,
    MAX_DRAWMODE
};

struct recast_debug_draw_gl3_t : duDebugDraw
{
    void Ready();

// Implementations
    virtual void depthMask(bool state);
    virtual void texture(bool state);
    virtual void begin(duDebugDrawPrimitives prim, float size = 1.0f);
    virtual void vertex(const float* pos, unsigned int color);
    virtual void vertex(const float x, const float y, const float z, unsigned int color);
    virtual void vertex(const float* pos, unsigned int color, const float* uv);
    virtual void vertex(const float x, const float y, const float z, unsigned int color, const float u, const float v);
    virtual void end();
    // virtual unsigned int areaToCol(unsigned int area);

    struct game_state *GameState = nullptr;
    struct support_renderer_t *SupportRenderer = nullptr;
private:
    duDebugDrawPrimitives CurrentPrimitiveDrawMode = DU_DRAW_TRIS;
    float CurrentSize = 1.f;
    dynamic_array<float> VertexBuffer;

};

#endif // INTERNAL_BUILD
