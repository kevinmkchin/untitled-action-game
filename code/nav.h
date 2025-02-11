#pragma once

#include <DebugDraw.h>

bool CreateRecastNavMesh();
void DestroyRecastNavMesh();
void DetourTesting();
void DoDebugDrawRecast(float *ProjMatrix, float *ViewMatrix, enum recast_debug_drawmode DrawMode);

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
    void Init();
    void Destroy();

    void Ready(float *ProjMatrix, float *ViewMatrix);

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

private:
    GLenum CurrentPrimitiveDrawMode = GL_TRIANGLES;

    GPUMesh DebugDrawMesh;
    dynamic_array<float> VertexBuffer;

    GPUShader RECAST_DEBUG_GL3_SHADER;
    const char* RECAST_DEBUG_GL3_VS =
    "#version 330\n"
    "layout (location = 0) in vec3 VertexPos;\n"
    "layout (location = 1) in vec4 VertexColor;\n"
    "layout (location = 2) in vec2 VertexUV;\n"
    "out vec4 Color;\n"
    "out vec2 UV;\n"
    "uniform mat4 ViewMatrix;\n"
    "uniform mat4 ProjMatrix;\n"
    "void main()\n"
    "{\n"
    "    Color = VertexColor;\n"
    "    UV = VertexUV;\n"
    "    gl_Position = ProjMatrix * ViewMatrix * vec4(VertexPos, 1.0);\n"
    "}";

    const char* RECAST_DEBUG_GL3_FS = 
    "#version 330\n"
    "\n"
    "in vec4 Color;\n"
    "in vec2 UV;\n"
    "\n"
    "layout(location = 0) out vec4 OutColor;\n"
    "\n"
    "uniform sampler2D Texture0;\n"
    "uniform int UseTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    if (UseTexture == 0) { \n"
    "        OutColor = Color;\n"
    "    } else { \n"
    "        OutColor = texture(Texture0, UV);\n"
    "    } \n"
    "}";
};

