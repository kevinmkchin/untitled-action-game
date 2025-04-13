#pragma once

/** Simple primitives, debug visualizations, and viewport interaction renderer
 * 
 *  Visualization with lines, discs, grids, etc.
 *  Pickable elements for picking things in the editor
 * 
 * */
#include "common.h"
#include "mem.h"
#include "shaders.h"
#include "gpu_resources.h"

vec3 CalculateTangent(vec3 Normal);

struct support_renderer_t
{
    void Initialize();
    void Destroy();

    void NewFrame();

public: // Primitives

    fixed_array<float> PRIMITIVE_TRIS_VB;
    fixed_array<float> PRIMITIVE_LINES_VB;
    fixed_array<float> PRIMITIVE_FATLINES_VB;

    // Queue a solid disc to be drawn later
    void DrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color);
    void DrawSolidDisc(vec3 center, vec3 normal, float radius);
    // Queue a solid rect to be drawn later
    void DrawSolidRect(vec3 center, vec3 normal, float halfWidth, vec4 color);
    void DrawColoredCube(vec3 center, float halfWidth,
        vec4 ColorPX, vec4 ColorNX, vec4 ColorPY, vec4 ColorNY, vec4 ColorPZ, vec4 ColorNZ);
    // Queue a line to be drawn later
    void DrawLine(vec3 p1, vec3 p2, vec4 color);
    void DrawLine(vec3 p1, vec3 p2, vec4 color, float thickness);
    void DrawTri(vec3 p1, vec3 p2, vec3 p3, vec4 color);

    // Immediately draw all queued primitive elements to the active 
    // frame buffer. Resets/empties all queued buffers.
    void FlushPrimitives(const mat4 *projectionMatrix, const mat4 *viewMatrix, 
        GLuint sceneDepthTextureId, vec2 framebufferSize);
    // Immediately draw a flat grid to the active frame buffer
    void DrawGrid(float scale, mat3 rotation, vec3 translation, 
        const mat4 *projectionMatrix, const mat4 *viewMatrix, 
        GLuint sceneDepthTextureId, vec2 framebufferSize);

private:
    u32 PRIM_VERTEX_POS_AND_COLOR_VAO;
    u32 PRIM_VERTEX_POS_AND_COLOR_VBO;
    u32 PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO;
    u32 PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO;

    GPUShader PRIMITIVES_TRIS_SHADER;
    GPUShader LINES_SHADER;
    GPUShader FATLINES_SHADER;

public: // Pickables

    fixed_array<float> HANDLES_VB;
    fixed_array<float> PICKABLE_BILLBOARDS_VB;
    
    // Get color id from u32
    vec3 HandleIdToRGB(u32 Id);

    // Add a pickable disc handle
    void DoDiscHandle(u32 Id, vec3 WorldPosition, vec3 WorldNormal, float Radius);
    // Add arbitrary triangles to pickable handles vertex buffer
    //      worldpos x y z color r g b
    void AddTrianglesToPickableHandles(float *Vertices, int Count);
    // Immediately draw given vertex buffer to the active frame buffer
    //      Uses HANDLES_SHADER but works with any triangle vertex buffer of XYZRGB layout
    void DrawHandlesVertexArray_GL(float *VertexBuffer, u32 VertexBufferCount, 
        float *ProjectionMat, float *ViewMat);

    // Queue a pickable billboard to be drawn with the provided parameters
    void DoPickableBillboard(u32 Id, vec3 WorldPos, 
        vec3 Normal, vec3 CamDirection, int BillboardId);
    // Immediately draw pickable editor billboards
    //      Can be used for color id picking or to draw the actual billboard textures
    void DrawPickableBillboards_GL(float *ProjectionMat, float *ViewMat, bool UseColorIds);
    void ClearPickableBillboards();

    // Immediately draw all pickable handles and return id of clicked handle
    //      Reset pickable handles buffers
    u32 FlushHandles(ivec2 clickat, const GPUFrameBuffer activeSceneTarget,
        const mat4& activeViewMatrix, const mat4& activeProjectionMatrix, bool orthographic);

public:
    // a texture atlas and a map from entity_types_t to a rect in that texture atlas
    GPUTexture EntityBillboardAtlas;
    vec4       EntityBillboardRectMap[64];
    float      EntityBillboardWidthMap[64];

    struct entity_billboard_data_t 
    {
        vec3 IdRGB;
        vec3 WorldPos;
        vec3 RightTangent;
        vec3 UpTangent;
        int BillboardId;
        float HowFarAlongCameraDirection;
    };
    std::vector<entity_billboard_data_t> BillboardsRequested;

private:
    GPUFrameBuffer mousePickingRenderTarget;

    u32 HANDLES_VAO = 0;
    u32 HANDLES_VBO = 0;
    GPUMesh PICKABLE_BILLBOARDS_MESH;
    u32 GRID_MESH_VAO;
    u32 GRID_MESH_VBO;

    GPUShader HANDLES_SHADER;
    GPUShader PICKABLE_BILLBOARDS_SHADER;
    GPUShader GRID_MESH_SHADER;
};