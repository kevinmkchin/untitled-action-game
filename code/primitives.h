#pragma once

/** Simple primitives, debug visualizations, and viewport interaction renderer
 * 
 *  Visualization with lines, discs, grids, etc.
 *  Pickable elements for picking things in the editor
 * 
 * */

struct billboard_t
{
    int Id;
    float Sz;
};

struct support_renderer_t
{
    void Initialize();
    // void Destroy();

public: // Primitives

    // Queue a solid disc to be drawn later
    void DrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color);
    void DrawSolidDisc(vec3 center, vec3 normal, float radius);
    // Queue a line to be drawn later
    void DrawLine(vec3 p1, vec3 p2, vec4 color);
    void DrawLine(vec3 p1, vec3 p2, vec4 color, float thickness);

    // Immediately draw all queued primitive elements to the active 
    // frame buffer. Resets/empties all queued buffers.
    void FlushPrimitives(const mat4 *projectionMatrix, const mat4 *viewMatrix, 
        GLuint sceneDepthTextureId, vec2 framebufferSize);
    // Immediately draw a flat grid to the active frame buffer
    void DrawGrid(float scale, mat3 rotation, vec3 translation, 
        const mat4 *projectionMatrix, const mat4 *viewMatrix, 
        GLuint sceneDepthTextureId, vec2 framebufferSize);

public: // Pickables
    
    // Get color id from u32
    vec3 HandleIdToRGB(u32 id);

    // Add a pickable disc handle
    void DoDiscHandle(u32 id, vec3 worldpos, vec3 normal, float radius);
    // Add arbitrary triangles to pickable handles vertex buffer
    //      worldpos x y z color r g b
    void AddTrianglesToPickableHandles(float *vertices, int count);
    // Immediately draw given vertex buffer to the active frame buffer
    //      Uses HANDLES_SHADER but works with any triangle vertex buffer of XYZRGB layout
    void DrawHandlesVertexArray_GL(float *vertexArrayData, u32 vertexArrayDataCount, 
        float *projectionMat, float *viewMat);

    // TODO I need a texture atlas and a map from entity_types_t to a rect in that texture atlas

    // Queue a pickable billboard to be drawn with the provided parameters
    void DoPickableBillboard(u32 id, vec3 worldpos, vec3 normal, billboard_t billboard);
    // Immediately draw pickable editor billboards
    //      Can be used for color id picking or to draw the actual billboard textures
    void DrawPickableBillboards_GL(float *ProjectionMat, float *ViewMat, bool UseColorIds);
    void ClearPickableBillboards();

    // Immediately draw all pickable handles and return id of clicked handle
    //      Reset pickable handles buffers
    u32 FlushHandles(ivec2 clickat, const GPUFrameBuffer activeSceneTarget,
        const mat4& activeViewMatrix, const mat4& activeProjectionMatrix, bool orthographic);
};

extern support_renderer_t SupportRenderer;
