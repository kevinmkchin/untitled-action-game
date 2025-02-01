#pragma once

void InitPrimitivesAndHandlesSystems();
// void DestroyPrimitivesAndHandlesSystems?

void DrawGrid(float scale, mat3 rotation, vec3 translation, 
    const mat4 *projectionMatrix, const mat4 *viewMatrix, 
    GLuint sceneDepthTextureId, vec2 framebufferSize);

void PrimitiveDrawAll(const mat4 *projectionMatrix, const mat4 *viewMatrix, 
    GLuint sceneDepthTextureId, vec2 framebufferSize);

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color);

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius);

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color);

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color, float thickness);


struct billboard_t
{
    int Id;
    float Sz;
};

// Should probably rename "Handles" to "Pickables"
vec3 HandleIdToRGB(u32 id);

void DoDiscHandle(u32 id, vec3 worldpos, vec3 normal, float radius);

void DoPickableBillboard(u32 id, vec3 worldpos, vec3 normal, billboard_t billboard);

// add arbitrary triangles to pickable handles vertex buffer
// worldpos x y z color r g b
void AddTrianglesToPickableHandles(float *vertices, int count);

// Uses HANDLES_SHADER but works with any triangle vertex buffer of XYZRGB layout
void DrawHandlesVertexArray_GL(float *vertexArrayData, u32 vertexArrayDataCount, 
    float *projectionMat, float *viewMat);

// Draw pickable editor billboards
// Can be used for color id picking or to draw the actual billboard textures
void DrawPickableBillboards_GL(float *ProjectionMat, float *ViewMat, bool UseColorIds);
void temp_clear_pickablebillboards();

// draw all handles, return id of clicked handle
u32 FlushHandles(ivec2 clickat, const GPUFrameBuffer activeSceneTarget,
                 const mat4& activeViewMatrix, const mat4& activeProjectionMatrix, bool orthographic);
