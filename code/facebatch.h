#pragma once

#include "gpu_resources.h"
#include "shaders.h"

struct face_batch_t
{
    // Set of triangles that all use the same texture
    // world pos x y z, normal i j k, uv1 u v, uv2 u v
    GPUTexture ColorTexture;
    GPUTexture LightMapTexture;
    u32 VAO = 0;
    u32 VBO = 0;
    u32 VertexCount = 0;
};

void CreateFaceBatch(face_batch_t *FaceBatch);
void RebindFaceBatch(face_batch_t *FaceBatch, size_t SizeInBytes, float *Data);
void RenderFaceBatch(const GPUShader *Shader, const face_batch_t *FaceBatch);
void DeleteFaceBatch(const face_batch_t *FaceBatch);
