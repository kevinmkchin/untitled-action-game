#pragma once


struct ModelGLTF
{
    struct GPUMeshIndexed *meshes   = NULL;
    struct GPUTexture     *color    = NULL;
    // animations and bones and shit
};

void FreeModelGLTF(ModelGLTF model);
void RenderModelGLTF(ModelGLTF model);
bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath);
