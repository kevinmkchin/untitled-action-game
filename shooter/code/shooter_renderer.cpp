#include "shooter_renderer.h"
#include "shooter_renderer_opengl.h"

void renderer::PushRenderItems(render_item *RenderItems, uint32 RenderItemsCount)
{
    // TODO(Kevin): Sort render items

    open_gl_model_instance *InstanceDataArray = 0;
    uint32 *InstanceDataArrayCount = 0;
    OpenGLGetPerInstanceDataArray((open_gl *)this, &InstanceDataArray, &InstanceDataArrayCount);

    for (uint32 SortedIndex = 0; SortedIndex < RenderItemsCount; ++SortedIndex)
    {
        render_item *Item = &RenderItems[SortedIndex];

        // TODO(Kevin): If using lightmap then maybe go into diff pipeline?

        mat4 ModelMatrix = TranslationMatrix(Item->WorldPosition) * 
                           RotationMatrix(Item->WorldRotation) * 
                           ScaleMatrix(Item->WorldScale.x, Item->WorldScale.y, Item->WorldScale.z);
        
        open_gl_model_instance *InstanceData = &InstanceDataArray[SortedIndex];

        *InstanceData = {};
        InstanceData->WorldFromModel = ModelMatrix;
        InstanceData->Descriptor = Item->Descriptor;
        // TODO(Kevin): whether an instance does sunlight or not is baked into light cube info!
        InstanceData->DoSunLight = true;

        ++(*InstanceDataArrayCount);
    }
}

mesh_handle renderer::UploadMeshData(PNUV1_vertex *VB, uint32 VC, uint32 *IB, uint32 IC)
{
    mesh_handle Result = OpenGLUploadMesh((open_gl *)this, VB, VC, IB, IC);
    return Result;
}

mesh_handle renderer::UploadMeshData(PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC)
{
    mesh_handle Result = OpenGLUploadMesh((open_gl *)this, VB, VC, IB, IC);
    return Result;
}

texture_handle renderer::UploadTexture2DData(void *Bitmap, uint32 W, uint32 H, 
                                             pixel_format Format, bool32 LinearFiltering)
{
    texture_handle Result = OpenGLUploadTexture2D((open_gl *)this, Bitmap, W, H, Format, LinearFiltering);
    return Result;
}

