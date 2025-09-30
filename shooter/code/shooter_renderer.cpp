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

        mat4 ModelMatrix = TranslationMatrix(Item->WorldPosition) * 
                           RotationMatrix(Item->WorldRotation) * 
                           ScaleMatrix(Item->WorldScale.x, Item->WorldScale.y, Item->WorldScale.z);
        
        open_gl_model_instance *InstanceData = &InstanceDataArray[SortedIndex];

        *InstanceData = {};
        InstanceData->WorldFromModel = ModelMatrix;
        InstanceData->Descriptor.Mesh = Item->Descriptor.Mesh;
        InstanceData->Descriptor.Albedo = Item->Descriptor.Albedo;
        InstanceData->DoSunLight = false;

        if (Item->UseLightmap)
        {
            InstanceData->Descriptor.Lightmap = Item->Descriptor.Lightmap;

            // NOTE(Kevin): Because Ambient+Diffuse is a multiplier to Albedo sample, it
            // cannot be zero even when we are solely relying on lightmap.
            for (uint32 Side = 0; Side < ArrayCount(InstanceData->AmbientCube); ++Side)
            {
                // TODO(Kevin): When everything is going through StaticLit pipeline, I could
                // use the ambient cube to apply tint to things, etc.
                InstanceData->AmbientCube[Side] = 1.0f;
            }
        }
        else
        {
            InstanceData->Descriptor.Lightmap = WhiteTexture_1x1;

            if (Item->UseLightCubes)
            {
                // TODO(Kevin): Query light cube cache for ambient cube and significant light sources

                // TODO(Kevin): Whether an instance does sunlight or not is baked into light cube info!
                InstanceData->DoSunLight = true;
            }
        }

        ++(*InstanceDataArrayCount);
    }
}

// mesh_handle renderer::UploadMeshData(PNUV1_vertex *VB, uint32 VC, uint32 *IB, uint32 IC)
// {
//     mesh_handle Result = OpenGLUploadMesh((open_gl *)this, VB, VC, IB, IC);
//     return Result;
// }

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

