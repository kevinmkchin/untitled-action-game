#include "shooter_renderer.h"
#include "shooter_renderer_opengl.h"

uint32 renderer::SomeTestServiceToGame(uint32 Value)
{
    uint32 Result = Value * 2;
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

