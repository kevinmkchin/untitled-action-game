#ifndef SHOOTER_RENDERER_H
#define SHOOTER_RENDERER_H

#include "shooter_types.h"

// TODO(Kevin): Should these mesh/texture handles, render items live on the game side?
// and the renderer+assetman systems implement ways to upload them and render them?
typedef uint32 mesh_handle;
typedef uint32 texture_handle;

enum pixel_format
{
    R_U8,
    R_Float,
    RGBA_U8,
    RGBA_Float,
    // TODO(Kevin): BGRA?
};

struct PNUV2_vertex
{
    vec3 XYZ;
    vec3 IJK;
    vec2 UV0;
    vec2 UV1;
};

//
// NOTE(Kevin): Renderer Frontend
//
struct renderer
{
// NOTE(Kevin): placement new to explicitly invoke the constructor
#define ConstructRendererServiceAt(Address) new(Address) renderer()

    uint32 BackBufferW;
    uint32 BackBufferH;

    platform_api PlatformAPI;

    // NOTE(Kevin): Service APIs to the game must be virtual to avoid linking with
    // definition. vtable essentially provides the function table of callable services
    // from the game layer. Therefore, renderer MUST be constructed to have a valid vtable
    // ptr. C-style function table will be more stable, but for now this is more convenient.
    virtual uint32 SomeTestServiceToGame(uint32 Value);

    virtual mesh_handle UploadMeshData(PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
    virtual texture_handle UploadTexture2DData(void *Bitmap, uint32 W, uint32 H, pixel_format Format, 
                                               bool32 LinearFiltering);
};

/*

DrawLine

*/

#endif // SHOOTER_RENDERER_H
