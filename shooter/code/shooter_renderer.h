#ifndef SHOOTER_RENDERER_H
#define SHOOTER_RENDERER_H

#include "shooter_types.h"
#include "shooter_vertext.h"

// TODO(Kevin): Should these mesh/texture handles, render items live on the game side?
// and the renderer+assetman systems implement ways to upload them and render them?
typedef uint32 mesh_handle;
typedef uint32 texture_handle;
typedef uint32 font_handle;
#define InvalidHandle 0xFFFFFFFF

enum pixel_format
{
    R_U8,
    R_Float,
    RGBA_U8,
    RGBA_Float,
    BGRA_U8,
    BGRA_Float,
};

struct PNUV2_vertex
{
    vec3 XYZ;
    vec3 IJK;
    vec2 UV0;
    vec2 UV1;
};

struct gui_vertex
{
    vec2 PixelP; // 0,0 is top left
    vec2 UV0;
    vec4 Color;
};

struct render_descriptor
{
    mesh_handle    Mesh;
    texture_handle Albedo;
    texture_handle Lightmap;

    bool operator==(render_descriptor &Other)
    {
        bool Result = Mesh == Other.Mesh &&
                      Albedo == Other.Albedo &&
                      Lightmap == Other.Lightmap;
        return Result;
    }
    bool operator!=(render_descriptor &Other)
    {
        bool Result = !(*this == Other);
        return Result;
    }
};

struct render_item
{
    // TODO(Kevin): which rendering pipeline to use?

    render_descriptor Descriptor;
    bool32 UseLightCubes; // TODO(Kevin): branch on UseLightCubes
    bool32 UseLightmap;
    vec3 ForcedAmbient;
    vec3 WorldPosition;
    quat WorldRotation;
    vec3 WorldScale;

    // bones
};

struct rendering_pipeline
{
    uint32 ShaderID;
    bool32 DepthTest;

    // VertexInputLayout
    // Rasterizer state (culling, polygon mode, scissor test)
    // Depth/stencil state (depth test, write mask, stencil rules)
    // Blend state
    // Render targets formats (color, depth attachment formats)
};

struct renderer_font
{
    vtxt_font FontInfo;
    texture_handle FontAtlasHandle;
};

struct create_font_result
{
    font_handle Handle;
    vtxt_font *FontInfo;
    texture_handle FontAtlasHandle;
};

//
// NOTE(Kevin): Renderer Frontend
//
struct renderer
{
// NOTE(Kevin): placement new to explicitly invoke constructor and get valid vfptr!
#define ConstructRendererServiceAt(Address) new(Address) renderer()

    uint32 BackBufferW;
    uint32 BackBufferH;

    rendering_pipeline StaticLit;
    rendering_pipeline UIPipeline;

    texture_handle WhiteTexture_1x1;

    uint32 StaticLitItemsMaxCount;

    renderer_font HandleToFont[8];
    u32 FontCount;

    mat4 ViewFromWorld;

    // NOTE(Kevin): Service APIs to the game must be virtual to avoid linking with
    // definition. vtable essentially provides the function table of callable services
    // from the game layer. Therefore, renderer MUST be constructed to have a valid vtable
    // ptr. C-style function table will be more stable, but for now this is more convenient.
    virtual void GetRenderTargetDimensions(uint32 *Width, uint32 *Height);
    virtual void PushViewMatrix(mat4 ViewFromWorld);

    virtual void PushRenderItems(render_item *RenderItems, uint32 RenderItemsCount);
    virtual void PushGuiDrawRequests(memory_arena *Scratch, struct gui_draw_request *Requests, uint32 RequestsCount);
    virtual void PushLine(vec3 WorldP0, vec3 WorldP1, vec3 Color, real32 PixelWidth);

    virtual mesh_handle UploadMeshData(PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
    virtual void DeleteMesh(mesh_handle Mesh);
    virtual texture_handle UploadTexture2DData(void *Bitmap, uint32 W, uint32 H, pixel_format Format, 
                                               bool32 LinearFiltering);

    virtual create_font_result DEBUGCreateBitmapFont_Monospace(void *Bitmap, u32 W, u32 H,
                                                               pixel_format Format,u32 NewLineGap);
    virtual create_font_result DEBUGCreateBitmapFont_TrueType(thread_context *Thread, platform_api PlatformAPI,
                                                              s32 FontHeightPx, char *TrueTypeFileName);

};

/*

DrawLine

*/

#endif // SHOOTER_RENDERER_H
