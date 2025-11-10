#ifndef SHOOTER_RENDERER_DX11_H
#define SHOOTER_RENDERER_DX11_H

#include "shooter_renderer.h"
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

// NOTE(Kevin): Configurable counts for fixed size storage
#define STATIC_LIT_ITEMS_MAX_COUNT 10000
#define DX11_MESH_MAX_COUNT        11000
#define DX11_TEXTURE_MAX_COUNT     512

struct dx11_mesh
{
    ID3D11Buffer *VB;
    ID3D11Buffer *IB;
    u32 VertexStride;
    u32 IndicesCount;
    ID3D11InputLayout *Layout;
};

struct dx11_texture
{
    ID3D11Texture2D *Texture;
    ID3D11ShaderResourceView *SRV;
    ID3D11SamplerState *Sampler;
    DXGI_FORMAT Format;
};

struct dx11_structured_buffer
{
    ID3D11Buffer *Buffer;
    ID3D11ShaderResourceView *SRV;
};

struct dx11_vertex_stream
{
    ID3D11Buffer *Buffer;
    u32 Capacity;
    u32 Cursor;
};

struct vstream_frame_chunk
{
    void *WriteTo;
    memory_index ChunkSizeBytes;
};

struct gui_vstream_item
{
    umm OffsetBytes;
    u32 VertexCount;
    texture_handle ColorTexture;
    s32 ScissorX, ScissorY;
    u32 ScissorW, ScissorH;
};

struct alignas(16) static_lit_model_instance
{
    mat4 WorldFromModel;            // 64      : 64
    vec4 PointLightsPos[4];         // 64      : 128
    vec4 AmbientCube[6];            // 16*6=96 : 224
    s32 DoSunLight;                 // 4       : 228
    s32 PointLightsCount;           // 4       : 232
    real32 PointLightsAttLin[4];    // 4*4=16  : 248
    real32 PointLightsAttQuad[4];   // 4*4=16  : 264

    render_descriptor Descriptor;   // 12      : 276
    u32 _padding_[3];               // 4*3     : 288
};

struct alignas(16) static_lit_cb0
{
    mat4 ProjFromWorld;
    vec3 DirectionToSun;
};

struct alignas(16) static_lit_cb1
{
    u32 BaseInstance;
};

struct alignas(16) ui_cb0
{
    mat4 ClipFromScreen2D;
};

struct alignas(16) world_line_cb0
{
    mat4 ProjFromWorld;
    vec2 ViewportSize;
    float FadeOutZ;
};

struct world_line_instance
{
    vec3 P0;
    vec3 P1;
    real32 Thickness;
    vec4 Color;
};

struct dx11
{
    renderer Header;

    D3D_FEATURE_LEVEL ObtainedFeatureLevel;
    DXGI_FORMAT       BackBufferFormat;
    bool32            AllowVRRTearing;
    u32               FramesInFlight;
    // TODO(Kevin): These should be ->Release on shutdown
    IDXGISwapChain1 *SwapChain;
    ID3D11Device *Device;
    ID3D11DeviceContext *Context;
    IDXGIFactory2 *Factory2;

    ID3D11RenderTargetView *RenderTargetView;
    ID3D11Texture2D        *DepthTexture;
    ID3D11ShaderResourceView *DepthSRV;
    ID3D11DepthStencilView *DepthStencilView;

    ID3D11RasterizerState *MainRasterState;
    ID3D11DepthStencilState *YesDepthNoStencilState;
    ID3D11DepthStencilState *NoDepthNoStencilState;
    ID3D11BlendState *AlphaBlendState;

    ID3D11VertexShader *StaticLitVS;
    ID3D11PixelShader *StaticLitPS;
    ID3D11InputLayout *PNUV2_IL;
    ID3D11Buffer *StaticLitCB0;
    ID3D11Buffer *StaticLitCB1;

    ID3D11VertexShader *UIVS;
    ID3D11PixelShader *UIPS;
    ID3D11InputLayout *UIIL;
    ID3D11Buffer *UICB0;

    ID3D11VertexShader *WorldLineVS;
    ID3D11PixelShader *WorldLinePS;
    ID3D11InputLayout *WorldLineIL;
    ID3D11Buffer *WorldLineCB0;

    // TODO(Kevin): Create these once per material style and reuse
    ID3D11SamplerState *AnisotropicSampler;
    ID3D11SamplerState *PointSampler;

    dx11_mesh HandleToMesh[DX11_MESH_MAX_COUNT];
    mesh_handle FreeMeshHandles[DX11_MESH_MAX_COUNT];
    u32 FreeMeshHandlesCount;

    dx11_texture HandleToTexture[DX11_TEXTURE_MAX_COUNT];
    texture_handle FreeTextureHandles[DX11_TEXTURE_MAX_COUNT];
    u32 FreeTextureHandlesCount;

    dx11_structured_buffer PerInstanceDataSB;
    static_lit_model_instance *PerInstanceDataChunk;
    u32 PerInstanceDataCount;

    dx11_vertex_stream UIVertexStream;
    gui_vertex         GuiVertexStreamStaging[10000];
    u32                GuiVertexStreamStagingCount;
    gui_vstream_item   GuiVertexStreamItems[1024];
    u32                GuiVertexStreamItemsCount;

    dx11_vertex_stream  WorldLinesVertexStream;
    dx11_mesh           LineCornersVB;
    world_line_instance WorldLineInstances[40000];
    u32                 WorldLineInstancesCount;

    HWND Window;
};

void DX11GetWritablePerInstanceDataArray(dx11 *DX, static_lit_model_instance **OutArray, u32 **OutCount);
void DX11GetGuiVertexStreamFrame(dx11 *DX, vstream_frame_chunk *OutVertexStreamFrame, u32 **VertexStagingCount);
void DX11GetGuiVertexStreamItemsArray(dx11 *DX, gui_vstream_item **OutArray, u32 **OutCount);

dx11_mesh DX11UploadVertexBuffer(dx11 *DX, ID3D11InputLayout *IL, void *VB, u32 VertexStride, u32 VC);
mesh_handle DX11UploadMesh(dx11 *DX, ID3D11InputLayout *IL, void *VB, 
                           u32 VertexStride, u32 VC, u32 *IB, u32 IC);
mesh_handle DX11UploadMesh(dx11 *DX, PNUV2_vertex *VB, u32 VC, u32 *IB, u32 IC);
void DX11BindToInputAssembler(dx11 *DX, dx11_mesh Mesh);
void DX11DrawMesh(dx11 *DX, mesh_handle MeshHandle);
void DX11DrawMeshInstanced(dx11 *DX, mesh_handle MeshHandle, u32 Count);
void DX11DeleteMesh(dx11 *DX, mesh_handle MeshHandle);

texture_handle DX11UploadTexture2D(dx11 *DX, void *Bitmap, u32 W, u32 H, 
                                     pixel_format Format, bool32 LinearFiltering);
void DX11BindTexture2D(dx11 *DX, u32 Slot, texture_handle TextureHandle);
void DX11UpdateTexture2D(dx11 *DX, texture_handle TextureHandle, void *Bitmap, u32 W, u32 H);
void DX11DeleteTexture2D(dx11 *DX, texture_handle TextureHandle);

void DX11CreateVertexStream(dx11 *DX, dx11_vertex_stream *VStream, u32 SizeBytes);
u32 DX11VertexStreamAlloc(dx11 *DX, dx11_vertex_stream *VStream, u32 SizeBytes, void *Vertices);

#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);
void Win32BeginFrame(renderer *Renderer);
void Win32EndFrame(renderer *Renderer);
void Win32ReleaseRenderer(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_DX11_H
