#ifndef SHOOTER_RENDERER_DX11_H
#define SHOOTER_RENDERER_DX11_H

#include "shooter_renderer.h"
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

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

struct dx11
{
    renderer Header;

    D3D_FEATURE_LEVEL ObtainedFeatureLevel;
    DXGI_FORMAT       BackBufferFormat;
    bool32            AllowVRRTearing;
    // TODO(Kevin): These should be ->Release on shutdown
    IDXGISwapChain1 *SwapChain;
    ID3D11Device *Device;
    ID3D11DeviceContext *Context;
    IDXGIFactory2 *Factory2;

    ID3D11RenderTargetView *RenderTargetView;
    ID3D11Texture2D        *DepthTexture;
    ID3D11DepthStencilView *DepthStencilView;

    ID3D11RasterizerState *MainRasterState;

    ID3D11VertexShader *StaticLitVS;
    ID3D11PixelShader *StaticLitPS;
    ID3D11InputLayout *PNUV2_InputLayout;

    // TODO(Kevin): Create these once per material style and reuse
    ID3D11SamplerState *AnisotropicSampler;
    ID3D11SamplerState *PointSampler;

    dx11_mesh HandleToMesh[1024];
    mesh_handle FreeMeshHandles[1024];
    uint32 FreeMeshHandlesCount;

    dx11_texture HandleToTexture[512];
    mesh_handle FreeTextureHandles[512];
    uint32 FreeTextureHandlesCount;

    HWND Window;
};

mesh_handle DX11UploadMesh(dx11 *DX, PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC);
void DX11DrawMesh(dx11 *DX, mesh_handle MeshHandle);
void DX11DrawMeshInstanced(dx11 *DX, mesh_handle MeshHandle, uint32 Count);
void DX11DeleteMesh(dx11 *DX, mesh_handle MeshHandle);

texture_handle DX11UploadTexture2D(dx11 *DX, void *Bitmap, uint32 W, uint32 H, 
                                     pixel_format Format, bool32 LinearFiltering);
void DX11BindTexture2D(dx11 *DX, uint32 Slot, texture_handle TextureHandle);
void DX11UpdateTexture2D(dx11 *DX, texture_handle TextureHandle, void *Bitmap, uint32 W, uint32 H);
void DX11DeleteTexture2D(dx11 *DX, texture_handle TextureHandle);

#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);
void Win32BeginFrame(renderer *Renderer);
void Win32EndFrame(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_DX11_H
