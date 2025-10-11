#ifndef SHOOTER_RENDERER_DX11_H
#define SHOOTER_RENDERER_DX11_H

#include "shooter_renderer.h"
#include <d3d11.h>
#include <dxgi1_6.h>

struct dx11_mesh
{

};

struct dx11_texture
{

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

    dx11_mesh HandleToMesh[1024];
    mesh_handle FreeMeshHandles[1024];
    uint32 FreeMeshHandlesCount;

    dx11_texture HandleToTexture[512];
    mesh_handle FreeTextureHandles[512];
    uint32 FreeTextureHandlesCount;

    HWND Window;
};

#ifdef SHOOTER_PLATFORM_WIN32

renderer *Win32LoadRenderer(HDC WindowDC);
void Win32BeginFrame(renderer *Renderer);
void Win32EndFrame(renderer *Renderer);

#endif // SHOOTER_PLATFORM_WIN32

#endif // SHOOTER_RENDERER_DX11_H
