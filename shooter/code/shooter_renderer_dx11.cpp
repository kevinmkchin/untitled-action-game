#include "shooter_renderer_dx11.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

internal void
FailDx11HRESULT(HRESULT HResult, char *Message, HWND Window)
{
    if (FAILED(HResult))
    {
        char Buf[512];
        snprintf(Buf, sizeof(Buf), "%s (HResult=0x%08X)\n", Message, (unsigned)HResult);
        MessageBoxA(Window, Buf, "D3D11 Error", MB_ICONERROR | MB_OK);
        Assert(0);
    }
}

internal void
DX11Init(dx11 *DX)
{
    DX->FreeMeshHandlesCount = ArrayCount(DX->FreeMeshHandles);
    for (s32 FreeHandle = 0;
         FreeHandle < SafeCastToS32(DX->FreeMeshHandlesCount);
         ++FreeHandle)
    {
        DX->FreeMeshHandles[DX->FreeMeshHandlesCount - FreeHandle - 1] = 
            (mesh_handle)SafeCastToU32(FreeHandle);
    }

    DX->FreeTextureHandlesCount = ArrayCount(DX->FreeTextureHandles);
    for (s32 FreeHandle = 0; 
         FreeHandle < SafeCastToS32(DX->FreeTextureHandlesCount);
         ++FreeHandle)
    {
        DX->FreeTextureHandles[DX->FreeTextureHandlesCount - FreeHandle - 1] = 
            (texture_handle)SafeCastToU32(FreeHandle);
    }
}

internal void
DX11EndFrame(dx11 *DX)
{
    vec4 ClearColor = { 0.10f, 0.18f, 0.30f, 1.0f };
    DX->Context->ClearRenderTargetView(DX->RenderTargetView, (float *)&ClearColor);
    DX->Context->ClearDepthStencilView(DX->DepthStencilView, D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL, 1.0f, 0);



    bool32 VsyncEnabled = true;
    UINT SyncInterval = VsyncEnabled ? 1 : 0;
    UINT PresentFlags = 0;
    if (!VsyncEnabled && DX->AllowVRRTearing)
    {
        PresentFlags |= DXGI_PRESENT_ALLOW_TEARING;
    }

    HRESULT HResult = DX->SwapChain->Present(SyncInterval, PresentFlags);
    if (HResult == DXGI_ERROR_DEVICE_REMOVED || HResult == DXGI_ERROR_DEVICE_RESET)
    {
        // TODO(Kevin): Device lost — recreate device and size-dependent resources
        Assert(0);
        // g_rtv.Reset(); g_dsv.Reset(); g_ds.Reset(); DX->SwapChain.Reset();
        // g_context.Reset(); g_device.Reset(); g_factory2.Reset();
        // DX11CreateDeviceAndSwapchain();
        // DX11CreateSizeDependentResources(g_width, g_height);
    }
    else
    {
        FailDx11HRESULT(HResult, "Present failed", DX->Window);
    }
}

#ifdef SHOOTER_PLATFORM_WIN32

internal void *
Win32RendererAlloc(umm Size)
{
    void *Result = VirtualAlloc(0, Size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    return Result;
}

internal void
DX11CreateDeviceAndSwapchain(dx11 *DX, HWND Window)
{
    UINT CreateFlags = 0;
#if HANDMADE_INTERNAL
    CreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };

    DX->ObtainedFeatureLevel = {};

    HRESULT HResult = D3D11CreateDevice(NULL,
                                        D3D_DRIVER_TYPE_HARDWARE,
                                        NULL,
                                        CreateFlags,
                                        FeatureLevels,
                                        ArrayCount(FeatureLevels),
                                        D3D11_SDK_VERSION,
                                        &DX->Device,
                                        &DX->ObtainedFeatureLevel,
                                        &DX->Context);
    FailDx11HRESULT(HResult, "D3D11CreateDevice failed", Window);

    // Create DXGI factory
    UINT FactoryFlags = 0;
#if HANDMADE_INTERNAL
    FactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
    HResult = CreateDXGIFactory2(FactoryFlags, IID_PPV_ARGS(&DX->Factory2));
    FailDx11HRESULT(HResult, "CreateDXGIFactory2 failed", Window);
    Assert(DX->Factory2);

    // Check tearing support
    DX->AllowVRRTearing = false;
    IDXGIFactory5 *Factory5;
    HResult = DX->Factory2->QueryInterface(__uuidof(IDXGIFactory5), reinterpret_cast<void**>(&Factory5));
    if (SUCCEEDED(HResult))
    {
        BOOL AllowVRRTearing = FALSE;
        if (SUCCEEDED(Factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &AllowVRRTearing, sizeof(AllowVRRTearing))))
        {
            DX->AllowVRRTearing = AllowVRRTearing == TRUE;
        }
        Factory5->Release();
    }

    DX->BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

    DXGI_SWAP_CHAIN_DESC1 SwapChainDesc = {};
    SwapChainDesc.Width = 1;
    SwapChainDesc.Height = 1;
    SwapChainDesc.Format = DX->BackBufferFormat;
    SwapChainDesc.Stereo = FALSE;
    SwapChainDesc.SampleDesc.Count = 1;
    SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    SwapChainDesc.BufferCount = 2;
    SwapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    SwapChainDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    SwapChainDesc.Flags = DX->AllowVRRTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    // TODO(Kevin): DXGI_SWAP_CHAIN_FULLSCREEN_DESC ?

    // Create swap chain for HWND
    HResult = DX->Factory2->CreateSwapChainForHwnd(DX->Device,
                                                   Window,
                                                   &SwapChainDesc,
                                                   NULL /*fullscreen desc*/,
                                                   NULL /*restrictToOutput*/,
                                                   &DX->SwapChain);
    FailDx11HRESULT(HResult, "CreateSwapChainForHwnd failed", Window);

    // Make-windowed full-screen transitions use DXGI
    DX->Factory2->MakeWindowAssociation(Window, DXGI_MWA_NO_ALT_ENTER);
}

internal void
DX11CreateSizeDependentResources(dx11 *DX, UINT W, UINT H, HWND Window)
{
    DX->Context->OMSetRenderTargets(0, NULL, NULL);
    if (DX->RenderTargetView)
    {
        DX->RenderTargetView->Release();
        DX->RenderTargetView = NULL;
    }
    if (DX->DepthStencilView)
    {
        DX->DepthStencilView->Release();
        DX->DepthStencilView = NULL;
    }
    if (DX->DepthTexture)
    {
        DX->DepthTexture->Release();
        DX->DepthTexture = NULL;
    }

    // Resize swapchain buffers (0 keeps current count/format)
    UINT SwapChainFlags = DX->AllowVRRTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    HRESULT HResult = DX->SwapChain->ResizeBuffers(0, W, H, DX->BackBufferFormat, SwapChainFlags);
    FailDx11HRESULT(HResult, "ResizeBuffers failed", Window);

    // Acquire backbuffer and create RTV
    ID3D11Texture2D *BackBufferTexture;
    HResult = DX->SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBufferTexture));
    FailDx11HRESULT(HResult, "GetBuffer(0) failed", Window);
    HResult = DX->Device->CreateRenderTargetView(BackBufferTexture, NULL, &DX->RenderTargetView);
    FailDx11HRESULT(HResult, "Create RenderTargetView failed", Window);

    // Create depth-stencil (24/8)
    D3D11_TEXTURE2D_DESC DepthStencilDesc = {};
    DepthStencilDesc.Width = W;
    DepthStencilDesc.Height = H;
    DepthStencilDesc.MipLevels = 1;
    DepthStencilDesc.ArraySize = 1;
    DepthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    DepthStencilDesc.SampleDesc.Count = 1;
    DepthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
    DepthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    HResult = DX->Device->CreateTexture2D(&DepthStencilDesc, NULL, &DX->DepthTexture);
    FailDx11HRESULT(HResult, "Create depth texture failed", Window);
    HResult = DX->Device->CreateDepthStencilView(DX->DepthTexture, NULL, &DX->DepthStencilView);
    FailDx11HRESULT(HResult, "Create DepthStencilView failed", Window);

    // Bind and set viewport
    ID3D11RenderTargetView *RenderTargetViews[] =
    {
        DX->RenderTargetView
    };
    DX->Context->OMSetRenderTargets(ArrayCount(RenderTargetViews), RenderTargetViews, DX->DepthStencilView);

    D3D11_VIEWPORT Viewport = {};
    Viewport.TopLeftX = 0.0f;
    Viewport.TopLeftY = 0.0f;
    Viewport.Width = (real32)W;
    Viewport.Height = (real32)H;
    Viewport.MinDepth = 0.0f;
    Viewport.MaxDepth = 1.0f;
    DX->Context->RSSetViewports(1, &Viewport);
}

renderer *
Win32LoadRenderer(HDC WindowDC)
{
    dx11 *DX = (dx11 *)Win32RendererAlloc(sizeof(dx11));
    ConstructRendererServiceAt((&(DX->Header)));

    DX->Window = WindowFromDC(WindowDC);
    RECT ClientRect;
    GetClientRect(DX->Window, &ClientRect);
    UINT Width = ClientRect.right - ClientRect.left;
    UINT Height = ClientRect.bottom - ClientRect.top;

    DX11CreateDeviceAndSwapchain(DX, DX->Window);
    DX11CreateSizeDependentResources(DX, Width, Height, DX->Window);
    DX11Init(DX);

    return (renderer *)DX;
}

void
Win32BeginFrame(renderer *Renderer)
{
    // OpenGLBeginFrame((open_gl *)Renderer);
}

void
Win32EndFrame(renderer *Renderer)
{
    DX11EndFrame((dx11 *)Renderer);
}

#endif // SHOOTER_PLATFORM_WIN32