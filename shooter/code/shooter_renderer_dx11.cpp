#include "shooter_renderer_dx11.h"

#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")


internal void
FailDx11HRESULT(HRESULT HResult, char *Message, HWND Window, bool32 Fatal = true)
{
    if (FAILED(HResult))
    {
        char Buf[512];
        snprintf(Buf, sizeof(Buf), "%s (HRESULT=0x%08X)\n", Message, (unsigned)HResult);
        MessageBox(Window, Buf, "D3D11 Error", MB_ICONERROR | MB_OK);
        Assert(!Fatal);
    }
}

internal void
DX11Init(dx11 *DX)
{
    D3D11_RASTERIZER_DESC RasterDesc = {};
    RasterDesc.FillMode = D3D11_FILL_SOLID;  // D3D11_FILL_SOLID or D3D11_FILL_WIREFRAME
    RasterDesc.CullMode = D3D11_CULL_BACK;   // D3D11_CULL_NONE, D3D11_CULL_FRONT, D3D11_CULL_BACK
    RasterDesc.FrontCounterClockwise = TRUE; // TRUE = CCW front-facing, FALSE = CW front-facing
    RasterDesc.DepthBias = 0;
    RasterDesc.DepthBiasClamp = 0.0f;
    RasterDesc.SlopeScaledDepthBias = 0.0f;
    RasterDesc.DepthClipEnable = TRUE;
    RasterDesc.ScissorEnable = FALSE;
    RasterDesc.MultisampleEnable = FALSE;
    RasterDesc.AntialiasedLineEnable = FALSE;

    HRESULT HR = DX->Device->CreateRasterizerState(&RasterDesc, &DX->MainRasterState);
    FailDx11HRESULT(HR, "Failed to CreateRasterizerState", DX->Window);


    ComPtr<ID3DBlob> VSBlob, PSBlob, Err;

    // TODO(Kevin): For shipping, use precompiled .cso blobs.
    HR = D3DCompileFromFile(L"../data/hlsl/vStaticLit.hlsl", NULL, NULL, "main", "vs_5_0", 0, 0, &VSBlob, &Err);
    FailDx11HRESULT(HR, "Failed to compile vStaticLit.hlsl", DX->Window);
    HR = D3DCompileFromFile(L"../data/hlsl/pStaticLit.hlsl", NULL, NULL, "main", "ps_5_0", 0, 0, &PSBlob, &Err);
    FailDx11HRESULT(HR, "Failed to compile pStaticLit.hlsl", DX->Window);
    // TODO(Kevin): Check Err

    HR = DX->Device->CreateVertexShader(VSBlob->GetBufferPointer(), VSBlob->GetBufferSize(), NULL, 
                                        &DX->StaticLitVS);
    FailDx11HRESULT(HR, "Failed to create vertex shader", DX->Window);
    HR = DX->Device->CreatePixelShader(PSBlob->GetBufferPointer(), PSBlob->GetBufferSize(), NULL,
                                       &DX->StaticLitPS);
    FailDx11HRESULT(HR, "Failed to create pixel shader", DX->Window);


    D3D11_INPUT_ELEMENT_DESC PNUV2_InputLayout[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,   D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT,    0, 32,  D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    DX->Device->CreateInputLayout(PNUV2_InputLayout, ArrayCount(PNUV2_InputLayout),
                                  VSBlob->GetBufferPointer(), VSBlob->GetBufferSize(), 
                                  &DX->PNUV2_InputLayout);


    D3D11_SAMPLER_DESC Desc = {};
    Desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    Desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    Desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    Desc.MipLODBias = 0.0f;
    Desc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    Desc.MinLOD = 0.0f;
    Desc.MaxLOD = D3D11_FLOAT32_MAX;
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_filter
    Desc.Filter = D3D11_FILTER_ANISOTROPIC;
    Desc.MaxAnisotropy = 8; // TODO(Kevin): Make this adjustable [1, 16]
    DX->Device->CreateSamplerState(&Desc, &DX->AnisotropicSampler);
    Desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    DX->Device->CreateSamplerState(&Desc, &DX->PointSampler);


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

    // Bind and set viewport
    DX->Context->RSSetState(DX->MainRasterState);

    ID3D11RenderTargetView *RenderTargetViews[] =
    {
        DX->RenderTargetView
    };
    DX->Context->OMSetRenderTargets(ArrayCount(RenderTargetViews), RenderTargetViews, DX->DepthStencilView);

    D3D11_VIEWPORT Viewport = {};
    Viewport.TopLeftX = 0.0f;
    Viewport.TopLeftY = 0.0f;
    Viewport.Width = (real32)DX->Header.BackBufferW;
    Viewport.Height = (real32)DX->Header.BackBufferH;
    Viewport.MinDepth = 0.0f;
    Viewport.MaxDepth = 1.0f;
    DX->Context->RSSetViewports(1, &Viewport);

    real32 AspectRatio = real32(DX->Header.BackBufferW) / real32(DX->Header.BackBufferH);
    real32 FOVX = 90.f;
    real32 FOVYRad = HorizontalFOVToVerticalFOV(FOVX*GM_DEG2RAD, AspectRatio);
    // TODO(Kevin): Z has to be 0 to 1
    mat4 ClipFromView = ProjectionMatrixPerspective(FOVYRad, AspectRatio, 4.f, 3200.f);

    //
    //
    //

    // Constant buffer: must be multiple of 16 bytes
    struct alignas(16) camera_cb
    {
        mat4 ProjView;
    };
    local_persist ID3D11Buffer *CameraCB = NULL;
    if (!CameraCB)
    {
        D3D11_BUFFER_DESC ConstantBufferDesc = {};
        ConstantBufferDesc.ByteWidth      = sizeof(camera_cb);
        ConstantBufferDesc.Usage          = D3D11_USAGE_DYNAMIC;
        ConstantBufferDesc.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        ConstantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        HRESULT HR = DX->Device->CreateBuffer(&ConstantBufferDesc, NULL, &CameraCB);
        FailDx11HRESULT(HR, "Failed to create constant buffer", DX->Window);
    }

    // Update per-frame (Map/Unmap with WRITE_DISCARD)
    camera_cb CBData = { ClipFromView * DX->Header.ViewFromWorld };
    D3D11_MAPPED_SUBRESOURCE MappedSubResource;
    DX->Context->Map(CameraCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource);
    Copy(MappedSubResource.pData, &CBData, sizeof(CBData));
    DX->Context->Unmap(CameraCB, 0);

    // Shaders
    DX->Context->VSSetShader(DX->StaticLitVS, NULL, 0);
    DX->Context->PSSetShader(DX->StaticLitPS, NULL, 0);

    // Constant buffers to VS slot b0
    ID3D11Buffer *ConstantBuffers[] =
    {
        CameraCB
    };
    DX->Context->VSSetConstantBuffers(0, ArrayCount(ConstantBuffers), ConstantBuffers);

    texture_handle Albedo = 2;
    texture_handle Lightmap = DX->Header.WhiteTexture_1x1;
    DX11BindTexture2D(DX, 0, Albedo);
    DX11BindTexture2D(DX, 1, Lightmap);
    DX11DrawMesh(DX, 0);

    //
    //
    //

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
        FailDx11HRESULT(HResult, "Present failed!", DX->Window);
    }
}

internal mesh_handle
NextAvailableMeshHandle(dx11 *DX)
{
    Assert(DX->FreeMeshHandlesCount > 0);
    mesh_handle FreeHandle = DX->FreeMeshHandles[--DX->FreeMeshHandlesCount];
    return FreeHandle;
}

internal texture_handle
NextAvailableTextureHandle(dx11 *DX)
{
    Assert(DX->FreeTextureHandlesCount > 0);
    texture_handle FreeHandle = DX->FreeTextureHandles[--DX->FreeTextureHandlesCount];
    return FreeHandle;
}

mesh_handle
DX11UploadMesh(dx11 *DX, PNUV2_vertex *VB, u32 VC, u32 *IB, u32 IC)
{
    dx11_mesh Mesh = {};

    Mesh.VertexStride = sizeof(PNUV2_vertex);
    Mesh.Layout = DX->PNUV2_InputLayout;

    D3D11_BUFFER_DESC VBDesc;
    VBDesc.Usage            = D3D11_USAGE_DEFAULT;
    VBDesc.ByteWidth        = Mesh.VertexStride * VC;
    VBDesc.BindFlags        = D3D11_BIND_VERTEX_BUFFER;
    VBDesc.CPUAccessFlags   = 0;
    VBDesc.MiscFlags        = 0;

    D3D11_SUBRESOURCE_DATA VBInitData;
    VBInitData.pSysMem = VB;
    VBInitData.SysMemPitch = 0;
    VBInitData.SysMemSlicePitch = 0;

    HRESULT HR = DX->Device->CreateBuffer(&VBDesc, &VBInitData, &Mesh.VB);
    FailDx11HRESULT(HR, "Failed to create vertex buffer.", DX->Window);

    D3D11_BUFFER_DESC IBDesc;
    IBDesc.Usage           = D3D11_USAGE_DEFAULT;
    IBDesc.ByteWidth       = sizeof(u32) * IC;
    IBDesc.BindFlags       = D3D11_BIND_INDEX_BUFFER;
    IBDesc.CPUAccessFlags  = 0;
    IBDesc.MiscFlags       = 0;

    D3D11_SUBRESOURCE_DATA IBInitData;
    IBInitData.pSysMem = IB;
    IBInitData.SysMemPitch = 0;
    IBInitData.SysMemSlicePitch = 0;

    // Create the buffer with the device.
    HR = DX->Device->CreateBuffer(&IBDesc, &IBInitData, &Mesh.IB);
    FailDx11HRESULT(HR, "Failed to create index buffer.", DX->Window);

    Mesh.IndicesCount = IC;

    mesh_handle NewHandle = NextAvailableMeshHandle(DX);
    DX->HandleToMesh[NewHandle] = Mesh;

    return NewHandle;
}

void
DX11DrawMesh(dx11 *DX, mesh_handle MeshHandle)
{
    dx11_mesh Mesh = DX->HandleToMesh[MeshHandle];

    Assert(Mesh.IndicesCount > 0);

    // IA stage
    DX->Context->IASetInputLayout(Mesh.Layout);
    UINT StartSlot = 0;
    UINT NumBuffers = 1;
    UINT Strides[] = { Mesh.VertexStride };
    UINT Offsets[] = { 0 };
    DX->Context->IASetVertexBuffers(StartSlot, NumBuffers, &Mesh.VB, Strides, Offsets);
    UINT IBOffset = 0;
    DX->Context->IASetIndexBuffer(Mesh.IB, DXGI_FORMAT_R32_UINT, IBOffset);
    DX->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Minimal fixed-function state (rasterizer/depth/blend use defaults if not set)
    UINT StartIndexLocation = 0;
    INT BaseVertexLocation = 0;
    DX->Context->DrawIndexed(Mesh.IndicesCount, StartIndexLocation, BaseVertexLocation);
}

void
DX11DrawMeshInstanced(dx11 *DX, mesh_handle MeshHandle, u32 Count)
{

}

void
DX11DeleteMesh(dx11 *DX, mesh_handle MeshHandle)
{
    dx11_mesh Mesh = DX->HandleToMesh[MeshHandle];

    if (Mesh.IB != 0)
    {
        Mesh.IB->Release();
        Mesh.IB = 0;
    }
    if (Mesh.VB != 0)
    {
        Mesh.VB->Release();
        Mesh.VB = 0;
    }
    Mesh.IndicesCount = 0;

    DX->FreeMeshHandles[DX->FreeMeshHandlesCount++] = MeshHandle;
}

texture_handle DX11UploadTexture2D(dx11 *DX, void *Bitmap, u32 W, u32 H, 
                                   pixel_format Format, bool32 LinearFiltering)
{
    dx11_texture Texture = {};

    u32 SrcRowPitchBytes = 0;
    // TODO(Kevin): SRGB!!!
    switch (Format)
    {
    case R_U8:
        // NOTE(Kevin): In DX shader, the unused color channels are default values when sampling.
        Texture.Format = DXGI_FORMAT_R8_UNORM;
        SrcRowPitchBytes = W;
        break;
    case R_Float:
        Texture.Format = DXGI_FORMAT_R32_FLOAT;
        SrcRowPitchBytes = W*sizeof(float);
        break;
    default:
    case RGBA_U8:
        Texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        SrcRowPitchBytes = W*4;
        break;
    case RGBA_Float:
        Texture.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        SrcRowPitchBytes = W*4*sizeof(float);
        break;
    case BGRA_U8:
        Texture.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        SrcRowPitchBytes = W*4;
        break;
    case BGRA_Float:
        // NOTE(Kevin): Unsupported
        Assert(0);
        break;
    }

    DXGI_SAMPLE_DESC MultisamplingParams;
    MultisamplingParams.Count = 1;
    MultisamplingParams.Quality = 0; // Valid range [0, ID3D11Device::CheckMultisampleQualityLevels)

    // TODO(Kevin): Mip levels?
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_texture2d_desc
    D3D11_TEXTURE2D_DESC TexDesc = {};
    TexDesc.Width = W;
    TexDesc.Height = H;
    TexDesc.MipLevels = 0; // 0 = generate full set of subtextures, 1 = for multisampled texture
    TexDesc.ArraySize = 1;
    TexDesc.Format = Texture.Format;
    TexDesc.SampleDesc = MultisamplingParams;
    TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    TexDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

    // TODO(Kevin): Check HRESULTs
    DX->Device->CreateTexture2D(&TexDesc, NULL, &Texture.Texture);
    DX->Context->UpdateSubresource(Texture.Texture, 0, NULL, Bitmap, SrcRowPitchBytes, 0);

    D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
    SRVDesc.Format = TexDesc.Format;
    SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_tex1d_srv
    SRVDesc.Texture2D.MipLevels = (UINT)-1; // -1 = all mipmap levels

    DX->Device->CreateShaderResourceView(Texture.Texture, &SRVDesc, &Texture.SRV);
    DX->Context->GenerateMips(Texture.SRV);

    if (LinearFiltering)
    {
        Texture.Sampler = DX->AnisotropicSampler;
    }
    else
    {
        Texture.Sampler = DX->PointSampler;
    }

    texture_handle NewHandle = NextAvailableTextureHandle(DX);
    DX->HandleToTexture[NewHandle] = Texture;

    return NewHandle;
}

void DX11BindTexture2D(dx11 *DX, u32 Slot, texture_handle TextureHandle)
{
    dx11_texture Texture = DX->HandleToTexture[TextureHandle];

    // SRVs & samplers to PS slots
    ID3D11ShaderResourceView *SRVs[] =
    {
        Texture.SRV
    };
    ID3D11SamplerState *Samplers[] = 
    {
        Texture.Sampler
    };

    DX->Context->PSSetShaderResources(Slot, ArrayCount(SRVs), SRVs);
    DX->Context->PSSetSamplers(Slot, ArrayCount(Samplers), Samplers);
}

void DX11UpdateTexture2D(dx11 *DX, texture_handle TextureHandle, void *Bitmap, u32 W, u32 H)
{
    // TODO(Kevin): Implement
    Assert(0);
}

void DX11DeleteTexture2D(dx11 *DX, texture_handle TextureHandle)
{
    // TODO(Kevin): Implement
    Assert(0);
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
    HResult = DX->Factory2->QueryInterface(__uuidof(IDXGIFactory5), (void**)&Factory5);
    if (SUCCEEDED(HResult))
    {
        BOOL AllowVRRTearing = FALSE;
        if (SUCCEEDED(Factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &AllowVRRTearing, sizeof(AllowVRRTearing))))
        {
            DX->AllowVRRTearing = AllowVRRTearing == TRUE;
        }
        Factory5->Release();
    }

    u32 FramesInFlight = 1;
    IDXGIDevice1 *dxgiDevice;
    HResult = DX->Device->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgiDevice);
    if (SUCCEEDED(HResult))
    {
        FailDx11HRESULT(HResult, "Failed to set maximum frame latency.", DX->Window, false);
        HResult = dxgiDevice->SetMaximumFrameLatency(FramesInFlight);
        FailDx11HRESULT(HResult, "Failed to set maximum frame latency.", DX->Window, false);
        dxgiDevice->Release();
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