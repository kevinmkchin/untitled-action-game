#include "shooter_renderer_dx11.h"

#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

global u32 PreviousBackBufferW;
global u32 PreviousBackBufferH;

#define DX11ReleaseCOM(ObjectPtr)\
    do {\
        if (ObjectPtr)\
        {\
            (ObjectPtr)->Release();\
            (ObjectPtr) = NULL;\
        }\
    } while (0);

#ifdef SHOOTER_PLATFORM_WIN32
internal void *
Win32RendererAlloc(umm Size)
{
    void *Result = VirtualAlloc(0, Size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    return Result;
}

internal void
DX11CreateDeviceAndSwapchain(dx11 *DX, HWND Window);
#endif

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
DX11CreateSizeDependentResources(dx11 *DX, UINT W, UINT H)
{
    DX->Context->OMSetRenderTargets(0, NULL, NULL);
    DX11ReleaseCOM(DX->RenderTargetView);
    DX11ReleaseCOM(DX->DepthStencilView);
    DX11ReleaseCOM(DX->DepthSRV);
    DX11ReleaseCOM(DX->DepthTexture);

    UINT SwapChainFlags = DX->AllowVRRTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    HRESULT HResult = DX->SwapChain->ResizeBuffers(0, W, H, DX->BackBufferFormat, SwapChainFlags);
    FailDx11HRESULT(HResult, "SwapChain->ResizeBuffers failed", DX->Window);

    ComPtr<ID3D11Texture2D> BackBufferTexture;
    HResult = DX->SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBufferTexture));
    FailDx11HRESULT(HResult, "GetBuffer(0) failed", DX->Window);
    HResult = DX->Device->CreateRenderTargetView(BackBufferTexture.Get(), NULL, &DX->RenderTargetView);
    FailDx11HRESULT(HResult, "Create RenderTargetView failed", DX->Window);

    D3D11_TEXTURE2D_DESC DepthStencilDesc = {};
    DepthStencilDesc.Width = W;
    DepthStencilDesc.Height = H;
    DepthStencilDesc.MipLevels = 1;
    DepthStencilDesc.ArraySize = 1;
    DepthStencilDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
    DepthStencilDesc.SampleDesc.Count = 1; // Valid range [0, ID3D11Device::CheckMultisampleQualityLevels) must match swapchain
    DepthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
    DepthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;

    D3D11_DEPTH_STENCIL_VIEW_DESC DSVDesc = {};
    DSVDesc.Format             = DXGI_FORMAT_D24_UNORM_S8_UINT;
    DSVDesc.ViewDimension      = D3D11_DSV_DIMENSION_TEXTURE2D;
    DSVDesc.Flags              = 0;
    DSVDesc.Texture2D.MipSlice = 0;

    D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
    SRVDesc.Format                    = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    SRVDesc.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
    SRVDesc.Texture2D.MostDetailedMip = 0;
    SRVDesc.Texture2D.MipLevels       = 1;

    HResult = DX->Device->CreateTexture2D(&DepthStencilDesc, NULL, &DX->DepthTexture);
    FailDx11HRESULT(HResult, "Create depth texture failed", DX->Window);
    HResult = DX->Device->CreateDepthStencilView(DX->DepthTexture, &DSVDesc, &DX->DepthStencilView);
    FailDx11HRESULT(HResult, "Create DepthStencilView failed", DX->Window);
    HResult = DX->Device->CreateShaderResourceView(DX->DepthTexture, &SRVDesc, &DX->DepthSRV);
    FailDx11HRESULT(HResult, "Create depth shader resource view failed", DX->Window);

    PreviousBackBufferW = W;
    PreviousBackBufferH = H;

    OutputDebugStringA("DX11: Resized swapchain and render target view.\n");
}

internal void
DX11CreateConstantBuffer(dx11 *DX, ID3D11Buffer **ConstantBuffer, u32 ByteSize)
{
    D3D11_BUFFER_DESC ConstantBufferDesc = {};
    ConstantBufferDesc.ByteWidth      = ByteSize;
    ConstantBufferDesc.Usage          = D3D11_USAGE_DYNAMIC;
    ConstantBufferDesc.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    ConstantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    HRESULT HR = DX->Device->CreateBuffer(&ConstantBufferDesc, NULL, ConstantBuffer);
    FailDx11HRESULT(HR, "Failed to create constant buffer!", DX->Window);
}

internal void
DX11UpdateConstantBuffer(dx11 *DX, ID3D11Buffer *ConstantBuffer, void *Data, u32 ByteSize,
                         bool32 PassToVS, u32 VSBufferSlot, bool32 PassToPS, u32 PSBufferSlot)
{
    D3D11_MAPPED_SUBRESOURCE MappedSubResource;
    DX->Context->Map(ConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource);
    Copy(MappedSubResource.pData, Data, ByteSize);
    DX->Context->Unmap(ConstantBuffer, 0);
    if (PassToVS)
    {
        DX->Context->VSSetConstantBuffers(VSBufferSlot, 1, &ConstantBuffer);
    }
    if (PassToPS)
    {
        DX->Context->PSSetConstantBuffers(PSBufferSlot, 1, &ConstantBuffer);
    }
}

internal void
DEBUGDX11CreateShaderSetFromFile(dx11 *DX, LPCWSTR VSFileName, ID3D11VertexShader **VS, 
                                 LPCWSTR PSFileName, ID3D11PixelShader **PS,
                                 D3D11_INPUT_ELEMENT_DESC *ILDesc, u32 ILDescCount,
                                 ID3D11InputLayout **IL)
{
    ComPtr<ID3DBlob> VSBlob, PSBlob, Err;

    HRESULT HR = D3DCompileFromFile(VSFileName, NULL, NULL, "main", "vs_5_0", 0, 0, &VSBlob, &Err);
    FailDx11HRESULT(HR, "Failed to compile from file vertex shader.", DX->Window);
    HR = D3DCompileFromFile(PSFileName, NULL, NULL, "main", "ps_5_0", 0, 0, &PSBlob, &Err);
    FailDx11HRESULT(HR, "Failed to compile from file pixel shader.", DX->Window);
    // TODO(Kevin): (char *)Err->GetBufferPointer()
    HR = DX->Device->CreateVertexShader(VSBlob->GetBufferPointer(), VSBlob->GetBufferSize(), NULL, VS);
    FailDx11HRESULT(HR, "Failed to create vertex shader.", DX->Window);
    HR = DX->Device->CreatePixelShader(PSBlob->GetBufferPointer(), PSBlob->GetBufferSize(), NULL, PS);
    FailDx11HRESULT(HR, "Failed to create pixel shader.", DX->Window);

    HR = DX->Device->CreateInputLayout(ILDesc, ILDescCount, VSBlob->GetBufferPointer(),
                                       VSBlob->GetBufferSize(), IL);
    FailDx11HRESULT(HR, "Failed to create Input Layout object.", DX->Window);
}

internal mesh_handle
UseNextAvailableMeshHandle(dx11 *DX)
{
    Assert(DX->FreeMeshHandlesCount > 0);
    mesh_handle FreeHandle = DX->FreeMeshHandles[--DX->FreeMeshHandlesCount];
    return FreeHandle;
}

internal texture_handle
UseNextAvailableTextureHandle(dx11 *DX)
{
    Assert(DX->FreeTextureHandlesCount > 0);
    texture_handle FreeHandle = DX->FreeTextureHandles[--DX->FreeTextureHandlesCount];
    return FreeHandle;
}

internal void
DX11Init(dx11 *DX)
{
    //
    // Rasterizer State and Depth Stencil State
    //
    D3D11_RASTERIZER_DESC RasterDesc = {};
    RasterDesc.FillMode = D3D11_FILL_SOLID;  // D3D11_FILL_SOLID or D3D11_FILL_WIREFRAME
    RasterDesc.CullMode = D3D11_CULL_BACK;   // D3D11_CULL_NONE, D3D11_CULL_FRONT, D3D11_CULL_BACK
    RasterDesc.FrontCounterClockwise = TRUE; // TRUE = CCW front-facing, FALSE = CW front-facing
    RasterDesc.DepthBias = 0;
    RasterDesc.DepthBiasClamp = 0.0f;
    RasterDesc.SlopeScaledDepthBias = 0.0f;
    RasterDesc.DepthClipEnable = TRUE;
    RasterDesc.ScissorEnable = TRUE;
    RasterDesc.MultisampleEnable = FALSE;
    RasterDesc.AntialiasedLineEnable = FALSE;

    HRESULT HR = DX->Device->CreateRasterizerState(&RasterDesc, &DX->MainRasterState);
    FailDx11HRESULT(HR, "Failed to CreateRasterizerState", DX->Window);

    D3D11_DEPTH_STENCIL_DESC DepthStencilStateDesc = {};
    DepthStencilStateDesc.DepthEnable = TRUE;
    DepthStencilStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    DepthStencilStateDesc.DepthFunc = D3D11_COMPARISON_LESS;
    DepthStencilStateDesc.StencilEnable = FALSE;

    DX->Device->CreateDepthStencilState(&DepthStencilStateDesc, &DX->YesDepthNoStencilState);

    DepthStencilStateDesc = {};
    DepthStencilStateDesc.DepthEnable = FALSE;
    DepthStencilStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;

    DX->Device->CreateDepthStencilState(&DepthStencilStateDesc, &DX->NoDepthNoStencilState);

    D3D11_RENDER_TARGET_BLEND_DESC RtBlendDesc = {};
    RtBlendDesc.BlendEnable = TRUE;
    RtBlendDesc.SrcBlend       = D3D11_BLEND_SRC_ALPHA;     // Src.rgb * Src.a
    RtBlendDesc.DestBlend      = D3D11_BLEND_INV_SRC_ALPHA; // Dest.rgb * (1 - Src.a)
    RtBlendDesc.BlendOp        = D3D11_BLEND_OP_ADD;
    RtBlendDesc.SrcBlendAlpha  = D3D11_BLEND_ONE;           // Src.a * 1
    RtBlendDesc.DestBlendAlpha = D3D11_BLEND_ZERO;          // Dest.a * 0
    RtBlendDesc.BlendOpAlpha   = D3D11_BLEND_OP_ADD;
    RtBlendDesc.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    D3D11_BLEND_DESC BlendDesc = {};
    BlendDesc.AlphaToCoverageEnable = FALSE;
    BlendDesc.IndependentBlendEnable = FALSE;
    BlendDesc.RenderTarget[0] = RtBlendDesc;

    DX->Device->CreateBlendState(&BlendDesc, &DX->AlphaBlendState);

    //
    // Shaders, Input Layouts, Constant Buffers
    //
    D3D11_INPUT_ELEMENT_DESC PNUV2_IL[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,   D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT,    0, 32,  D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    // TODO(Kevin): For shipping, use precompiled .cso blobs.
    DEBUGDX11CreateShaderSetFromFile(DX, L"../data/hlsl/vStaticLit.hlsl", &DX->StaticLitVS,
                                     L"../data/hlsl/pStaticLit.hlsl", &DX->StaticLitPS,
                                     PNUV2_IL, ArrayCount(PNUV2_IL), &DX->PNUV2_IL);
    DX11CreateConstantBuffer(DX, &DX->StaticLitCB0, sizeof(static_lit_cb0));
    DX11CreateConstantBuffer(DX, &DX->StaticLitCB1, sizeof(static_lit_cb1));

    D3D11_INPUT_ELEMENT_DESC UIIL[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 8,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    DEBUGDX11CreateShaderSetFromFile(DX, L"../data/hlsl/vUI.hlsl", &DX->UIVS,
                                     L"../data/hlsl/pUI.hlsl", &DX->UIPS,
                                     UIIL, ArrayCount(UIIL), &DX->UIIL);
    DX11CreateConstantBuffer(DX, &DX->UICB0, sizeof(ui_cb0));

    D3D11_INPUT_ELEMENT_DESC WorldLineIL[] =
    {
        {"CORNER_T_SIGN", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 0,  D3D11_INPUT_PER_VERTEX_DATA,   0},
        {"POSITION",      0, DXGI_FORMAT_R32G32B32_FLOAT,    1, 0,  D3D11_INPUT_PER_INSTANCE_DATA, 1},
        {"POSITION",      1, DXGI_FORMAT_R32G32B32_FLOAT,    1, 12, D3D11_INPUT_PER_INSTANCE_DATA, 1},
        {"THICKNESS",     0, DXGI_FORMAT_R32_FLOAT,          1, 24, D3D11_INPUT_PER_INSTANCE_DATA, 1},
        {"COLOR",         0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 28, D3D11_INPUT_PER_INSTANCE_DATA, 1},
    };
    DEBUGDX11CreateShaderSetFromFile(DX, L"../data/hlsl/vWorldLine.hlsl", &DX->WorldLineVS,
                                     L"../data/hlsl/pWorldLine.hlsl", &DX->WorldLinePS,
                                     WorldLineIL, ArrayCount(WorldLineIL), &DX->WorldLineIL);
    DX11CreateConstantBuffer(DX, &DX->WorldLineCB0, sizeof(world_line_cb0));

    //
    // World Lines Vertex Stream
    //
    real32 LineCorners_t_Sign[] =
    {
        0.0f, 1.f, 0.0f, -1.f, 1.0f, -1.f, // tri 0
        0.0f, 1.f, 1.0f, -1.f, 1.0f,  1.f, // tri 1
    };
    DX->LineCornersVB = DX11UploadVertexBuffer(DX, DX->WorldLineIL, LineCorners_t_Sign, sizeof(real32)*2, 6);
    DX11CreateVertexStream(DX, &DX->WorldLinesVertexStream, sizeof(DX->WorldLineInstances));

    //
    // Samplers
    //
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

    //
    // Per Instance Data Buffers
    //
    DX->Header.StaticLitItemsMaxCount = STATIC_LIT_ITEMS_MAX_COUNT;
    umm PerInstanceDataChunkByteSize = sizeof(static_lit_model_instance)*DX->Header.StaticLitItemsMaxCount;
    DX->PerInstanceDataChunk = (static_lit_model_instance *)Win32RendererAlloc(PerInstanceDataChunkByteSize);

    D3D11_BUFFER_DESC InstanceDataBufferDesc = {};
    InstanceDataBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    InstanceDataBufferDesc.ByteWidth = SafeTruncateU64(PerInstanceDataChunkByteSize);
    InstanceDataBufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    InstanceDataBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    InstanceDataBufferDesc.StructureByteStride = sizeof(static_lit_model_instance);

    HR = DX->Device->CreateBuffer(&InstanceDataBufferDesc, NULL, &DX->PerInstanceDataSB.Buffer);
    FailDx11HRESULT(HR, "Failed to create PerInstanceData buffer!", DX->Window);

    D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
    SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    SRVDesc.Format = DXGI_FORMAT_UNKNOWN;
    SRVDesc.Buffer.FirstElement = 0;
    SRVDesc.Buffer.NumElements  = DX->Header.StaticLitItemsMaxCount;

    HR = DX->Device->CreateShaderResourceView(DX->PerInstanceDataSB.Buffer, &SRVDesc, &DX->PerInstanceDataSB.SRV);
    FailDx11HRESULT(HR, "Failed to create PerInstanceData shader resource view!", DX->Window);

    //
    // UI Vertex Stream
    //
    u32 WorstCaseUIBytesPerFrame = sizeof(DX->GuiVertexStreamStaging);
    u32 UIVertexStreamSizeBytes = WorstCaseUIBytesPerFrame * (DX->FramesInFlight + 1);
    DX11CreateVertexStream(DX, &DX->UIVertexStream, UIVertexStreamSizeBytes);

    //
    // Handle values
    //
    // TODO(Kevin): Upload "default" mesh for the 0 handle
    DX->FreeMeshHandlesCount = ArrayCount(DX->FreeMeshHandles);
    PopulateFreeHandles(DX->FreeMeshHandles, DX->FreeMeshHandlesCount);
    UseNextAvailableMeshHandle(DX); // NOTE(Kevin): 0 is reserved

    DX->FreeTextureHandlesCount = ArrayCount(DX->FreeTextureHandles);
    PopulateFreeHandles(DX->FreeTextureHandles, DX->FreeTextureHandlesCount);
}

internal void
DX11BeginFrame(dx11 *DX)
{
    DX->PerInstanceDataCount = 0;
    DX->GuiVertexStreamStagingCount = 0;
    DX->GuiVertexStreamItemsCount = 0;
    DX->WorldLineInstancesCount = 0;

    if (PreviousBackBufferW != DX->Header.BackBufferW ||
        PreviousBackBufferH != DX->Header.BackBufferH)
    {
        if (DX->Header.BackBufferW > 0 && DX->Header.BackBufferH > 0)
        {
            DX11CreateSizeDependentResources(DX, DX->Header.BackBufferW, DX->Header.BackBufferH);
        }
    }
}

internal void
DX11EndFrame(dx11 *DX)
{
    vec4 ClearColor = { 0.10f, 0.18f, 0.30f, 1.0f };
    DX->Context->ClearRenderTargetView(DX->RenderTargetView, (float *)&ClearColor);
    DX->Context->ClearDepthStencilView(DX->DepthStencilView, D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL, 1.0f, 0);

    // Bind and set viewport
    DX->Context->OMSetRenderTargets(1, &DX->RenderTargetView, DX->DepthStencilView);

    D3D11_VIEWPORT MainViewport = {};
    MainViewport.TopLeftX = 0.0f;
    MainViewport.TopLeftY = 0.0f;
    MainViewport.Width = (real32)PreviousBackBufferW;
    MainViewport.Height = (real32)PreviousBackBufferH;
    MainViewport.MinDepth = 0.0f;
    MainViewport.MaxDepth = 1.0f;
    DX->Context->RSSetViewports(1, &MainViewport);

    //
    // StaticLit
    //
    real32 AspectRatio = real32(PreviousBackBufferW) / real32(PreviousBackBufferH);
    real32 FOVX = 90.f;
    real32 FOVYRad = HorizontalFOVToVerticalFOV(FOVX*GMATH_DEG2RAD, AspectRatio);
    mat4 ClipFromView = ClipSpaceFromCameraSpace_PerspectiveProjection(FOVYRad, AspectRatio, 4.f, 3200.f, true);

    mat4 ProjFromWorld = ClipFromView * DX->Header.ViewFromWorld;
    vec3 DirectionToSun = Normalize(vec3(-1,2,0.5f));

    DX->Context->RSSetState(DX->MainRasterState);
    DX->Context->OMSetDepthStencilState(DX->YesDepthNoStencilState, 0);
    DX->Context->OMSetBlendState(DX->AlphaBlendState, NULL, 0xFFFFFFFF);

    DX->Context->VSSetShader(DX->StaticLitVS, NULL, 0);
    DX->Context->PSSetShader(DX->StaticLitPS, NULL, 0);

    if (DX->PerInstanceDataCount > 0)
    {
        static_lit_model_instance *PerInstanceData = DX->PerInstanceDataChunk;

        DX->Context->UpdateSubresource(DX->PerInstanceDataSB.Buffer, 0, NULL, PerInstanceData, 0, 0);
        u32 StructuredBufferSlot = 0;
        DX->Context->VSSetShaderResources(StructuredBufferSlot, 1, &DX->PerInstanceDataSB.SRV);
        DX->Context->PSSetShaderResources(StructuredBufferSlot, 1, &DX->PerInstanceDataSB.SRV);

        uint32 CurrentDescriptorCount = 0;
        uint32 CurrentOffsetFromBase = 0;
        render_descriptor CurrentDescriptor = PerInstanceData[0].Descriptor;

        static_lit_cb0 DrawParams0 =
        {
            ProjFromWorld,
            DirectionToSun,
        };
        DX11UpdateConstantBuffer(DX, DX->StaticLitCB0, &DrawParams0, sizeof(DrawParams0), true, 0, true, 0);

        for (uint32 PerInstanceDataIndex = 0;
             PerInstanceDataIndex <= DX->PerInstanceDataCount; 
             ++PerInstanceDataIndex)
        {
            if (PerInstanceDataIndex == DX->PerInstanceDataCount ||
                PerInstanceData[PerInstanceDataIndex].Descriptor != CurrentDescriptor)
            {
                static_lit_cb1 DrawParams1 =
                {
                    CurrentOffsetFromBase
                };
                DX11UpdateConstantBuffer(DX, DX->StaticLitCB1, &DrawParams1, sizeof(DrawParams1), true, 1, false, 0);

                DX11BindTexture2D(DX, 0, CurrentDescriptor.Albedo);
                DX11BindTexture2D(DX, 1, CurrentDescriptor.Lightmap);
                DX11DrawMeshInstanced(DX, CurrentDescriptor.Mesh, CurrentDescriptorCount);

                if (PerInstanceDataIndex == DX->PerInstanceDataCount)
                {
                    break;
                }

                CurrentOffsetFromBase += CurrentDescriptorCount;
                CurrentDescriptor = PerInstanceData[PerInstanceDataIndex].Descriptor;
                CurrentDescriptorCount = 0;
            }

            ++CurrentDescriptorCount;
        }
    }


    //
    // World Line Rendering
    //
    // NOTE(Kevin): To sample depth texture, must unbind depth stencil view from render target
    // DX->Context->OMSetRenderTargets(1, &DX->RenderTargetView, NULL);
    // DX->Context->OMSetDepthStencilState(DX->NoDepthNoStencilState, 0);
    DX->Context->VSSetShader(DX->WorldLineVS, NULL, 0);
    DX->Context->PSSetShader(DX->WorldLinePS, NULL, 0);

    DX->WorldLinesVertexStream.Cursor = 0;
    DX11VertexStreamAlloc(DX, &DX->WorldLinesVertexStream, DX->WorldLineInstancesCount*
                          sizeof(world_line_instance), DX->WorldLineInstances);

    world_line_cb0 WorldLineDrawParams =
    {
        ProjFromWorld,
        { (real32)PreviousBackBufferW, (real32)PreviousBackBufferH },
        1// 0.990f
    };
    DX11UpdateConstantBuffer(DX, DX->WorldLineCB0, &WorldLineDrawParams, sizeof(WorldLineDrawParams), true, 0, true, 0);

    // DX->Context->PSSetShaderResources(8, 1, &DX->DepthSRV);
    // DX->Context->PSSetSamplers(0, 1, &DX->PointSampler);

    DX->Context->IASetInputLayout(DX->LineCornersVB.Layout);
    ID3D11Buffer *WorldLineBufs[] = { DX->LineCornersVB.VB, DX->WorldLinesVertexStream.Buffer };
    UINT WorldLineBufStrides[] = { DX->LineCornersVB.VertexStride, sizeof(world_line_instance) };
    UINT WorldLineBufOffsets[] = { 0, 0 };
    DX->Context->IASetVertexBuffers(0, 2, WorldLineBufs, WorldLineBufStrides, WorldLineBufOffsets);
    DX->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    DX->Context->DrawInstanced(6, DX->WorldLineInstancesCount, 0, 0);


    //
    // UI
    //

    DX->Context->RSSetState(DX->MainRasterState);
    DX->Context->OMSetDepthStencilState(DX->NoDepthNoStencilState, 0);
    DX->Context->OMSetBlendState(DX->AlphaBlendState, NULL, 0xFFFFFFFF);

    DX->Context->VSSetShader(DX->UIVS, NULL, 0);
    DX->Context->PSSetShader(DX->UIPS, NULL, 0);

    gui_vertex *UIVertices = DX->GuiVertexStreamStaging;
    u32 UIVertexCount = DX->GuiVertexStreamStagingCount;
    u32 UIVertexStride = sizeof(gui_vertex);
    u32 UIVerticesSize = UIVertexStride * UIVertexCount;
    u32 VStreamOffset = DX11VertexStreamAlloc(DX, &DX->UIVertexStream, UIVerticesSize, UIVertices);
    u32 VStreamStartIndex = VStreamOffset / UIVertexStride;

    dx11_mesh MeshInfo = {};
    MeshInfo.VB = DX->UIVertexStream.Buffer;
    MeshInfo.VertexStride = UIVertexStride;
    MeshInfo.Layout = DX->UIIL;
    DX11BindToInputAssembler(DX, MeshInfo);

    mat4 ClipFromScreen2D = ProjectionMatrixOrthographicNoZ(0.f, (real32)PreviousBackBufferW, 
                                                            (real32)PreviousBackBufferH, 0.f);

    ui_cb0 UIDrawParams =
    {
        ClipFromScreen2D
    };
    DX11UpdateConstantBuffer(DX, DX->UICB0, &UIDrawParams, sizeof(UIDrawParams), true, 0, false, 0);

    u32 PrevScissorX = 0xFFFFFFFF;
    u32 PrevScissorY = 0xFFFFFFFF;
    u32 PrevScissorW = 0xFFFFFFFF;
    u32 PrevScissorH = 0xFFFFFFFF;
    for (uint32 GuiItemIndex = 0; 
         GuiItemIndex < DX->GuiVertexStreamItemsCount; 
         ++GuiItemIndex)
    {
        gui_vstream_item Item = DX->GuiVertexStreamItems[GuiItemIndex];
        u32 ScissorX = Item.ScissorX;
        u32 ScissorY = Item.ScissorY;
        u32 ScissorW = Item.ScissorW;
        u32 ScissorH = Item.ScissorH;
        if (ScissorX != PrevScissorX || ScissorY != PrevScissorY ||
            ScissorW != PrevScissorW || ScissorH != PrevScissorH)
        {
            D3D11_RECT Scissor;
            Scissor.left = ScissorX;
            Scissor.top = ScissorY;
            Scissor.right = ScissorX + ScissorW;
            Scissor.bottom = ScissorY + ScissorH;
            DX->Context->RSSetScissorRects(1, &Scissor);
            PrevScissorX = ScissorX;
            PrevScissorY = ScissorY;
            PrevScissorW = ScissorW;
            PrevScissorH = ScissorH;
        }

        u32 LocalVertexCount = Item.VertexCount;
        u32 LocalIndexOffset = SafeTruncateU64(Item.OffsetBytes) / UIVertexStride;
        DX11BindTexture2D(DX, 0, Item.ColorTexture);
        DX->Context->Draw(LocalVertexCount, VStreamStartIndex + LocalIndexOffset);
    }

    D3D11_RECT Scissor;
    Scissor.left = 0;
    Scissor.top = 0;
    Scissor.right = PreviousBackBufferW;
    Scissor.bottom = PreviousBackBufferH;
    DX->Context->RSSetScissorRects(1, &Scissor);

    //
    // Presentation
    //

#if SHOOTER_DEBUG_VSYNC_OFF
    bool32 VsyncEnabled = false;
#else
    bool32 VsyncEnabled = true;
#endif
    UINT SyncInterval = VsyncEnabled ? 1 : 0;
    UINT PresentFlags = 0;
    if (!VsyncEnabled && DX->AllowVRRTearing)
    {
        PresentFlags |= DXGI_PRESENT_ALLOW_TEARING;
    }

    HRESULT HResult = DX->SwapChain->Present(SyncInterval, PresentFlags);
    if (HResult == DXGI_ERROR_DEVICE_REMOVED || HResult == DXGI_ERROR_DEVICE_RESET)
    {
        // NOTE(Kevin): This is if a video card has been physically removed from the computer,
        // or a driver upgrade for the video card has occurred.
        FailDx11HRESULT(HResult, "DX11 device removed or reset! Attempting to recreate.", DX->Window, false);

        Win32ReleaseRenderer((renderer *)DX);

        DX11CreateDeviceAndSwapchain(DX, DX->Window);
        DX11CreateSizeDependentResources(DX, DX->Header.BackBufferW, DX->Header.BackBufferH);
        DX11Init(DX);
    }
    else
    {
        FailDx11HRESULT(HResult, "Present failed!", DX->Window);
    }
}

void DX11GetWritablePerInstanceDataArray(dx11 *DX, static_lit_model_instance **OutArray, u32 **OutCount)
{
    // NOTE(Kevin): The out array is offset by the data count as to not overwrite the existing data.
    *OutArray = ((static_lit_model_instance *)DX->PerInstanceDataChunk) + DX->PerInstanceDataCount;
    *OutCount = &DX->PerInstanceDataCount;
}

void DX11GetGuiVertexStreamFrame(dx11 *DX, vstream_frame_chunk *OutVertexStreamFrame, u32 **VertexStagingCount)
{
    vstream_frame_chunk Result;
    Result.WriteTo = DX->GuiVertexStreamStaging;
    Result.ChunkSizeBytes = sizeof(DX->GuiVertexStreamStaging);
    *OutVertexStreamFrame = Result;
    *VertexStagingCount = &DX->GuiVertexStreamStagingCount;
}

void DX11GetGuiVertexStreamItemsArray(dx11 *DX, gui_vstream_item **OutArray, u32 **OutCount)
{
    *OutArray = DX->GuiVertexStreamItems;
    *OutCount = &DX->GuiVertexStreamItemsCount;
}

dx11_mesh
DX11UploadVertexBuffer(dx11 *DX, ID3D11InputLayout *IL, void *VB, 
                       u32 VertexStride, u32 VC)
{
    dx11_mesh Mesh = {};

    Mesh.VertexStride = VertexStride;
    Mesh.Layout = IL;

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

    return Mesh;
}

mesh_handle
DX11UploadMesh(dx11 *DX, ID3D11InputLayout *IL, void *VB, 
               u32 VertexStride, u32 VC, u32 *IB, u32 IC)
{
    dx11_mesh Mesh = DX11UploadVertexBuffer(DX, IL, VB, VertexStride, VC);

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
    HRESULT HR = DX->Device->CreateBuffer(&IBDesc, &IBInitData, &Mesh.IB);
    FailDx11HRESULT(HR, "Failed to create index buffer.", DX->Window);

    Mesh.IndicesCount = IC;

    mesh_handle NewHandle = UseNextAvailableMeshHandle(DX);
    DX->HandleToMesh[NewHandle] = Mesh;

    return NewHandle;
}

mesh_handle
DX11UploadMesh(dx11 *DX, PNUV2_vertex *VB, u32 VC, u32 *IB, u32 IC)
{
    mesh_handle Result = DX11UploadMesh(DX, DX->PNUV2_IL, VB, sizeof(PNUV2_vertex), VC, IB, IC);
    return Result;
}

void
DX11BindToInputAssembler(dx11 *DX, dx11_mesh Mesh)
{
    DX->Context->IASetInputLayout(Mesh.Layout);
    UINT StartSlot = 0;
    UINT NumBuffers = 1;
    UINT Strides[] = { Mesh.VertexStride };
    UINT Offsets[] = { 0 };
    DX->Context->IASetVertexBuffers(StartSlot, NumBuffers, &Mesh.VB, Strides, Offsets);
    if (Mesh.IB)
    {
        UINT IBOffset = 0;
        DX->Context->IASetIndexBuffer(Mesh.IB, DXGI_FORMAT_R32_UINT, IBOffset);
    }
    DX->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void
DX11DrawMesh(dx11 *DX, mesh_handle MeshHandle)
{
    dx11_mesh Mesh = DX->HandleToMesh[MeshHandle];

    Assert(Mesh.IndicesCount > 0);

    DX11BindToInputAssembler(DX, Mesh);

    UINT StartIndexLocation = 0;
    INT BaseVertexLocation = 0;
    DX->Context->DrawIndexed(Mesh.IndicesCount, StartIndexLocation, BaseVertexLocation);
}

void
DX11DrawMeshInstanced(dx11 *DX, mesh_handle MeshHandle, u32 Count)
{
    dx11_mesh Mesh = DX->HandleToMesh[MeshHandle];

    Assert(Mesh.IndicesCount > 0);

    DX11BindToInputAssembler(DX, Mesh);

    UINT StartIndexLocation = 0;
    INT BaseVertexLocation = 0;
    DX->Context->DrawIndexedInstanced(Mesh.IndicesCount, Count, StartIndexLocation, BaseVertexLocation, 0);
}

void
DX11DeleteMesh(dx11 *DX, mesh_handle MeshHandle)
{
    dx11_mesh Mesh = DX->HandleToMesh[MeshHandle];

    DX11ReleaseCOM(Mesh.IB);
    DX11ReleaseCOM(Mesh.VB);
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
    HRESULT HR = DX->Device->CreateTexture2D(&TexDesc, NULL, &Texture.Texture);
    FailDx11HRESULT(HR, "Failed to CreateTexture2D.", DX->Window);
    DX->Context->UpdateSubresource(Texture.Texture, 0, NULL, Bitmap, SrcRowPitchBytes, 0);

    D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
    SRVDesc.Format = TexDesc.Format;
    SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_tex1d_srv
    SRVDesc.Texture2D.MipLevels = (UINT)-1; // -1 = all mipmap levels

    DX->Device->CreateShaderResourceView(Texture.Texture, &SRVDesc, &Texture.SRV);
    FailDx11HRESULT(HR, "Failed to CreateShaderResourceView for Texture2D.", DX->Window);
    DX->Context->GenerateMips(Texture.SRV);
    FailDx11HRESULT(HR, "Failed to generate mips.", DX->Window);

    if (LinearFiltering)
    {
        Texture.Sampler = DX->AnisotropicSampler;
    }
    else
    {
        Texture.Sampler = DX->PointSampler;
    }

    texture_handle NewHandle = UseNextAvailableTextureHandle(DX);
    DX->HandleToTexture[NewHandle] = Texture;

    return NewHandle;
}

void DX11BindTexture2D(dx11 *DX, u32 Slot, texture_handle TextureHandle)
{
    dx11_texture Texture = DX->HandleToTexture[TextureHandle];

    //
    // NOTE(Kevin): First 8 texture SRV slots are reserved for StructuredBuffers
    //
    u32 TextureSRVSlotStart = 8;

    // SRVs & samplers to PS slots
    ID3D11ShaderResourceView *SRVs[] =
    {
        Texture.SRV
    };
    ID3D11SamplerState *Samplers[] = 
    {
        Texture.Sampler
    };

    DX->Context->PSSetShaderResources(TextureSRVSlotStart+Slot, ArrayCount(SRVs), SRVs);
    DX->Context->PSSetSamplers(Slot, ArrayCount(Samplers), Samplers);
}

void DX11UpdateTexture2D(dx11 *DX, texture_handle TextureHandle, void *Bitmap, u32 W, u32 H)
{
    // TODO(Kevin): Implement
    Assert(0);
}

void DX11DeleteTexture2D(dx11 *DX, texture_handle TextureHandle)
{
    dx11_texture Texture = DX->HandleToTexture[TextureHandle];

    DX11ReleaseCOM(Texture.Texture);
    DX11ReleaseCOM(Texture.SRV);

    DX->FreeTextureHandles[DX->FreeTextureHandlesCount++] = TextureHandle;
}

void DX11CreateVertexStream(dx11 *DX, dx11_vertex_stream *VStream, u32 SizeBytes)
{
    D3D11_BUFFER_DESC BufferDesc = {};
    BufferDesc.ByteWidth = SizeBytes;
    BufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    HRESULT HR = DX->Device->CreateBuffer(&BufferDesc, nullptr, &VStream->Buffer);
    FailDx11HRESULT(HR, "Failed to create vertex stream.", DX->Window);

    VStream->Capacity = SizeBytes;
    VStream->Cursor = 0;
}

u32 DX11VertexStreamAlloc(dx11 *DX, dx11_vertex_stream *VStream, u32 SizeBytes, void *Vertices)
{
    Assert(SizeBytes < VStream->Capacity);

    D3D11_MAPPED_SUBRESOURCE MappedSubRes = {};

    if (VStream->Cursor == 0 || VStream->Cursor + SizeBytes > VStream->Capacity)
    {
        // MAP_WRITE_DISCARD: write into new allocation while the GPU is still reading older ones
        DX->Context->Map(VStream->Buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubRes);
        VStream->Cursor = 0;
    }
    else
    {
        // MAP_WRITE_NO_OVERWRITE: append into unused space of the same allocation
        DX->Context->Map(VStream->Buffer, 0, D3D11_MAP_WRITE_NO_OVERWRITE, 0, &MappedSubRes);
    }

    u32 Offset = VStream->Cursor;

    Copy((uint8_t*)MappedSubRes.pData + Offset, Vertices, SizeBytes);
    DX->Context->Unmap(VStream->Buffer, 0);

    VStream->Cursor += SizeBytes;

    return Offset;
}

#ifdef SHOOTER_PLATFORM_WIN32

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

    DX->FramesInFlight = 1;
    ComPtr<IDXGIDevice1> DXGIDevice;
    HResult = DX->Device->QueryInterface(__uuidof(IDXGIDevice1), (void**)&DXGIDevice);
    if (SUCCEEDED(HResult))
    {
        FailDx11HRESULT(HResult, "Failed to set maximum frame latency.", DX->Window, false);
        HResult = DXGIDevice->SetMaximumFrameLatency(DX->FramesInFlight);
        FailDx11HRESULT(HResult, "Failed to set maximum frame latency.", DX->Window, false);
        if (FAILED(HResult))
        {
            DX->FramesInFlight = 3;
        }
    }
    else
    {
        DX->FramesInFlight = 3;
    }

    // Get DXGI factory
    ComPtr<IDXGIAdapter> DXGIAdapter;
    HResult = DXGIDevice->GetAdapter(&DXGIAdapter);
    FailDx11HRESULT(HResult, "Failed to get IDXGIAdapter from IDXGIDevice1.", Window);
    HResult = DXGIAdapter->GetParent(IID_PPV_ARGS(&DX->Factory2));
    FailDx11HRESULT(HResult, "Failed to get IDXGIFactory2 from IDXGIAdapter.", Window);

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

    DX->BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

    DXGI_SWAP_CHAIN_DESC1 SwapChainDesc = {};
    SwapChainDesc.Width = 1;
    SwapChainDesc.Height = 1;
    SwapChainDesc.Format = DX->BackBufferFormat;
    SwapChainDesc.Stereo = FALSE;
    SwapChainDesc.SampleDesc.Count = 1; // Valid range [0, ID3D11Device::CheckMultisampleQualityLevels)
    SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    SwapChainDesc.BufferCount = 2;
    SwapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    SwapChainDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    SwapChainDesc.Flags = DX->AllowVRRTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    // Create swap chain for HWND
    HResult = DX->Factory2->CreateSwapChainForHwnd(DX->Device,
                                                   Window,
                                                   &SwapChainDesc,
                                                   NULL /*DXGI_SWAP_CHAIN_FULLSCREEN_DESC*/,
                                                   NULL /*restrictToOutput*/,
                                                   &DX->SwapChain);
    FailDx11HRESULT(HResult, "CreateSwapChainForHwnd failed", Window);

    DX->Factory2->MakeWindowAssociation(Window, DXGI_MWA_NO_ALT_ENTER);
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
    DX11CreateSizeDependentResources(DX, Width, Height);
    DX11Init(DX);

    return (renderer *)DX;
}

void
Win32BeginFrame(renderer *Renderer)
{
    DX11BeginFrame((dx11 *)Renderer);
}

void
Win32EndFrame(renderer *Renderer)
{
    DX11EndFrame((dx11 *)Renderer);
}

void
Win32ReleaseRenderer(renderer *Renderer)
{
    dx11 *DX = (dx11 *)Renderer;

    DX->Context->ClearState();
    DX->Context->Flush();

    for (u32 Mesh = 0; Mesh < ArrayCount(DX->HandleToMesh); ++Mesh)
    {
        DX11DeleteMesh(DX, Mesh);
    }
    for (u32 Texture = 0; Texture < ArrayCount(DX->HandleToTexture); ++Texture)
    {
       DX11DeleteTexture2D(DX, Texture);
    }

    DX11ReleaseCOM(DX->PerInstanceDataSB.Buffer);
    DX11ReleaseCOM(DX->PerInstanceDataSB.SRV);

    DX11ReleaseCOM(DX->MainRasterState);
    DX11ReleaseCOM(DX->YesDepthNoStencilState);
    DX11ReleaseCOM(DX->NoDepthNoStencilState);
    DX11ReleaseCOM(DX->AlphaBlendState);

    DX11ReleaseCOM(DX->StaticLitVS);
    DX11ReleaseCOM(DX->StaticLitPS);
    DX11ReleaseCOM(DX->PNUV2_IL);
    DX11ReleaseCOM(DX->StaticLitCB0);
    DX11ReleaseCOM(DX->StaticLitCB1);

    DX11ReleaseCOM(DX->UIVS);
    DX11ReleaseCOM(DX->UIPS);
    DX11ReleaseCOM(DX->UIIL);
    DX11ReleaseCOM(DX->UICB0);

    DX11ReleaseCOM(DX->AnisotropicSampler);
    DX11ReleaseCOM(DX->PointSampler);

    DX11ReleaseCOM(DX->RenderTargetView);
    DX11ReleaseCOM(DX->DepthSRV);
    DX11ReleaseCOM(DX->DepthTexture);
    DX11ReleaseCOM(DX->DepthStencilView);

    // TODO(Kevin): DX->SwapChain->SetFullscreenState

    DX11ReleaseCOM(DX->Factory2);
    DX11ReleaseCOM(DX->SwapChain);
    DX11ReleaseCOM(DX->Context);
#if HANDMADE_INTERNAL
    ID3D11Debug *DebugLayer;
    if (SUCCEEDED(DX->Device->QueryInterface(__uuidof(ID3D11Debug), (void**)&DebugLayer)))
    {
        DebugLayer->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL | D3D11_RLDO_IGNORE_INTERNAL);
        DebugLayer->Release();
    }
#endif
    DX11ReleaseCOM(DX->Device);
}

#endif // SHOOTER_PLATFORM_WIN32