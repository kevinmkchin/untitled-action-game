#include "shooter_renderer.h"
#include "shooter_renderer_dx11.h"
#include "shooter_gui.h"
#define VERTEXT_STATIC
#define VT_USING_STB_TRUETYPE
#define VERTEXT_IMPLEMENTATION
// #define STBTT_malloc(x,u)  ((void)(u),malloc(x))
// #define STBTT_free(x,u)    ((void)(u),free(x))
#include "shooter_vertext.h"

void renderer::GetRenderTargetDimensions(uint32 *Width, uint32 *Height)
{
    *Width = BackBufferW;
    *Height = BackBufferH;
}

void renderer::PushViewMatrix(mat4 InViewFromWorld)
{
    ViewFromWorld = InViewFromWorld;
}

void renderer::PushRenderItems(render_item *RenderItems, uint32 RenderItemsCount)
{
    // TODO(Kevin): Sort render items

    static_lit_model_instance *InstanceDataArray = 0;
    uint32 *InstanceDataArrayCount = 0;

    DX11GetWritablePerInstanceDataArray((dx11 *)this, &InstanceDataArray, &InstanceDataArrayCount);

    for (uint32 SortedIndex = 0; SortedIndex < RenderItemsCount; ++SortedIndex)
    {
        render_item *Item = &RenderItems[SortedIndex];

        mat4 ModelMatrix = TranslationMatrix(Item->WorldPosition) * 
                           RotationMatrix(Item->WorldRotation) * 
                           ScaleMatrix(Item->WorldScale.x, Item->WorldScale.y, Item->WorldScale.z);
        
        static_lit_model_instance *InstanceData = &InstanceDataArray[SortedIndex];

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
                if (Item->ForcedAmbient.X == 0 && 
                    Item->ForcedAmbient.Y == 0 &&
                    Item->ForcedAmbient.Z == 0)
                {
                    InstanceData->AmbientCube[Side] = vec4(1,1,1,0);
                }
                else
                {
                    // NOTE(Kevin): I can use the ambient cube to apply tint to things
                    InstanceData->AmbientCube[Side] = vec4(Item->ForcedAmbient,0);
                }
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
            else
            {
                // TODO(Kevin): later maybe use forced flat ambient term for no lightmap & no light cube
                InstanceData->DoSunLight = true;
            }
        }

        ++(*InstanceDataArrayCount);
    }
}

void renderer::PushGuiDrawRequests(memory_arena *Scratch, gui_draw_request *Requests, uint32 RequestsCount)
{
    vstream_frame_chunk VertexStreamFrame;
    u32 *VertexStagingCount;
    DX11GetGuiVertexStreamFrame((dx11 *)this, &VertexStreamFrame, &VertexStagingCount);
    gui_vstream_item *GuiStreamItems;
    u32 *GuiStreamItemsCount;
    DX11GetGuiVertexStreamItemsArray((dx11 *)this, &GuiStreamItems, &GuiStreamItemsCount);

    umm OffsetBytes = 0;
    for (uint32 Index = 0; Index < RequestsCount; ++Index)
    {
        gui_draw_request *Request = &Requests[Index];

        // TODO(Kevin): Write into VertexStreamFrame directly?
        uint32 GuiItemVerticesCount = 0;

        switch (Request->DrawType)
        {
            case GuiDrawRect:
            {
                gui_vertex *GuiVertices = (gui_vertex *)(((uint8 *)VertexStreamFrame.WriteTo) + OffsetBytes);
                GuiItemVerticesCount = 6;

                real32 Left = (real32)Request->Rect.X;
                real32 Top = (real32)Request->Rect.Y;
                real32 Bottom = (real32)Request->Rect.Y + Request->Rect.H;
                real32 Right = (real32)Request->Rect.X + Request->Rect.W;
                vec2 LeftTopUV = { 0.0f, 1.0f };
                vec2 LeftBottomUV = { 0.0f, 0.0f };
                vec2 RightTopUV = { 1.0f, 1.0f };
                vec2 RightBottomUV = { 1.0f, 0.0f };
                vec4 Color = Request->Color;
                gui_vertex GuiItemVertices[] = 
                {
                    { { Left, Top }, LeftTopUV, Color },
                    { { Left, Bottom }, LeftBottomUV, Color },
                    { { Right, Bottom }, RightBottomUV, Color },
                    { { Left, Top }, LeftTopUV, Color },
                    { { Right, Bottom }, RightBottomUV, Color },
                    { { Right, Top }, RightTopUV, Color },
                };

                memory_index GuiItemVerticesSizeInBytes = sizeof(gui_vertex)*GuiItemVerticesCount;
                if (OffsetBytes + GuiItemVerticesSizeInBytes > VertexStreamFrame.ChunkSizeBytes)
                {
                    Assert(0); // break;
                }
                Copy(GuiVertices, (void *)GuiItemVertices, GuiItemVerticesSizeInBytes);

                break;
            }
            case GuiDrawText:
            {
                gui_vertex *GuiVertices = (gui_vertex *)(((uint8 *)VertexStreamFrame.WriteTo) + OffsetBytes);
                Assert(sizeof(gui_vertex) == sizeof(vtxt_vertex1));

                // TODO(Kevin): If going to write directly into GuiVertices, make sure it is big enough

                vtxt_font *FontInfo = &(HandleToFont[Request->Font].FontInfo);
                vtxt_state TextState = {};
                TextState.VertexBuffer1 = (vtxt_vertex1 *)GuiVertices;
                TextState.VertexCount1 = 0;
                TextState.Font = FontInfo;
                TextState.CursorX = Request->X;
                TextState.CursorY = Request->Y;
                TextState.PutNewLinesAbove = Request->PutNewLinesAbove;

                vtxt_align Alignment = VT_Left;
                switch (Request->Align)
                {
                    case Align_Left: Alignment = VT_Left; break;
                    case Align_Center: Alignment = VT_Center; break;
                    case Align_Right: Alignment = VT_Right; break;
                }

                VT_AppendLine1(&TextState, Request->Text, Request->TextScale,
                               (float *)&Request->Color, Alignment);

                Assert(TextState.VertexCount1 > 0);

                GuiItemVerticesCount = TextState.VertexCount1;
                memory_index GuiItemVerticesSizeInBytes = sizeof(gui_vertex)*GuiItemVerticesCount;
                if (OffsetBytes + GuiItemVerticesSizeInBytes > VertexStreamFrame.ChunkSizeBytes)
                {
                    Assert(0); // break;
                }

                break;
            }
        }

        gui_vstream_item *GuiStreamItem = &GuiStreamItems[Index];

        *GuiStreamItem = {};
        GuiStreamItem->OffsetBytes = OffsetBytes;
        GuiStreamItem->VertexCount = GuiItemVerticesCount;
        GuiStreamItem->ScissorX = Request->Scissor.X;
        GuiStreamItem->ScissorY = Request->Scissor.Y;
        GuiStreamItem->ScissorW = Request->Scissor.W;
        GuiStreamItem->ScissorH = Request->Scissor.H;
        if (Request->DrawType == GuiDrawRect)
        {
            if (Request->Texture != InvalidHandle)
            {
                GuiStreamItem->ColorTexture = Request->Texture;
            }
            else
            {
                GuiStreamItem->ColorTexture = WhiteTexture_1x1;
            }
        }
        else if (Request->DrawType ==GuiDrawText)
        {
            GuiStreamItem->ColorTexture = HandleToFont[Request->Font].FontAtlasHandle;
        }
        ++(*GuiStreamItemsCount);
        *VertexStagingCount += GuiItemVerticesCount;

        if (*GuiStreamItemsCount >= ArrayCount(((dx11 *)0)->GuiVertexStreamItems))
        {
            break;
        }

        memory_index GuiItemVerticesSizeInBytes = sizeof(gui_vertex)*GuiItemVerticesCount;
        OffsetBytes += GuiItemVerticesSizeInBytes;
    }
}

void renderer::PushLine(vec3 WorldP0, vec3 WorldP1, vec3 Color, real32 PixelWidth)
{
    dx11 *DX = (dx11 *)this;
    if (DX->WorldLineInstancesCount < ArrayCount(DX->WorldLineInstances))
    {
        DX->WorldLineInstances[DX->WorldLineInstancesCount++] =
        {
            WorldP0, WorldP1, PixelWidth, vec4(Color, 1.0f)
        };
    }
}

mesh_handle renderer::UploadMeshData(PNUV2_vertex *VB, uint32 VC, uint32 *IB, uint32 IC)
{
    mesh_handle Result = InvalidHandle;
    Result = DX11UploadMesh((dx11 *)this, VB, VC, IB, IC);
    return Result;
}

void renderer::DeleteMesh(mesh_handle Mesh)
{
    DX11DeleteMesh((dx11 *)this, Mesh);
}

texture_handle renderer::UploadTexture2DData(void *Bitmap, uint32 W, uint32 H, 
                                             pixel_format Format, bool32 LinearFiltering)
{
    texture_handle Result = InvalidHandle;
    Result = DX11UploadTexture2D((dx11 *)this, Bitmap, W, H, Format, LinearFiltering);
    return Result;
}

create_font_result renderer::DEBUGCreateBitmapFont_Monospace(void *Bitmap, u32 W, u32 H,
                                                             pixel_format Format, u32 NewLineGap)
{
    create_font_result Result;

    u32 GlyphW = W / 16;
    u32 GlyphH = H / 16;

    Result.Handle = FontCount;
    renderer_font *Font = &HandleToFont[FontCount++];

    Font->FontAtlasHandle = UploadTexture2DData(Bitmap, W, H, Format, false);
    Result.FontAtlasHandle = Font->FontAtlasHandle;

    vtxt_font *VtxtFont = &(Font->FontInfo);
    VtxtFont->FontHeightPx = (float)GlyphH;
    VtxtFont->LineHeight = (float)(GlyphH + NewLineGap);
    Result.FontInfo = VtxtFont;

    for(u8 Codepoint = VT_UNICODE_RANGE_START;
        Codepoint <= VT_UNICODE_RANGE_END;
        ++Codepoint)
    {
        vtxt_glyph *Glyph = &(VtxtFont->Glyphs[Codepoint - VT_UNICODE_RANGE_START]);
        Glyph->Codepoint = Codepoint;
        Glyph->W = float(GlyphW);
        Glyph->H = float(GlyphH);
        Glyph->Advance = float(GlyphW);
        Glyph->OffsetX = 0;
        Glyph->OffsetY = -float(GlyphH);
        Glyph->MinU = float(Codepoint % 16 * GlyphW) / W;
        Glyph->MinV = 1.f - (float((Codepoint / 16 + 1) * GlyphH) / H);
        Glyph->MaxU = float((Codepoint % 16 + 1) * GlyphW) / W;
        Glyph->MaxV = 1.f - (float(Codepoint / 16 * GlyphH) / H);
    }

    return Result;
}

create_font_result renderer::DEBUGCreateBitmapFont_TrueType(thread_context *Thread, platform_api PlatformAPI,
                                                            s32 FontHeightPx, char *TrueTypeFileName)
{
    create_font_result Result = {};

    debug_read_file_result ReadResult = PlatformAPI.DEBUGPlatformReadEntireFile(Thread, TrueTypeFileName);

    if (ReadResult.ContentsSize)
    {
        vtxt_u8 *TrueTypeData = (vtxt_u8 *)ReadResult.Contents;

        // NOTE(Kevin): This is not an exact calculation of the atlas dimensions!
        real32 Side = SquareRoot((real32)CeilReal32ToPow2((real32)VT_UNICODE_RANGE+1));
        vtxt_u32 AtlasW = CeilReal32ToPow2(FontHeightPx*Side);
        vtxt_u32 AtlasH = CeilReal32ToPow2(FontHeightPx*Side);
        vtxt_u8 *AtlasPixelsMonoStart = (vtxt_u8 *)calloc(AtlasW*AtlasH, 1);

        Result.Handle = FontCount;
        renderer_font *Font = &HandleToFont[FontCount++];
        Result.FontInfo = &(Font->FontInfo);

        VT_CreateFontFromTTF(TrueTypeData, FontHeightPx, &(Font->FontInfo),
                             AtlasW, AtlasH, AtlasPixelsMonoStart);

        u32 *PixelsStart = (u32 *)calloc(AtlasW*AtlasH, sizeof(u32));
        u8 *AtlasPixelsMono = AtlasPixelsMonoStart;
        u32 *Pixels = PixelsStart;
        for (u32 Y = 0; Y < AtlasH; ++Y)
        {
            for (u32 X = 0; X < AtlasW; ++X)
            {
                u32 Alpha = (u32)(*AtlasPixelsMono);
                *Pixels = (Alpha << 24 |
                           Alpha << 16 |
                           Alpha << 8 |
                           Alpha);
                ++AtlasPixelsMono;
                ++Pixels;
            }
        }

        Font->FontAtlasHandle = UploadTexture2DData(PixelsStart, AtlasW, AtlasH, RGBA_U8, true);
        Result.FontAtlasHandle = Font->FontAtlasHandle;

        free(AtlasPixelsMonoStart);
        free(PixelsStart);
    }
    else
    {
        Assert(0);
    }

    PlatformAPI.DEBUGPlatformFreeFileMemory(Thread, ReadResult.Contents);

    return Result;
}
