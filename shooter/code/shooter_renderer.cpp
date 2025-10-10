#include "shooter_renderer.h"
#include "shooter_renderer_opengl.h"
#include "shooter_gui.h"
#define VERTEXT_IMPLEMENTATION
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

    open_gl_model_instance *InstanceDataArray = 0;
    uint32 *InstanceDataArrayCount = 0;
    OpenGLGetPerInstanceDataArray((open_gl *)this, &InstanceDataArray, &InstanceDataArrayCount);

    for (uint32 SortedIndex = 0; SortedIndex < RenderItemsCount; ++SortedIndex)
    {
        render_item *Item = &RenderItems[SortedIndex];

        mat4 ModelMatrix = TranslationMatrix(Item->WorldPosition) * 
                           RotationMatrix(Item->WorldRotation) * 
                           ScaleMatrix(Item->WorldScale.x, Item->WorldScale.y, Item->WorldScale.z);
        
        open_gl_model_instance *InstanceData = &InstanceDataArray[SortedIndex];

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
                // TODO(Kevin): When everything is going through StaticLit pipeline, I could
                // use the ambient cube to apply tint to things, etc.
                InstanceData->AmbientCube[Side] = 1.0f;
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
        }

        ++(*InstanceDataArrayCount);
    }
}

void renderer::PushGuiDrawRequests(memory_arena *Scratch, gui_draw_request *Requests, uint32 RequestsCount)
{
    vstream_frame_chunk VertexStreamFrame;
    OpenGLGetGuiVertexStreamFrame((open_gl *)this, &VertexStreamFrame);
    gui_vstream_item *GuiStreamItems;
    uint32 *GuiStreamItemsCount;
    OpenGLGetGuiVertexStreamItemsArray((open_gl *)this, &GuiStreamItems, &GuiStreamItemsCount);

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

                vtxt_font *FontInfo = &(HandleToFont[Request->Font].FontInfo);
                u32 Flags = 0;
                if (Request->PutNewLinesAbove)
                {
                    Flags |= VTXT_NEWLINE_ABOVE;
                }
                vtxt_setflags(Flags);
                vtxt_clear_buffer();
                vtxt_move_cursor(Request->X, Request->Y);
                switch (Request->Align)
                {
                    case Align_Left:
                    {
                        vtxt_append_line(Request->Text, FontInfo, Request->FontSize, Request->Color);
                        break;
                    }
                    case Align_Center:
                    {
                        vtxt_append_line_centered(Request->Text, FontInfo, Request->FontSize, Request->Color);
                        break;
                    }
                    case Align_Right:
                    {
                        vtxt_append_line_align_right(Request->Text, FontInfo, Request->FontSize, Request->Color);
                        break;
                    }
                }
                vtxt_vertex_buffer TextVertices = vtxt_grab_buffer();
                Assert(TextVertices.vertices_array_count > 0);

                GuiItemVerticesCount = TextVertices.vertex_count;
                memory_index GuiItemVerticesSizeInBytes = sizeof(gui_vertex)*GuiItemVerticesCount;
                if (OffsetBytes + GuiItemVerticesSizeInBytes > VertexStreamFrame.ChunkSizeBytes)
                {
                    Assert(0); // break;
                }
                Copy(GuiVertices, TextVertices.vertex_buffer, GuiItemVerticesSizeInBytes);

                vtxt_clear_buffer();

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

        if (*GuiStreamItemsCount >= ArrayCount(((open_gl *)0)->GuiVertexStreamItems))
        {
            break;
        }

        memory_index GuiItemVerticesSizeInBytes = sizeof(gui_vertex)*GuiItemVerticesCount;
        OffsetBytes += GuiItemVerticesSizeInBytes;
    }
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

font_handle renderer::CreateBitmapFont_ExtendedASCII(void *Bitmap, u32 W, u32 H, pixel_format Format, u32 NewLineGap)
{    
    u32 GlyphW = W / 16;
    u32 GlyphH = H / 16;

    font_handle Result = FontCount;
    renderer_font *Font = &HandleToFont[FontCount++];

    Font->FontAtlasHandle = UploadTexture2DData(Bitmap, W, H, Format, false);

    vtxt_font *VtxtFont = &(Font->FontInfo);
    VtxtFont->font_height_px = GlyphH;
    VtxtFont->ascender = float(GlyphH);
    VtxtFont->descender = 0;
    VtxtFont->linegap = (float)NewLineGap;

    for(int CodePoint = 0; CodePoint < 256; ++CodePoint)
    {
        vtxt_glyph *Glyph = &(VtxtFont->glyphs[CodePoint]);
        Glyph->codepoint = (unsigned char)CodePoint;
        Glyph->width = float(GlyphW);
        Glyph->height = float(GlyphH);
        Glyph->advance = float(GlyphW);
        Glyph->offset_x = 0;
        Glyph->offset_y = -float(GlyphH);
        Glyph->min_u = float(CodePoint % 16 * GlyphW) / W;
        Glyph->min_v = 1.f - (float((CodePoint / 16 + 1) * GlyphH) / H);
        Glyph->max_u = float((CodePoint % 16 + 1) * GlyphW) / W;
        Glyph->max_v = 1.f - (float(CodePoint / 16 * GlyphH) / H);
    }

    return Result;
}

