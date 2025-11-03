/*

    vertext.h

    Text mesh generation library (https://kevin.gd/)
    Generates meshes composed of quads for text rendering

    Build a vtxt_font manually or use the helper function

*/
#ifndef INCLUDE_VERTEXT_H
#define INCLUDE_VERTEXT_H

#ifndef VT_UNICODE_RANGE_START
#define VT_UNICODE_RANGE_START 32 // starting ASCII codepoint to collect glyph data for
#endif
#ifndef VT_UNICODE_RANGE_END
#define VT_UNICODE_RANGE_END 126 // ending ASCII codepoint to collect glyph data for
#endif
#define VT_UNICODE_RANGE (VT_UNICODE_RANGE_END - VT_UNICODE_RANGE_START + 1)

#ifdef VERTEXT_STATIC
#define VTXT_DEF static
#else
#define VTXT_DEF extern
#endif

typedef int           vtxt_s32;
typedef unsigned int  vtxt_u32;
typedef unsigned char vtxt_u8;

struct vtxt_vertex1
{
    float X, Y;
    float U, V;
    float R, G, B, A;
};

struct vtxt_glyph
{
    vtxt_u8 Codepoint;
    float W, H, Advance, OffsetX, OffsetY, MinU, MinV, MaxU, MaxV;
};

struct vtxt_font
{
    vtxt_s32   FontHeightPx;
    float      LineHeight; // Ascender + -Descender + Linegap
    vtxt_glyph Glyphs[VT_UNICODE_RANGE];
};

struct vtxt_state
{
    vtxt_vertex1 *VertexBuffer1;
    vtxt_u32      VertexCount1;
    vtxt_font    *Font;
    vtxt_s32      CursorX, CursorY; // TODO(Kevin): Should the cursor be in floats?
    // [optional]
    bool PutNewLinesAbove;
    bool FlipY;
    // bool          CreateIndexBuffer;
    // vtxt_u32     *IndexBuffer;
    // vtxt_u32      IndexCount;
    // bool          GetNormalizedDeviceCoords;
    // vtxt_u32      RenderW, RenderH;
};

enum vtxt_align
{
    VT_Left,
    VT_Center,
    VT_Right,
};

VTXT_DEF void
VT_AppendLine1(vtxt_state *State, char *Text, float Scale,
              float *ColorRGBA, vtxt_align Align = VT_Left);

VTXT_DEF void
VT_AppendGlyph1(vtxt_state *State, char GlyphIndex, float Scale, float OffsetX, float *ColorRGBA);

VTXT_DEF void
VT_GetTextBoundingBoxDimensions(vtxt_font *Font, char *Text, float Scale,
                                float *OutWidth, float *OutHeight, vtxt_s32 Length = 0);
VTXT_DEF void
VT_NewLine(vtxt_state *State, float Scale, vtxt_s32 X);


#ifdef VT_USING_STB_TRUETYPE

VTXT_DEF void
VT_CreateFontFromTTF(vtxt_u8 *TrueTypeData, float FontHeightPx, vtxt_font *OutFont,
                     vtxt_s32 AtlasW, vtxt_s32 AtlasH, vtxt_u8 *OutAtlasPixels);

#endif // VT_USING_STB_TRUETYPE

#endif // INCLUDE_VERTEXT_H


#ifdef VERTEXT_IMPLEMENTATION

VTXT_DEF void
VT_AppendGlyph1(vtxt_state *State, char GlyphIndex, float Scale, float OffsetX, float *ColorRGBA)
{
    if(GlyphIndex < VT_UNICODE_RANGE_START || GlyphIndex > VT_UNICODE_RANGE_END)
    {
        return;
    }

    // TODO(Kevin): Check if can fit in vertex buffer?
    // if(VTXT_MAX_CHAR_IN_BUFFER * 6 < _vtxt_vertex_count + 6)
    // {
    //     return;
    // }

    // TODO(Kevin): Should everything be in floats to avoid rounding?

    vtxt_font* Font = State->Font;
    vtxt_glyph Glyph = Font->Glyphs[GlyphIndex - VT_UNICODE_RANGE_START];
    Glyph.Advance *= Scale;
    Glyph.W *= Scale;
    Glyph.H *= Scale;
    Glyph.OffsetX *= Scale;
    Glyph.OffsetY *= Scale;

    int Stride = sizeof(vtxt_vertex1);

    float Top = State->CursorY + Glyph.OffsetY;
    float Bot = State->CursorY + Glyph.OffsetY + Glyph.H;
    float Left = State->CursorX + Glyph.OffsetX + OffsetX;
    float Right = State->CursorX + Glyph.OffsetX + Glyph.W + OffsetX;

    if (State->FlipY)
    {
        Top = State->CursorY - Glyph.OffsetY;
        Bot = State->CursorY - Glyph.OffsetY - Glyph.H;
    }

    // if(State->GetNormalizedDeviceCoords)
    // {
    //     Top = (1.f - ((Top / _vtxt_screen_h_for_clipspace) * 2.f));
    //     Bot = (1.f - ((Bot / _vtxt_screen_h_for_clipspace) * 2.f));
    //     Left = ((Left / _vtxt_screen_w_for_clipspace) * 2.f) - 1.f;
    //     Right = ((Right / _vtxt_screen_w_for_clipspace) * 2.f) - 1.f;
    // }

    State->CursorX += (vtxt_s32)Glyph.Advance;

    if (GlyphIndex <= 32)
    {
        // No need to append vertices for whitespace
        return;
    }

    vtxt_u32 BaseOffset = State->VertexCount1;
    State->VertexBuffer1[BaseOffset+0] = { Left, Bot, Glyph.MinU, Glyph.MinV, 
                                           ColorRGBA[0], ColorRGBA[1], ColorRGBA[2], ColorRGBA[3] };
    State->VertexBuffer1[BaseOffset+1] = { Right, Top, Glyph.MaxU, Glyph.MaxV, 
                                           ColorRGBA[0], ColorRGBA[1], ColorRGBA[2], ColorRGBA[3] };
    State->VertexBuffer1[BaseOffset+2] = { Left, Top, Glyph.MinU, Glyph.MaxV,
                                           ColorRGBA[0], ColorRGBA[1], ColorRGBA[2], ColorRGBA[3] };
    State->VertexBuffer1[BaseOffset+3] = { Left, Bot, Glyph.MinU, Glyph.MinV,
                                           ColorRGBA[0], ColorRGBA[1], ColorRGBA[2], ColorRGBA[3] };
    State->VertexBuffer1[BaseOffset+4] = { Right, Bot, Glyph.MaxU, Glyph.MinV,
                                           ColorRGBA[0], ColorRGBA[1], ColorRGBA[2], ColorRGBA[3] };
    State->VertexBuffer1[BaseOffset+5] = { Right, Top, Glyph.MaxU, Glyph.MaxV,
                                           ColorRGBA[0], ColorRGBA[1], ColorRGBA[2], ColorRGBA[3] };
    State->VertexCount1 += 6;
}

VTXT_DEF void
VT_AppendLine1(vtxt_state *State, char *Text, float Scale,
              float *ColorRGBA, vtxt_align Align)
{
    while (*Text != '\0')
    {
        char *LineStart = Text;
        vtxt_s32 LineStartX = State->CursorX;

        vtxt_s32 LineLength = 0;

        while (*Text != '\0' && *Text != '\n')
        {
            ++Text;
            ++LineLength;
        }

        float LineW = 0.0f;
        float LineH;
        if (Align != VT_Left)
        {
            VT_GetTextBoundingBoxDimensions(State->Font, LineStart, Scale,
                                            &LineW, &LineH, LineLength);
        }

        for (vtxt_s32 Index = 0; Index < LineLength; ++Index)
        {
            char in_glyph = LineStart[Index];

            // TODO(Kevin): Make sure we are not exceeding the array size

            float OffsetX = 0.0f;
            if (Align == VT_Center)
            {
                OffsetX = (float)(-LineW)/2.0f;
            }
            else if (Align == VT_Right)
            {
                OffsetX = (float)(-LineW);
            }
            VT_AppendGlyph1(State, in_glyph, Scale, OffsetX, ColorRGBA);
        }

        if (*Text == '\n')
        {
            VT_NewLine(State, Scale, LineStartX);
            ++Text;
        }
    }
}

VTXT_DEF void
VT_GetTextBoundingBoxDimensions(vtxt_font *Font, char *Text, float Scale,
                                float *OutWidth, float *OutHeight, vtxt_s32 Length)
{
    float BoxWidth = 0;
    float BoxHeight = 0;
    float LineWidth = 0;
    vtxt_s32 CharCount = 0;

    while (*Text != '\0' && (Length == 0 || CharCount < Length))
    {
        if (*Text != '\n')
        {
            if (*Text < VT_UNICODE_RANGE_START || *Text > VT_UNICODE_RANGE_END)
            {
                continue;
            }

            bool IsLastGlyphInLine = *(Text + 1) == '\0' || *(Text + 1) == '\n';

            vtxt_glyph Glyph = Font->Glyphs[*Text - VT_UNICODE_RANGE_START];
            Glyph.Advance *= Scale;
            Glyph.W *= Scale;
            Glyph.H *= Scale;
            Glyph.OffsetX *= Scale;
            // Glyph.OffsetY *= Scale;

            LineWidth += IsLastGlyphInLine ? Glyph.OffsetX + Glyph.W : Glyph.Advance;

            if (IsLastGlyphInLine)
            {
                if (LineWidth > BoxWidth)
                {
                    BoxWidth = LineWidth;
                }

                float LineHeight = Font->LineHeight * Scale;

                // TODO(Kevin): Only add font->linegap if there is a new line with legit characters
                BoxHeight += LineHeight;
            }
        }
        ++Text; // next character
        ++CharCount;
    }

    *OutWidth = BoxWidth;
    *OutHeight = BoxHeight;
}

VTXT_DEF void
VT_NewLine(vtxt_state *State, float Scale, vtxt_s32 X)
{
    State->CursorX = X;

    float YAdvanceBeforeScale = State->Font->LineHeight;

    // TODO(Kevin): No implicit casting!
    vtxt_s32 YAdvance = (vtxt_s32)(YAdvanceBeforeScale*Scale);

    if (State->PutNewLinesAbove)
    {
        YAdvance = -YAdvance;
    }
    if (State->FlipY)
    {
        YAdvance = -YAdvance;
    }

    State->CursorY += YAdvance;
}

#ifdef VT_USING_STB_TRUETYPE

#define STB_RECT_PACK_IMPLEMENTATION
#include "stb_rect_pack.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

VTXT_DEF void
VT_CreateFontFromTTF(vtxt_u8 *TrueTypeData, float FontHeightPx, vtxt_font *OutFont,
                     vtxt_s32 AtlasW, vtxt_s32 AtlasH, vtxt_u8 *OutAtlasPixels)
{
    stbtt_pack_context PackContext;

    if (stbtt_PackBegin(&PackContext, OutAtlasPixels, AtlasW, AtlasH, 0, 1, NULL))
    {
        stbtt_packedchar PackedGlyphData[VT_UNICODE_RANGE];
        stbtt_PackFontRange(&PackContext, TrueTypeData, 0, FontHeightPx, VT_UNICODE_RANGE_START,
                            VT_UNICODE_RANGE, PackedGlyphData);
        stbtt_PackEnd(&PackContext);

        stbtt_fontinfo FontInfo;
        stbtt_InitFont(&FontInfo, TrueTypeData, 0);
        float ScaleForPx = stbtt_ScaleForPixelHeight(&FontInfo, FontHeightPx);
        int Ascender, Descender, Linegap;
        stbtt_GetFontVMetrics(&FontInfo, &Ascender, &Descender, &Linegap);

        *OutFont = {};
        OutFont->FontHeightPx = (vtxt_s32)FontHeightPx;
        OutFont->LineHeight = (float)(Ascender+Descender+Linegap)*ScaleForPx;

        for (vtxt_u8 Codepoint = VT_UNICODE_RANGE_START;
             Codepoint <= VT_UNICODE_RANGE_END;
             ++Codepoint)
        {
            stbtt_packedchar PackedGlyph = PackedGlyphData[Codepoint - VT_UNICODE_RANGE_START];

            vtxt_glyph *Glyph = &OutFont->Glyphs[Codepoint - VT_UNICODE_RANGE_START];
            Glyph->Codepoint = Codepoint;
            Glyph->MinU = (float)PackedGlyph.x0/(float)AtlasW;
            Glyph->MinV = (float)PackedGlyph.y1/(float)AtlasH;
            Glyph->MaxU = (float)PackedGlyph.x1/(float)AtlasW;
            Glyph->MaxV = (float)PackedGlyph.y0/(float)AtlasH;
            Glyph->OffsetX = PackedGlyph.xoff;
            Glyph->OffsetY = PackedGlyph.yoff;
            Glyph->Advance = PackedGlyph.xadvance;
            Glyph->W = (float)(PackedGlyph.x1-PackedGlyph.x0);
            Glyph->H = (float)(PackedGlyph.y1-PackedGlyph.y0);
        }
    }
}
#endif // VT_USING_STB_TRUETYPE

#endif // VERTEXT_IMPLEMENTATION
