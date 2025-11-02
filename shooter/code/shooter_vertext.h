/*

    vertext.h

    Text mesh generation library (https://kevin.gd/)
    Generates meshes composed of quads for text rendering

    Build a vtxt_font manually or use the helper function

*/
#ifndef INCLUDE_VERTEXT_H
#define INCLUDE_VERTEXT_H

#ifndef VTXT_ASCII_FROM
#define VTXT_ASCII_FROM 0 // starting ASCII codepoint to collect glyph data for
#endif
#ifndef VTXT_ASCII_TO
#define VTXT_ASCII_TO 255 // ending ASCII codepoint to collect glyph data for
#endif
#define VTXT_GLYPH_COUNT VTXT_ASCII_TO - VTXT_ASCII_FROM + 1

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
    vtxt_glyph Glyphs[VTXT_GLYPH_COUNT];
};

struct vtxt_mono_bitmap
{
    vtxt_s32 W, H;
    char *Pixels;
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
VT_AppendLine(vtxt_state *State, char *Text, float Scale,
              float *ColorRGBA, vtxt_align Align = VT_Left);

VTXT_DEF void
VT_AppendGlyph1(vtxt_state *State, char GlyphIndex, float Scale, float OffsetX, float *ColorRGBA);

VTXT_DEF void
VT_GetTextBoundingBoxDimensions(vtxt_font *Font, char *Text, float Scale,
                                float *OutWidth, float *OutHeight, vtxt_s32 Length = 0);
VTXT_DEF void
VT_NewLine(vtxt_state *State, float Scale, vtxt_s32 X);

#endif // INCLUDE_VERTEXT_H


#ifdef VERTEXT_IMPLEMENTATION

VTXT_DEF void
VT_AppendGlyph1(vtxt_state *State, char GlyphIndex, float Scale, float OffsetX, float *ColorRGBA)
{
    if(GlyphIndex < VTXT_ASCII_FROM || GlyphIndex > VTXT_ASCII_TO)
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
    vtxt_glyph Glyph = Font->Glyphs[GlyphIndex - VTXT_ASCII_FROM];
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
VT_AppendLine(vtxt_state *State, char *Text, float Scale,
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
            if (*Text < VTXT_ASCII_FROM || *Text > VTXT_ASCII_TO)
            {
                continue;
            }

            bool IsLastGlyphInLine = *(Text + 1) == '\0' || *(Text + 1) == '\n';

            vtxt_glyph Glyph = Font->Glyphs[*Text - VTXT_ASCII_FROM];
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

// #if VT_USING_STB_TRUETYPE
// VTXT_DEF void
// VT_GenerateFont()
// {
//     // TODO(Kevin): Implement
// }
// #endif

#endif // VERTEXT_IMPLEMENTATION

#if 0

#define _vtxt_internal static      // vtxt local static variable
#ifndef VTXT_MAX_CHAR_IN_BUFFER
#define VTXT_MAX_CHAR_IN_BUFFER 400  // maximum characters allowed in vertex buffer ("canvas")
#endif
#define VTXT_MAX_FONT_RESOLUTION 100   // maximum font resolution when initializing font
#define VTXT_DESIRED_ATLAS_WIDTH 400   // width of the font atlas
#define VTXT_ATLAS_PAD_X 1                // x padding between the glyph textures on the texture atlas
#define VTXT_ATLAS_PAD_Y 1                // y padding between the glyph textures on the texture atlas

#define _vtxt_ceil(num) ((num) == (float)((int)(num)) ? (int)(num) : (((int)(num)) + 1))

// Buffers for vertices and texture_coords before they are written to GPU memory.
// If you have a pointer to these buffers, DO NOT let these buffers be overwritten
// before you copy it or upload to GPU memory.
_vtxt_internal float _vtxt_vertex_buffer[VTXT_MAX_CHAR_IN_BUFFER * 6 * 8]; // n characters * 6 vertices * (2 xy + 2 uv + 4 rgba)
_vtxt_internal int _vtxt_vertex_count = 0; // Each vertex takes up 4 places in the assembly_buffer
_vtxt_internal unsigned int _vtxt_index_buffer[VTXT_MAX_CHAR_IN_BUFFER * 6];
_vtxt_internal int _vtxt_index_count = 0;
_vtxt_internal int _vtxt_config = 0b0;
_vtxt_internal float _vtxt_linegap_offset = 0.f;
_vtxt_internal int _vtxt_cursor_x = 0;   // top left of the screen is pixel (0, 0), bot right of the screen is pixel (screen buffer width, screen buffer height)
_vtxt_internal int _vtxt_cursor_y = 100; // cursor points to the base line at which to start drawing the glyph
_vtxt_internal int _vtxt_screen_w_for_clipspace = 800;
_vtxt_internal int _vtxt_screen_h_for_clipspace = 600;

VTXT_DEF vtxt_bitmap
vtxt_init_font(vtxt_font* font_handle, unsigned char* font_buffer, int font_height_in_pixels)
{
    int desired_atlas_width = VTXT_DESIRED_ATLAS_WIDTH;

    if(font_height_in_pixels > VTXT_MAX_FONT_RESOLUTION)
    {
        return;
    }

    // Font metrics
    stbtt_fontinfo stb_font_info;
    stbtt_InitFont(&stb_font_info, font_buffer, 0);
    float stb_scale = stbtt_ScaleForPixelHeight(&stb_font_info, (float)font_height_in_pixels);
    int stb_ascender;
    int stb_descender;
    int stb_linegap;
    stbtt_GetFontVMetrics(&stb_font_info, &stb_ascender, &stb_descender, &stb_linegap);
    font_handle->font_height_px = font_height_in_pixels;
    font_handle->ascender = (float)stb_ascender * stb_scale;
    font_handle->descender = (float)stb_descender * stb_scale;
    font_handle->linegap = (float)stb_linegap * stb_scale;

    // LOAD GLYPH BITMAP AND INFO FOR EVERY CHARACTER WE WANT IN THE FONT
    vtxt_bitmap temp_glyph_bitmaps[VTXT_GLYPH_COUNT];
    int tallest_glyph_height = 0;
    int aggregate_glyph_width = 0;
    // load glyph data
    for (int char_index = VTXT_ASCII_FROM; char_index <= VTXT_ASCII_TO; ++char_index) // ASCII
    {
        vtxt_glyph glyph;
        
        // get glyph metrics from stbtt
        int stb_advance;
        int stb_leftbearing;
        stbtt_GetCodepointHMetrics(&stb_font_info, 
                                   char_index, 
                                   &stb_advance, 
                                   &stb_leftbearing);
        glyph.codepoint = (unsigned char)char_index;
        glyph.advance = (float)stb_advance * stb_scale;
        int stb_width, stb_height;
        int stb_offset_x, stb_offset_y;
        unsigned char* stb_bitmap_temp = stbtt_GetCodepointBitmap(&stb_font_info,
                                                                0, stb_scale,
                                                                char_index,
                                                                &stb_width,
                                                                &stb_height,
                                                                &stb_offset_x,
                                                                &stb_offset_y);
        glyph.width = (float)stb_width;
        glyph.height = (float)stb_height;
        glyph.offset_x = (float)stb_offset_x;
        glyph.offset_y = (float)stb_offset_y;

        // Copy stb_bitmap_temp bitmap into glyph's pixels bitmap so we can free stb_bitmap_temp
        int iter = char_index - VTXT_ASCII_FROM;
        temp_glyph_bitmaps[iter].pixels = (unsigned char*) calloc((size_t)glyph.width * (size_t)glyph.height, 1);
        for(int row = 0; row < (int) glyph.height; ++row)
        {
            for(int col = 0; col < (int) glyph.width; ++col)
            {
                // Flip the bitmap image from top to bottom to bottom to top
                temp_glyph_bitmaps[iter].pixels[row * (int) glyph.width + col] = stb_bitmap_temp[((int) glyph.height - row - 1) * (int) glyph.width + col];
            }
        }
        temp_glyph_bitmaps[iter].width = (int) glyph.width;
        temp_glyph_bitmaps[iter].height = (int) glyph.height;
        aggregate_glyph_width += (int)glyph.width + VTXT_ATLAS_PAD_X;
        if(tallest_glyph_height < (int)glyph.height)
        {
            tallest_glyph_height = (int)glyph.height;
        }
        stbtt_FreeBitmap(stb_bitmap_temp, 0);

        font_handle->glyphs[iter] = glyph;
    }

    int desired_atlas_height = (tallest_glyph_height + VTXT_ATLAS_PAD_Y)
                               * _vtxt_ceil((float)aggregate_glyph_width / (float)desired_atlas_width);
    // Build font atlas bitmap based on these parameters
    vtxt_bitmap atlas;
    atlas.pixels = (unsigned char*) calloc(desired_atlas_width * desired_atlas_height, 1); // TODO avoid calloc here
    atlas.width = desired_atlas_width;
    atlas.height = desired_atlas_height;
    // COMBINE ALL GLYPH BITMAPS INTO FONT ATLAS
    int glyph_count = VTXT_GLYPH_COUNT;
    int atlas_x = 0;
    int atlas_y = 0;
    for(int i = 0; i < glyph_count; ++i)
    {
        vtxt_bitmap glyph_bitmap = temp_glyph_bitmaps[i];
        if (atlas_x + glyph_bitmap.width > atlas.width) // check if move atlas bitmap cursor to next line
        {
            atlas_x = 0;
            atlas_y += tallest_glyph_height + VTXT_ATLAS_PAD_Y;
        }

        for(int glyph_y = 0; glyph_y < glyph_bitmap.height; ++glyph_y)
        {
            for(int glyph_x = 0; glyph_x < glyph_bitmap.width; ++glyph_x)
            {
                atlas.pixels[(atlas_y + glyph_y) * atlas.width + atlas_x + glyph_x]
                    = glyph_bitmap.pixels[glyph_y * glyph_bitmap.width + glyph_x];
            }
        }
        font_handle->glyphs[i].MinU = (float) atlas_x / (float) atlas.width;
        font_handle->glyphs[i].MinV = (float) atlas_y / (float) atlas.height;
        font_handle->glyphs[i].max_u = (float) (atlas_x + glyph_bitmap.width) / (float) atlas.width;
        font_handle->glyphs[i].max_v = (float) (atlas_y + glyph_bitmap.height) / (float) atlas.height;

        atlas_x += glyph_bitmap.width + VTXT_ATLAS_PAD_X; // move the atlas bitmap cursor by glyph bitmap width

        free(glyph_bitmap.pixels);
    }
    return atlas;
}

#endif