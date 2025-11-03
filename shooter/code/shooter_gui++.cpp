#define VERTEXT_STATIC
#define VERTEXT_IMPLEMENTATION
#include "shooter_vertext.h"

#define EditorBgColor1 vec4(0.05f, 0.05f, 0.05f, 1.0f)
#define EditorFgColor1 vec4(1,1,1,1)
#define EditorPadL 2
#define EditorPadR 2
#define EditorPadT 2
#define EditorPadB 2

struct window_state
{
    gui_state *GuiState;
    gui_rect Rect;
    gui_font TextFont;
    real32 TextScale;
    vec4 TextColor;
    vec4 BgColor;

    ivec2 NextElementOffset;
    ivec2 PrevElementSize;
    bool32 HorizontalMode;
};

internal window_state
BeginWindow(gui_state *GuiState, gui_rect Rect)
{
    window_state Window = {};

    Window.GuiState = GuiState;
    Window.Rect = Rect;
    Window.NextElementOffset = { Rect.X + EditorPadL, Rect.Y + EditorPadT };
    Window.TextFont = GuiFontCambria40;
    Window.TextScale = 1.0f; // TODO(Kevin): Text scaling option
    Window.TextColor = EditorFgColor1;
    Window.BgColor = EditorBgColor1;

    GuiPushScissor(Window.GuiState, Window.Rect);
    GuiPanel(Window.GuiState, Window.Rect, Window.BgColor);

    return Window;
}

internal void
EndWindow(window_state *Window)
{
    GuiPopScissor(Window->GuiState);
}

internal void
StageElementSize(window_state *Window, s32 W, s32 H)
{
    W += EditorPadL + EditorPadR;
    H += EditorPadT + EditorPadB;

    if (Window->HorizontalMode)
    {
        Window->PrevElementSize.X = W;
        Window->PrevElementSize.Y = GMATH_Max(H, Window->PrevElementSize.Y);
    }
    else
    {
        Window->PrevElementSize.X = GMATH_Max(W, Window->PrevElementSize.X);
        Window->PrevElementSize.Y = H;
    }
}

internal void
CommitElementSize(window_state *Window)
{
    if (Window->HorizontalMode)
    {
        Window->NextElementOffset.X += Window->PrevElementSize.X;
    }
    else
    {
        Window->NextElementOffset.X = Window->Rect.X + EditorPadL;
        Window->NextElementOffset.Y += Window->PrevElementSize.Y;
    }
}

internal void
WindowText(window_state *Window, char *Text)
{
    CommitElementSize(Window);

    gui_font FontIndex = Window->TextFont;
    vtxt_font *FontInfo = Window->GuiState->Fonts[FontIndex].FontInfo;

    s32 X = Window->NextElementOffset.X;
    s32 Y = Window->NextElementOffset.Y + 
            RoundReal32ToInt32((real32)FontInfo->FontHeightPx * Window->TextScale);
    vec4 TextColor = Window->TextColor;
    GuiText(Window->GuiState, X, Y, FontIndex, Window->TextScale,
            Align_Left, false, TextColor, Text);

    real32 W, H;
    VT_GetTextBoundingBoxDimensions(FontInfo, Text, Window->TextScale, &W, &H);

    StageElementSize(Window, CeilReal32ToS32(W), CeilReal32ToS32(H));
}
