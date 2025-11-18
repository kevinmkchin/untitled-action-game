#include "shooter_gui++.h"
#define VERTEXT_STATIC
#define VERTEXT_IMPLEMENTATION
#include "shooter_vertext.h"

#define EditorBgColor1 vec4(0.05f, 0.05f, 0.05f, 1.0f)
#define EditorFgColor1 vec4(1,1,1,1)
#define EditorPadL 0
#define EditorPadR 0
#define EditorPadT 0
#define EditorPadB 0
#define EditorButtonNormalColor     vec4(RGB255TO1(20, 50, 65), 1)
#define EditorButtonHoveredColor    vec4(RGB255TO1(30, 60, 80), 1)
#define EditorButtonActiveColor     vec4(RGB255TO1(10, 30, 60), 1)

internal u32
MakeWindowElementID(window_state *Window)
{
    u32 BaseWindowID = 0xA0000001;
    u32 Result = BaseWindowID + (Window->GuiState->WindowCount*0x10000 + 
                                 Window->ElementIDCount++);
    return Result;
}

internal vtxt_font *
GetWindowFontInfo(window_state *Window)
{
    gui_font FontIndex = Window->TextFont;
    vtxt_font *FontInfo = Window->GuiState->Fonts[FontIndex].FontInfo;
    return FontInfo;
}

// TODO(Kevin): User provides a unique window id or name which becomes a stable base ID
internal window_state
BeginWindow(gui_state *GuiState, gui_rect Rect)
{
    window_state Window = {};

    Window.GuiState = GuiState;
    Window.Rect = Rect;
    Window.NextElementOffset = { Rect.X + EditorPadL, Rect.Y + EditorPadT };
    Window.TextFont = GuiFontCambria40;
    Window.TextScale = 1.3f; // TODO(Kevin): Text scaling option (scaling down < 1.0 looks bad)
    Window.TextColor = EditorFgColor1;
    Window.BgColor = EditorBgColor1;
    Window.LineHeightPx = CeilReal32ToU32(GetWindowFontInfo(&Window)->LineHeight * Window.TextScale);

    GuiPushScissor(Window.GuiState, Window.Rect);
    GuiPanel(Window.GuiState, Window.Rect, Window.BgColor);

    u32 WindowID = MakeWindowElementID(&Window);
    if (MouseInside(GuiState, Rect))
    {
        RequestSetHovered(GuiState, WindowID);
    }

    GuiState->WindowCount++;

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

    vtxt_font *FontInfo = GetWindowFontInfo(Window);

    s32 X = Window->NextElementOffset.X;
    s32 Y = Window->NextElementOffset.Y + 
            RoundReal32ToInt32((real32)FontInfo->FontHeightPx * Window->TextScale);
    GuiText(Window->GuiState, X, Y,
            Window->TextFont, Window->TextScale,
            Align_Left, false,
            Window->TextColor, Text);

    real32 W, H;
    VT_GetTextBoundingBoxDimensions(FontInfo, Text, Window->TextScale, &W, &H);

    StageElementSize(Window, CeilReal32ToS32(W), CeilReal32ToS32(H));
}

internal bool32
WindowButton(window_state *Window, char *Text)
{
    CommitElementSize(Window);

    u32 ID = MakeWindowElementID(Window);
    gui_rect Rect = { Window->NextElementOffset.X, Window->NextElementOffset.Y,
                      Window->Rect.W, Window->LineHeightPx };
    bool32 Result = GuiButton(Window->GuiState, ID, Rect, EditorButtonNormalColor,
                              EditorButtonHoveredColor, EditorButtonActiveColor);

    vtxt_font *FontInfo = GetWindowFontInfo(Window);

    GuiPushScissor(Window->GuiState, Rect);
    GuiText(Window->GuiState,
            Rect.X + Rect.W/2,
            Rect.Y + (Rect.H + CeilReal32ToS32(FontInfo->FontHeightPx*0.6f*Window->TextScale))/2,
            Window->TextFont, Window->TextScale,
            Align_Center, false,
            Window->TextColor, Text);
    GuiPopScissor(Window->GuiState);

    StageElementSize(Window, Rect.W, Rect.H);

    return Result;
}