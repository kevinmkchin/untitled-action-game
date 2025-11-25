#ifndef SHOOTER_GUI_PP_H
#define SHOOTER_GUI_PP_H

struct window_state
{
    gui_state *GuiState;
    gui_rect Rect;
    gui_font TextFont;
    real32 TextScale;
    u32 LineHeightPx;
    vec4 TextColor;
    vec4 BgColor;

    ivec2 NextElementOffset;
    ivec2 PrevElementSize;
    bool32 HorizontalMode;

    u32 ElementIDCount;
};

internal window_state
BeginWindow(gui_state *GuiState, gui_rect Rect, bool32 DrawBackground);
internal void
EndWindow(window_state *Window);
internal void
StageElementSize(window_state *Window, s32 W, s32 H);
internal void
CommitElementSize(window_state *Window);
internal void
WindowText(window_state *Window, char *Text);
internal bool32
WindowButton(window_state *Window, char *Text, bool32 *Selected = 0);

#endif // SHOOTER_GUI_PP_H
