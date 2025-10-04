#ifndef SHOOTER_GUI_H
#define SHOOTER_GUI_H

#include "shooter_types.h"
#include "shooter_shared.h"
#include "shooter_renderer.h"

/*

    Thin immediate-mode GUI layer

    Handles input and generates draw requests
    Does not do any auto-styling -> all styling is up to the user

    Keep it dead simple -> no ordering just painter's alg

*/

enum gui_alignment
{
    Align_Left,
    Align_Center,
    Align_Right
};

struct gui_rect
{
    s32 X, Y;
    u32 W, H;
};

struct gui_vertex
{
    vec2 PixelP; // 0,0 is top left
    vec2 UV0;
    vec4 Color;
};

struct gui_draw_request
{
    gui_rect Scissor;
    vec4 Color;
    union
    {
        struct
        {
            gui_rect Rect;
            texture_handle Texture;
        };

        // struct
        // {
        //     char *Text;
        //     int32 FontSize, X, Y;
        //     gui_alignment Align;
        //     bool32 PutNewLinesAbove;
        // };
    };
};

struct gui_state
{

    #define MAX_GUI_DRAW_REQUESTS 1000
    gui_draw_request DrawRequests[MAX_GUI_DRAW_REQUESTS];
    u32 DrawRequestsCount;

    #define MAX_SCISSOR_STACK 8
    gui_rect ScissorStack[MAX_SCISSOR_STACK];
    u32 ScissorStackCount;
};

void GuiPushScissor(gui_state *GuiState, gui_rect Scissor);
void GuiPopScissor(gui_state *GuiState);

void GuiPanel(gui_state *GuiState, gui_rect Rect, vec4 RGBA);
void GuiPanel(gui_state *GuiState, gui_rect Rect, texture_handle Image);
// void GuiButton(u32 ID, gui_rect Rect, vec4 RGBA, vec4 RGBAHovered, vec4 RGBAActive);
// void GuiText(int32 X, int32 Y, int32 FontSize, gui_alignment Align, bool32 PutNewLinesAbove, char *Text);
// void GuiTextFmt(int32 X, int32 Y, int32 FontSize, gui_alignment Align, bool32 PutNewLinesAbove, char *Text, ...);
// void GuiCheckbox(u32 ID, gui_rect Rect, int32 Inset, bool32 *Value, vec4 RGBABackground, vec4 RGBAForeground);

void GuiInitialize(gui_state *GuiState);
void GuiBeginFrame(gui_state *GuiState, u32 CanvasWidth, u32 CanvasHeight);

#endif // SHOOTER_GUI_H
