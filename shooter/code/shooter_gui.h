#ifndef SHOOTER_GUI_H
#define SHOOTER_GUI_H

#include "shooter_types.h"
#include "shooter_shared.h"
#include "shooter_renderer.h"

//
// NOTE(Kevin): Thin immediate-mode GUI system
//              Handles input and generates draw requests
//              Does not do any auto-styling -> all styling is up to the user
//

enum gui_alignment
{
    Align_Left,
    Align_Center,
    Align_Right
};

struct gui_rect
{
    int32 X, Y, W, H;
};

struct gui_vertex
{
    vec2 PixelP; // 0,0 is top left
    vec2 UV0;
    vec4 Color;
};

struct gui_draw_request
{
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

};

// TODO(Kevin): everything can be masked/scissored, but scissor rects must be registered that frame

void GuiPanel(gui_rect Rect, vec4 RGBA);
void GuiPanel(gui_rect Rect, texture_handle Image);
// void GuiButton(uint32 ID, gui_rect Rect, vec4 RGBA, vec4 RGBAHovered, vec4 RGBAActive);
// void GuiText(int32 X, int32 Y, int32 FontSize, gui_alignment Align, bool32 PutNewLinesAbove, char *Text);
// void GuiTextFmt(int32 X, int32 Y, int32 FontSize, gui_alignment Align, bool32 PutNewLinesAbove, char *Text, ...);
// void GuiCheckbox(uint32 ID, gui_rect Rect, int32 Inset, bool32 *Value, vec4 RGBABackground, vec4 RGBAForeground);

void GuiInitialize();
void GuiBeginFrame();
// TODO(Kevin): Input event handling mouse and keyboard
// void GuiDraw(); // Maybe returns the list of ui draw requests

#endif // SHOOTER_GUI_H
