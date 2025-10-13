#ifndef SHOOTER_GUI_H
#define SHOOTER_GUI_H

#include "shooter_types.h"
#include "shooter_shared.h"
#include "shooter_renderer.h"

/*

    Thin immediate-mode GUI layer

    Handles input and generates draw requests
    Does not do any auto-styling -> all styling is up to the user
    User provides element ID

    Keep it dead simple -> no ordering just painter's alg

Usage

    uint32 CanvasW, CanvasH;
    RendererAPI->GetRenderTargetDimensions(&CanvasW, &CanvasH);
    GuiBeginFrame(GuiState, CanvasW, CanvasH, &GameState->ScratchArena, 
                  Input->MouseX, Input->MouseY, Input->MouseButtons[0], Input->MouseButtons[1]);
    GuiPushScissor(GuiState, { 250, 250, 1000, 100 });
    GuiPanel(GuiState, { 200, 200, 100, 100 }, 2);
    GuiPushScissor(GuiState, { 200, 200, 600, 125 });
    GuiPanel(GuiState, { 300, 300, 100, 100 }, vec4(1.0f, 0.0f, 1.0f, 1.0f));
    GuiPanel(GuiState, { 500, 100, 800, 400 }, 4);
    GuiPopScissor(GuiState);
    GuiPopScissor(GuiState);
    GuiPanel(GuiState, { 400, 400, 100, 100 }, vec4(0.0f, 1.0f, 1.0f, 1.0f));
    GuiText(GuiState, 50, 160, 36, GuiFontAnikki8x8, Align_Left, false, vec4(0,1,0,1), "Hello world");
    GuiText(GuiState, 50, 200, 36, GuiFontCurses6x9, Align_Left, false, vec4(0,1,1,0.5f), "Hello world");

    GuiPanel(GuiState, { 500, 100, 600, 600 }, 6);

    if (GuiButton(GuiState, 1, {100, 500, 200, 200}, vec4(0.0f,0.7f,0.0f,1.0f), 
                  vec4(0.0f,0.5f,0.0f,1.0f), vec4(0.0f,1.0f,0.0f,1.0f)))
    {
        GuiPanel(GuiState, { 500, 500, 200, 200 }, vec4(0,0,1,1));
    }

    GuiPanel(GuiState, { Input->MouseX, Input->MouseY, 16, 16 }, vec4(1.0f, 1.0f, 0.0f, 1.0f));

    RendererAPI->PushGuiDrawRequests(&GameState->ScratchArena, GuiState->DrawRequests, GuiState->DrawRequestsCount);


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

enum gui_font
{
    GuiFontCurses6x9,
    GuiFontAnikki8x8,
};

enum gui_draw_type
{
    GuiDrawRect,
    GuiDrawText
};

struct gui_draw_request
{
    gui_rect Scissor;
    vec4 Color;

    gui_draw_type DrawType;
    union
    {
        struct
        {
            gui_rect Rect;
            texture_handle Texture;
        };
        struct
        {
            char *Text;
            s32 X, Y;
            u32 FontSize;
            font_handle Font;
            gui_alignment Align;
            bool32 PutNewLinesAbove;
        };
    };
};

struct gui_persist_state
{
    u32 Active;
    u32 Hovered;
    u32 HoveredNextFrame;
};

struct gui_state
{
    u32 CallOrder; // TODO(Kevin): Can we use low 24 bits for call order and high 8 bits for "depth"? then sort draw requests by that

    #define MAX_GUI_DRAW_REQUESTS 1000
    gui_draw_request DrawRequests[MAX_GUI_DRAW_REQUESTS];
    u32 DrawRequestsCount;

    #define MAX_SCISSOR_STACK 8
    gui_rect ScissorStack[MAX_SCISSOR_STACK];
    u32 ScissorStackCount;

    s32 MouseX;
    s32 MouseY;
    bool32 LMBWentDown;
    bool32 LMBWentUp;
    bool32 RMBWentDown;
    bool32 RMBWentUp;
    // TODO(Kevin): Array of typed keys to read keyboard input from

    memory_arena *Scratch;

    font_handle Fonts[2];

    gui_persist_state InteractionState;
};

bool32 GuiIsAnyElementActive(gui_state *GuiState);
bool32 GuiBehaviourButton(gui_state *GuiState, u32 ID, gui_rect Rect);

void GuiPushScissor(gui_state *GuiState, gui_rect Scissor);
void GuiPopScissor(gui_state *GuiState);

void GuiPanel(gui_state *GuiState, gui_rect Rect, vec4 RGBA);
void GuiPanel(gui_state *GuiState, gui_rect Rect, texture_handle Image);
bool32 GuiButton(gui_state *GuiState, u32 ID, gui_rect Rect, vec4 RGBA, vec4 RGBAHovered, vec4 RGBAActive);
bool32 GuiButton(gui_state *GuiState, u32 ID, gui_rect Rect, texture_handle Image, 
                 texture_handle ImageHovered, texture_handle ImageActive);
// void GuiCheckbox(u32 ID, gui_rect Rect, int32 Inset, bool32 *Value, vec4 RGBABackground, vec4 RGBAForeground);
void GuiTextFmt(gui_state *GuiState, s32 X, s32 Y, u32 FontSize, gui_font FontIndex, gui_alignment Align,
                bool32 PutNewLinesAbove, vec4 TextColor, char *TextFmt, ...);
void GuiText(gui_state *GuiState, s32 X, s32 Y, u32 FontSize, gui_font FontIndex, gui_alignment Align,
             bool32 PutNewLinesAbove, vec4 TextColor, char *Text);

void GuiBeginFrame(gui_state *GuiState, u32 CanvasWidth, u32 CanvasHeight, memory_arena *Scratch,
                   s32 MouseX, s32 MouseY, game_button_state LMB, game_button_state RMB);

#endif // SHOOTER_GUI_H
