#include "shooter_gui.h"

// /* Immediate-mode state helpers */
// bool IsActive(ui_id id);
// bool IsHovered(ui_id id);
// void SetActive(ui_id id);
// void RequestSetHovered(ui_id id);
// bool MouseWentUp();
// bool MouseWentDown();
// bool MouseInside(const UIRect& rect);


void GuiPanel(gui_rect Rect, vec4 RGBA)
{
    gui_draw_request DrawRequest = {};
    DrawRequest->Rect = Rect;
    DrawRequest->Color = RGBA;
}

void GuiPanel(gui_rect Rect, texture_handle Image)
{

}


void GuiInitialize()
{

}

void GuiBeginFrame()
{

}



