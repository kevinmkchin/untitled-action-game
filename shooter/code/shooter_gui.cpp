#include "shooter_gui.h"
#include "stb_sprintf.h"

// /* Immediate-mode state helpers */
// bool IsActive(ui_id id);
// bool IsHovered(ui_id id);
// void SetActive(ui_id id);
// void RequestSetHovered(ui_id id);
// bool MouseWentUp();
// bool MouseWentDown();
// bool MouseInside(const UIRect& rect);

internal gui_draw_request *NextDrawRequest(gui_state *GuiState)
{
    gui_draw_request *Result = 0;
    if (GuiState->DrawRequestsCount < MAX_GUI_DRAW_REQUESTS)
    {
        Result = &(GuiState->DrawRequests[GuiState->DrawRequestsCount++]);
        Assert(GuiState->ScissorStackCount > 0);
        Result->Scissor = GuiState->ScissorStack[GuiState->ScissorStackCount-1];
    }
    return Result;
}

void GuiPushScissor(gui_state *GuiState, gui_rect Scissor)
{
    if (GuiState->ScissorStackCount < MAX_SCISSOR_STACK)
    {
        gui_rect NewScissor = Scissor;

        if (GuiState->ScissorStackCount > 0)
        {
            // NOTE(Kevin): Find intersection between LastScissor and Scissor
            gui_rect LastScissor = GuiState->ScissorStack[GuiState->ScissorStackCount-1];
            s32 IntersectionMinX = Maximum(LastScissor.X, Scissor.X);
            s32 IntersectionMinY = Maximum(LastScissor.Y, Scissor.Y);
            s32 IntersectionMaxX = Minimum(LastScissor.X+SafeCastToInt32(LastScissor.W), 
                                           Scissor.X+SafeCastToInt32(Scissor.W));
            s32 IntersectionMaxY = Minimum(LastScissor.Y+SafeCastToInt32(LastScissor.H), 
                                           Scissor.Y+SafeCastToInt32(Scissor.H));
            if (IntersectionMinX < IntersectionMaxX &&
                IntersectionMinY < IntersectionMaxY)
            {
                NewScissor = { IntersectionMinX, IntersectionMinY,
                               SafeCastToUInt32(IntersectionMaxX-IntersectionMinX),
                               SafeCastToUInt32(IntersectionMaxY-IntersectionMinY) };
            }
            else
            {
                NewScissor = { 0, 0, 0, 0 };
            }
        }

        GuiState->ScissorStack[GuiState->ScissorStackCount++] = NewScissor;
    }
}

void GuiPopScissor(gui_state *GuiState)
{
    --GuiState->ScissorStackCount;
}

void GuiPanel(gui_state *GuiState, gui_rect Rect, vec4 RGBA)
{
    gui_draw_request *DrawRequest = NextDrawRequest(GuiState);
    if (DrawRequest)
    {
        DrawRequest->Color = RGBA;
        DrawRequest->Rect = Rect;
        DrawRequest->Texture = InvalidHandle;
    }
}

void GuiPanel(gui_state *GuiState, gui_rect Rect, texture_handle Image)
{
    gui_draw_request *DrawRequest = NextDrawRequest(GuiState);
    if (DrawRequest)
    {
        DrawRequest->Color = vec4(1.0f,1.0f,1.0f,1.0f);
        DrawRequest->Rect = Rect;
        DrawRequest->Texture = Image;
    }
}

void GuiTextFmt(gui_state *GuiState, int32 X, int32 Y, int32 FontSize, gui_alignment Align,
                bool32 PutNewLinesAbove, vec4 TextColor, char *TextFmt, ...)
{
    if (TextFmt)
    {
        char *CurrentAddress = (char *)ArenaCurrentAddress(GuiState->Scratch);

        va_list ArgPtr;
        va_start(ArgPtr, TextFmt);
        int TextLength = vsprintf(CurrentAddress, TextFmt, ArgPtr);
        va_end(ArgPtr);

        GuiText(GuiState, X, Y, FontSize, Align, PutNewLinesAbove, TextColor, CurrentAddress);
    }
}

void GuiText(gui_state *GuiState, int32 X, int32 Y, int32 FontSize, gui_alignment Align,
             bool32 PutNewLinesAbove, vec4 TextColor, char *Text)
{
    if (Text == NULL)
    {
        return;
    }

    gui_draw_request *DrawRequest = NextDrawRequest(GuiState);
    if (DrawRequest)
    {
        char *CurrentAddress = (char *)ArenaCurrentAddress(GuiState->Scratch);
        int TextLength = StringLength(Text);
        if (CurrentAddress != Text)
        {
            // NOTE(Kevin): If string is at same location as CurrentAddress, then we only need
            // to move the used index by the text length. No need for copy.
            Copy(CurrentAddress, Text, TextLength);
        }
        PushArray(GuiState->Scratch, TextLength+1, char); // to move the used index

        DrawRequest->Color = TextColor;
        DrawRequest->Text = CurrentAddress;
        DrawRequest->X;
        DrawRequest->Y;
        DrawRequest->FontSize;
        DrawRequest->Align = Align;
        DrawRequest->PutNewLinesAbove = PutNewLinesAbove;
        // TODO(Kevin): font info
    }
}

void GuiInitialize(gui_state *GuiState)
{

}

void GuiBeginFrame(gui_state *GuiState, u32 CanvasWidth, u32 CanvasHeight, memory_arena *Scratch)
{
    GuiState->Scratch = Scratch;
    GuiState->DrawRequestsCount = 0;
    GuiState->ScissorStackCount = 0;
    GuiPushScissor(GuiState, { 0, 0, CanvasWidth, CanvasHeight });
}

