#include "shooter_gui.h"
#include "stb_sprintf.h"

internal bool32
IsActive(gui_state *GuiState, u32 ID)
{
    bool32 Result = GuiState->Active == ID;
    return Result;
}

internal bool32
IsHovered(gui_state *GuiState, u32 ID)
{
    bool32 Result = GuiState->Hovered == ID;
    return Result;
}

internal void
SetActive(gui_state *GuiState, u32 ID)
{
    GuiState->Active = ID;
}

internal void
RequestSetHovered(gui_state *GuiState, u32 ID)
{
    GuiState->HoveredNextFrame = ID;
}

internal bool32
MouseInside(gui_state *GuiState, gui_rect Rect)
{
    s32 Left = Rect.X;
    s32 Top = Rect.Y;
    s32 Right = Left + SafeCastToS32(Rect.W);
    s32 Bottom = Top + SafeCastToS32(Rect.H);

    if (Left <= GuiState->Input.MouseX && GuiState->Input.MouseX < Right && 
        Top <= GuiState->Input.MouseY && GuiState->Input.MouseY < Bottom)
    {
        return true;
    }
    else
    {
        return false;
    }
}

internal gui_draw_request *
NextDrawRequest(gui_state *GuiState)
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

internal bool32
GuiIsAnyElementActive(gui_state *GuiState)
{
    bool32 Result = GuiState->Active > 0;
    return Result;
}

internal bool32
GuiBehaviourButton(gui_state *GuiState, u32 ID, gui_rect Rect)
{
    bool32 Result = false;

    if (IsActive(GuiState, ID))
    {
        if (GuiState->Input.LMBWentUp)
        {
            if (IsHovered(GuiState, ID))
            {
                Result = true;
            }
            SetActive(GuiState, 0);
        }
    }
    else if (IsHovered(GuiState, ID))
    {
        if (GuiState->Input.LMBWentDown)
        {
            SetActive(GuiState, ID);
        }
    }

    if (MouseInside(GuiState, Rect))
    {
        RequestSetHovered(GuiState, ID);
    }

    return Result;
}

internal void
GuiPushScissor(gui_state *GuiState, gui_rect Scissor)
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
            s32 IntersectionMaxX = Minimum(LastScissor.X+SafeCastToS32(LastScissor.W), 
                                           Scissor.X+SafeCastToS32(Scissor.W));
            s32 IntersectionMaxY = Minimum(LastScissor.Y+SafeCastToS32(LastScissor.H), 
                                           Scissor.Y+SafeCastToS32(Scissor.H));
            if (IntersectionMinX < IntersectionMaxX &&
                IntersectionMinY < IntersectionMaxY)
            {
                NewScissor = { IntersectionMinX, IntersectionMinY,
                               SafeCastToU32(IntersectionMaxX-IntersectionMinX),
                               SafeCastToU32(IntersectionMaxY-IntersectionMinY) };
            }
            else
            {
                NewScissor = { 0, 0, 0, 0 };
            }
        }

        GuiState->ScissorStack[GuiState->ScissorStackCount++] = NewScissor;
    }
}

internal void
GuiPopScissor(gui_state *GuiState)
{
    --GuiState->ScissorStackCount;
}

internal void
GuiPanel(gui_state *GuiState, gui_rect Rect, vec4 RGBA)
{
    gui_draw_request *DrawRequest = NextDrawRequest(GuiState);
    if (DrawRequest)
    {
        DrawRequest->DrawType = GuiDrawRect;
        DrawRequest->Color = RGBA;
        DrawRequest->Rect = Rect;
        DrawRequest->Texture = InvalidHandle;
    }
}

internal void
GuiPanel(gui_state *GuiState, gui_rect Rect, texture_handle Image)
{
    gui_draw_request *DrawRequest = NextDrawRequest(GuiState);
    if (DrawRequest)
    {
        DrawRequest->DrawType = GuiDrawRect;
        DrawRequest->Color = vec4(1.0f,1.0f,1.0f,1.0f);
        DrawRequest->Rect = Rect;
        DrawRequest->Texture = Image;
    }
}

internal bool32
GuiButton(gui_state *GuiState, u32 ID, gui_rect Rect, vec4 RGBA, 
          vec4 RGBAHovered, vec4 RGBAActive)
{
    bool32 Result = GuiBehaviourButton(GuiState, ID, Rect);

    if (IsActive(GuiState, ID))
    {
        GuiPanel(GuiState, Rect, RGBAActive);
    }
    else if (IsHovered(GuiState, ID))
    {
        GuiPanel(GuiState, Rect, RGBAHovered);
    }
    else
    {
        GuiPanel(GuiState, Rect, RGBA);
    }

    return Result;
}

internal bool32
GuiButton(gui_state *GuiState, u32 ID, gui_rect Rect, texture_handle Image, 
          texture_handle ImageHovered, texture_handle ImageActive)
{
    bool32 Result = GuiBehaviourButton(GuiState, ID, Rect);

    if (IsActive(GuiState, ID))
    {
        GuiPanel(GuiState, Rect, ImageActive);
    }
    else if (IsHovered(GuiState, ID))
    {
        GuiPanel(GuiState, Rect, ImageHovered);
    }
    else
    {
        GuiPanel(GuiState, Rect, Image);
    }

    return Result;
}

internal void
GuiTextFmt(gui_state *GuiState, s32 X, s32 Y, gui_font FontIndex, float TextScale, gui_alignment Align,
           bool32 PutNewLinesAbove, vec4 TextColor, char *TextFmt, ...)
{
    if (TextFmt)
    {
        char *CurrentAddress = (char *)CurrentPtr(GuiState->Scratch);

        va_list ArgPtr;
        va_start(ArgPtr, TextFmt);
        int TextLength = stbsp_vsprintf(CurrentAddress, TextFmt, ArgPtr);
        va_end(ArgPtr);

        PushArray(GuiState->Scratch, TextLength+1, char); // to move the used index

        GuiText(GuiState, X, Y, FontIndex, TextScale, Align, PutNewLinesAbove, TextColor, CurrentAddress);
    }
}

internal void
GuiText(gui_state *GuiState, s32 X, s32 Y, gui_font FontIndex, float TextScale, gui_alignment Align,
        bool32 PutNewLinesAbove, vec4 TextColor, char *Text)
{
    if (Text == NULL)
    {
        return;
    }

    gui_draw_request *DrawRequest = NextDrawRequest(GuiState);
    if (DrawRequest)
    {
        int TextLength = StringLength(Text);

        char *String = (char *)CurrentPtr(GuiState->Scratch);
        if (String != Text)
        {
            // NOTE(Kevin): If string is at same location as CurrentAddress, then we only need
            // to move the used index by the text length. No need for copy.
            String = PushString(GuiState->Scratch, Text);
        }

        DrawRequest->DrawType = GuiDrawText;
        DrawRequest->Color = TextColor;
        DrawRequest->Text = String;
        DrawRequest->X = X;
        DrawRequest->Y = Y;
        DrawRequest->TextScale = TextScale;
        DrawRequest->Font = GuiState->Fonts[FontIndex].Handle;
        DrawRequest->Align = Align;
        DrawRequest->PutNewLinesAbove = PutNewLinesAbove;
    }
}

internal void
GuiBeginFrame(gui_state *GuiState, u32 CanvasWidth, u32 CanvasHeight,
              memory_arena *Scratch, kbm_input GuiInput)
{
    GuiState->Scratch = Scratch;
    GuiState->DrawRequests = PushArray(GuiState->Scratch, MAX_GUI_DRAW_REQUESTS, gui_draw_request);
    GuiState->DrawRequestsCount = 0;
    GuiState->ScissorStack = PushArray(GuiState->Scratch, MAX_SCISSOR_STACK, gui_rect);
    GuiState->ScissorStackCount = 0;

    GuiPushScissor(GuiState, { 0, 0, CanvasWidth, CanvasHeight });

    GuiState->Input = GuiInput;

    GuiState->Hovered = GuiState->HoveredNextFrame;
    GuiState->HoveredNextFrame = 0;
}

internal void
SetGuiTooltip(gui_state *GuiState, char *Tooltip)
{
    GuiState->TooltipLength = StringLength(Tooltip);
    Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
}
