#include "shooter_gui.h"
#include "stb_sprintf.h"

internal bool32 IsActive(gui_state *GuiState, u32 ID)
{
    bool32 Result = GuiState->InteractionState.Active == ID;
    return Result;
}

internal bool32 IsHovered(gui_state *GuiState, u32 ID)
{
    bool32 Result = GuiState->InteractionState.Hovered == ID;
    return Result;
}

internal void SetActive(gui_state *GuiState, u32 ID)
{
    GuiState->InteractionState.Active = ID;
}

internal void RequestSetHovered(gui_state *GuiState, u32 ID)
{
    GuiState->InteractionState.HoveredNextFrame = ID;
}

internal bool32 MouseInside(gui_state *GuiState, gui_rect Rect)
{
    s32 Left = Rect.X;
    s32 Top = Rect.Y;
    s32 Right = Left + SafeCastToS32(Rect.W);
    s32 Bottom = Top + SafeCastToS32(Rect.H);

    if (Left <= GuiState->MouseX && GuiState->MouseX < Right && 
        Top <= GuiState->MouseY && GuiState->MouseY < Bottom)
    {
        return true;
    }
    else
    {
        return false;
    }
}

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

bool32 GuiIsAnyElementActive(gui_state *GuiState)
{
    bool32 Result = GuiState->InteractionState.Active > 0;
    return Result;
}

bool32 GuiBehaviourButton(gui_state *GuiState, u32 ID, gui_rect Rect)
{
    bool32 Result = false;

    if (IsActive(GuiState, ID))
    {
        if (GuiState->LMBWentUp)
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
        if (GuiState->LMBWentDown)
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

void GuiPopScissor(gui_state *GuiState)
{
    --GuiState->ScissorStackCount;
}

void GuiPanel(gui_state *GuiState, gui_rect Rect, vec4 RGBA)
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

void GuiPanel(gui_state *GuiState, gui_rect Rect, texture_handle Image)
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

bool32 GuiButton(gui_state *GuiState, u32 ID, gui_rect Rect, vec4 RGBA, 
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

bool32 GuiButton(gui_state *GuiState, u32 ID, gui_rect Rect, texture_handle Image, 
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

void GuiTextFmt(gui_state *GuiState, s32 X, s32 Y, u32 FontSize, gui_font FontIndex, gui_alignment Align,
                bool32 PutNewLinesAbove, vec4 TextColor, char *TextFmt, ...)
{
    if (TextFmt)
    {
        char *CurrentAddress = (char *)ArenaCurrentAddress(GuiState->Scratch);

        va_list ArgPtr;
        va_start(ArgPtr, TextFmt);
        int TextLength = vsprintf(CurrentAddress, TextFmt, ArgPtr);
        va_end(ArgPtr);

        GuiText(GuiState, X, Y, FontSize, FontIndex, Align, PutNewLinesAbove, TextColor, CurrentAddress);
    }
}

void GuiText(gui_state *GuiState, s32 X, s32 Y, u32 FontSize, gui_font FontIndex, gui_alignment Align,
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

        DrawRequest->DrawType = GuiDrawText;
        DrawRequest->Color = TextColor;
        DrawRequest->Text = CurrentAddress;
        DrawRequest->X = X;
        DrawRequest->Y = Y;
        DrawRequest->FontSize = FontSize;
        DrawRequest->Font = GuiState->Fonts[FontIndex];
        DrawRequest->Align = Align;
        DrawRequest->PutNewLinesAbove = PutNewLinesAbove;
    }
}

void GuiBeginFrame(gui_state *GuiState, u32 CanvasWidth, u32 CanvasHeight, memory_arena *Scratch,
                   s32 MouseX, s32 MouseY, game_button_state LMB, game_button_state RMB)
{
    GuiState->Scratch = Scratch;

    GuiState->CallOrder = 0;

    GuiState->DrawRequestsCount = 0;
    GuiState->ScissorStackCount = 0;
    GuiPushScissor(GuiState, { 0, 0, CanvasWidth, CanvasHeight });

    GuiState->MouseX = MouseX;
    GuiState->MouseY = MouseY;
    // NOTE(Kevin): Ignoring multiple clicks between frames
    GuiState->LMBWentDown = (LMB.EndedDown && LMB.HalfTransitionCount > 0);
    GuiState->LMBWentUp = (!LMB.EndedDown && LMB.HalfTransitionCount > 0);
    GuiState->RMBWentDown = (RMB.EndedDown && RMB.HalfTransitionCount > 0);
    GuiState->RMBWentUp = (!RMB.EndedDown && RMB.HalfTransitionCount > 0);

    GuiState->InteractionState.Hovered = GuiState->InteractionState.HoveredNextFrame;
    GuiState->InteractionState.HoveredNextFrame = 0;
}
