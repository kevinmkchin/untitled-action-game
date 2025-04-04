#include "gui.h"


#define ISANYOF1(a, x) ((a) == (x))
#define ISANYOF2(a, x, y) ((a) == (x) || (a) == (y))
#define ISANYOF3(a, x, y, z) ((a) == (x) || (a) == (y) || (a) == (z))
#define ISANYOF4(a, x, y, z, w) ((a) == (x) || (a) == (y) || (a) == (z) || (a) == (w))



static ivec2 lastAddedElementDimension;
static bool flag_HorizontalMode = false;

static ui_id freshIdCounter = 0;
static ui_id FreshID()
{
    return ++freshIdCounter;
}

namespace GUI
{
    vec4 style_buttonNormalColor = vec4(0.18f, 0.18f, 0.18f, 1.f);
    vec4 style_buttonHoveredColor = vec4(0.08f, 0.08f, 0.08f, 1.f);
    vec4 style_buttonActiveColor = vec4(0.06f, 0.06f, 0.06f, 1.f);
    Font style_textFont;
    vec4 style_textColor = vec4(1.f, 1.f, 1.f, 1.0f);
    int style_paddingTop = 1;
    int style_paddingBottom = 1;
    int style_paddingLeft = 1;
    int style_paddingRight = 1;
    vec4 style_editorWindowBackgroundColor = vec4(0.1f, 0.1f, 0.1f, 0.85f);

    int GetFontSize()
    {
        if (style_textFont.ptr == nullptr)
        {
            LogError("GUI::style_textFont has null vtxt_font. Failed to retrieve font size.");
            return 0;
        }
        return style_textFont.ptr->font_height_px;
    }

    static c_array<vtxt_font, 10> s_vtxtLoadedFonts;
    static Font s_Fonts[32];
    static Font s_DefaultFont;

    Font FontCreateFromTTFFile(const std::string& fontFilePath, u8 fontSize, bool useNearestFiltering)
    {
        Font fontToReturn;
        vtxt_font fontHandle;

        BinaryFileHandle fontfile;
        ReadFileBinary(fontfile, fontFilePath.c_str());
        ASSERT(fontfile.memory);
        vtxt_init_font(&fontHandle, (u8*)fontfile.memory, fontSize);
        FreeFileBinary(fontfile);

        // load texture atlas into memory, then vram, then free from memory
        GPUTexture fontTexture;
        CreateGPUTextureFromBitmap(&fontTexture, fontHandle.font_atlas.pixels, fontHandle.font_atlas.width,
                fontHandle.font_atlas.height, GL_RED, GL_RED, (useNearestFiltering ? GL_NEAREST : GL_LINEAR));
        fontToReturn.textureId = fontTexture.id;
        free(fontHandle.font_atlas.pixels);

        s_vtxtLoadedFonts.put(fontHandle);
        fontToReturn.ptr = &s_vtxtLoadedFonts.back();
        return fontToReturn;
    }

    // 16 x 16 font layout bitmap!
    Font FontCreateFromBitmap(GPUTexture bitmapTexture, int desiredLineGapInPixels)
    {
        const int bitmapW = bitmapTexture.width;
        const int bitmapH = bitmapTexture.height;
        const int glyphW = bitmapW / 16;
        const int glyphH = bitmapH / 16;

        Font bitmapFont;
        bitmapFont.textureId = bitmapTexture.id;

        vtxt_font fontHandle;
        fontHandle.font_height_px = glyphH;
        fontHandle.ascender = float(glyphH);
        fontHandle.descender = 0;
        fontHandle.linegap = (float)desiredLineGapInPixels;
        fontHandle.font_atlas.width = bitmapW;
        fontHandle.font_atlas.height = bitmapH;

        for(int codepoint = 0; codepoint < 256; ++codepoint)
        {
            vtxt_glyph *glyph = &fontHandle.glyphs[codepoint];
            glyph->codepoint = codepoint;
            glyph->width = float(glyphW);
            glyph->height = float(glyphH);
            glyph->advance = float(glyphW);
            glyph->offset_x = 0;
            glyph->offset_y = float(-glyphH);
            glyph->min_u = float(codepoint % 16 * glyphW) / bitmapW;
            glyph->min_v = 1.f - (float((codepoint / 16 + 1) * glyphH) / bitmapH);
            glyph->max_u = float((codepoint % 16 + 1) * glyphW) / bitmapW;
            glyph->max_v = 1.f - (float(codepoint / 16 * glyphH) / bitmapH);
        }

        s_vtxtLoadedFonts.put(fontHandle);
        bitmapFont.ptr = &s_vtxtLoadedFonts.back();
        return bitmapFont;
    }

    UIRect::UIRect(struct ALH *layout) : x(layout->x), y(layout->y), w(layout->w), h(layout->h) {};

    static char __reservedTextMemory[4000000];
    static u32 __reservedTextMemoryIndexer = 0;
    c_array<SDL_Keycode, 32> keyboardInputASCIIKeycodeThisFrame;
    static c_array<char, 128> activeTextInputBuffer;

    static ui_id hoveredUI = null_ui_id;
    static ui_id activeUI = null_ui_id;
    bool anyElementHovered = false;
    bool anyElementActive = false;
    bool anyWindowHovered = false;

    static linear_arena_t drawRequestsFrameStorageBuffer;
#define MESAIMGUI_NEW_DRAW_REQUEST(type) new (drawRequestsFrameStorageBuffer.Alloc<type>()) type()


    struct WindowData
    {
        // data for layouting within windows and grid items
        ui_id zoneId = null_ui_id;
        UIRect zoneRect;
        int paddingFromLeft = 0;
        int topLeftXOffset = 0;
        int topLeftYOffset = 0;
    };
    static std::stack<WindowData> WINDOWSTACK;
    static WindowData gridItemData;
    WindowData *CurrentWindowOrGridItem()
    {
        if (gridItemData.zoneId != null_ui_id)
            return &gridItemData;

        ASSERT(!WINDOWSTACK.empty());
        return &WINDOWSTACK.top();
    }


    bool IsActive(ui_id id)
    {
        return activeUI == id;
    }

    bool IsHovered(ui_id id)
    {
        return hoveredUI == id;
    }

    void SetActive(ui_id id)
    {
        activeUI = id;

        if (id != null_ui_id) // if id == null_ui_id, set to false at beginning of next frame
            anyElementActive = true;
    }

    static c_array<u64, 16> hoveredThisFrame;
    void RequestSetHovered(ui_id id)
    {
        anyElementHovered = true;

        ASSERT((0xff000000 & id) == 0x00000000);
        u64 idWithEmbeddedDepth = (u64() << 48) | id;
        hoveredThisFrame.put(idWithEmbeddedDepth);
        // Actual setting of hoveredUI should be processed at NewFrame IMO
        // add to array with depth info and process later
    }

    int MouseXInGUI = 0;
    int MouseYInGUI = 0;

    bool MouseWentUp()
    {
        return MouseReleased & SDL_BUTTON_MASK(SDL_BUTTON_LEFT);
    }

    bool MouseWentDown()
    {
        return MousePressed & SDL_BUTTON_MASK(SDL_BUTTON_LEFT);
    }

    bool MouseInside(const UIRect& rect)
    {
        int left = rect.x;
        int top = rect.y;
        int right = left + rect.w;
        int bottom = top + rect.h;
        if (left <= MouseXInGUI && MouseXInGUI < right
            && top <= MouseYInGUI && MouseYInGUI < bottom)
        {
            return true;
        }
        else
        {
            return false;
        }
    }




    bool Behaviour_Button(ui_id id, UIRect rect)
    {
        bool result = false;

        if(IsActive(id))
        {
            if(MouseWentUp())
            {
                if(IsHovered(id))
                {
                    result = true;
                }
                SetActive(null_ui_id);
            }
        }
        else if(IsHovered(id))
        {
            if(MouseWentDown())
            {
                SetActive(id);
            }
        }

        if(MouseInside(rect))
        {
            RequestSetHovered(id);
        }

        return result;
    }

    vec3 CodeCharIndexToColor[32000];
    void PipCode(int x, int y, int size, const char* text)
    {
        if (text == NULL) return;

#if INTERNAL_BUILD
        // Giving 500,000 bytes of free space in the frame text buffer to be safe
        if (__reservedTextMemoryIndexer >= ARRAY_COUNT(__reservedTextMemory) - 500000)
        {
            LogWarning("Attempting to draw text in GUI.cpp but not enough reserved memory to store text.");
            return;
        }
#endif

        char* textBuffer = __reservedTextMemory + __reservedTextMemoryIndexer;
        strcpy(textBuffer, text);
        int numCharactersWritten = (int)strlen(text);
        __reservedTextMemoryIndexer += numCharactersWritten + 1;

        PipCodeDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(PipCodeDrawRequest);
        drawRequest->text = textBuffer;
        drawRequest->size = size;
        drawRequest->x = x;
        drawRequest->y = y;
        drawRequest->font = style_textFont;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }


    bool ImageButton(UIRect rect, u32 normalTexId, u32 hoveredTexId, u32 activeTexId)
    {
        ui_id id = FreshID();
        bool result = Behaviour_Button(id, rect);

        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->textureId = IsHovered(id) ? hoveredTexId : normalTexId;
        if (IsActive(id) || result) drawRequest->textureId = activeTexId;

        AppendToCurrentDrawRequestsCollection(drawRequest);

        return result;
    }


    bool PrimitiveButton(ui_id id, UIRect rect, vec4 normalColor, vec4 hoveredColor, vec4 activeColor, bool activeColorOnClickReleaseFrame)
    {
        bool result = Behaviour_Button(id, rect);

        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = IsHovered(id) ? hoveredColor : normalColor;
        if (IsActive(id)) drawRequest->color = activeColor;
        if (result && activeColorOnClickReleaseFrame)
        {
            drawRequest->color = activeColor;
        }

        AppendToCurrentDrawRequestsCollection(drawRequest);

        return result;
    }

    void PrimitivePanel(UIRect rect, vec4 colorRGBA)
    {
        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = colorRGBA;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitivePanel(UIRect rect, int cornerRadius, vec4 colorRGBA)
    {
        RoundedCornerRectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RoundedCornerRectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = colorRGBA;
        drawRequest->radius = cornerRadius;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitivePanel(UIRect rect, u32 glTextureId)
    {
        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = vec4(1.f, 0.f, 1.f, 1.f);
        drawRequest->textureId = glTextureId;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitivePanel(UIRect rect, int cornerRadius, u32 glTextureId, float normalizedCornerSizeInUV)
    {
        CorneredRectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(CorneredRectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = vec4(1.f, 0.f, 1.f, 1.f);
        drawRequest->textureId = glTextureId;
        drawRequest->radius = cornerRadius;
        drawRequest->normalizedCornerSizeInUV = normalizedCornerSizeInUV;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitiveTextFmt(int x, int y, int size, Align alignment, const char* textFmt, ...)
    {
        if (textFmt == NULL) return;

#if INTERNAL_BUILD
        // Giving 500,000 bytes of free space in the frame text buffer to be safe
        if (__reservedTextMemoryIndexer >= ARRAY_COUNT(__reservedTextMemory) - 500000)
        {
            LogWarning("Attempting to draw text in GUI.cpp but not enough reserved memory to store text.");
            return;
        }
#endif

        va_list argptr;
        char* formattedTextBuffer = __reservedTextMemory + __reservedTextMemoryIndexer;
        va_start(argptr, textFmt);
        int numCharactersWritten = stbsp_vsprintf(formattedTextBuffer, textFmt, argptr);
        va_end(argptr);
        __reservedTextMemoryIndexer += numCharactersWritten + 1;

        TextDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(TextDrawRequest);
        drawRequest->text = formattedTextBuffer;
        drawRequest->size = size;
        drawRequest->x = x;
        drawRequest->y = y;
        drawRequest->alignment = alignment;
        drawRequest->font = style_textFont;
        drawRequest->color = style_textColor;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitiveText(int x, int y, int size, Align alignment, const char* text)
    {
        if (text == NULL) return;

#if INTERNAL_BUILD
        // Giving 500,000 bytes of free space in the frame text buffer to be safe
        if (__reservedTextMemoryIndexer >= ARRAY_COUNT(__reservedTextMemory) - 500000)
        {
            LogWarning("Attempting to draw text in GUI.cpp but not enough reserved memory to store text.");
            return;
        }
#endif

        char* textBuffer = __reservedTextMemory + __reservedTextMemoryIndexer;
        strcpy(textBuffer, text);
        int numCharactersWritten = (int)strlen(text);
        __reservedTextMemoryIndexer += numCharactersWritten + 1;

        TextDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(TextDrawRequest);
        drawRequest->text = textBuffer;
        drawRequest->size = size;
        drawRequest->x = x;
        drawRequest->y = y;
        drawRequest->alignment = alignment;
        drawRequest->font = style_textFont;
        drawRequest->color = style_textColor;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitiveTextMasked(int x, int y, int size, Align alignment, const char* text, UIRect mask, int maskCornerRadius)
    {
        if (text == NULL) return;

#if INTERNAL_BUILD
        // Giving 500,000 bytes of free space in the frame text buffer to be safe
        if (__reservedTextMemoryIndexer >= ARRAY_COUNT(__reservedTextMemory) - 500000)
        {
            LogWarning("Attempting to draw text in GUI.cpp but not enough reserved memory to store text.");
            return;
        }
#endif

        char* textBuffer = __reservedTextMemory + __reservedTextMemoryIndexer;
        strcpy(textBuffer, text);
        int numCharactersWritten = (int)strlen(text);
        __reservedTextMemoryIndexer += numCharactersWritten + 1;

        TextDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(TextDrawRequest);
        drawRequest->text = textBuffer;
        drawRequest->size = size;
        drawRequest->x = x;
        drawRequest->y = y;
        drawRequest->alignment = alignment;
        drawRequest->font = style_textFont;
        drawRequest->color = style_textColor;
        drawRequest->rectMask = mask;
        drawRequest->rectMaskCornerRadius = maskCornerRadius;

        AppendToCurrentDrawRequestsCollection(drawRequest);
    }

    void PrimitiveIntegerInputField(ui_id id, UIRect rect, int* v)
    {
        bool bSetInactiveAndReturnValue = false;

        if (IsActive(id))
        {
            for (int i = 0; i < keyboardInputASCIIKeycodeThisFrame.count; ++i)
            {
                i32 keycodeASCII = keyboardInputASCIIKeycodeThisFrame[i];
                if (48 <= keycodeASCII && keycodeASCII <= 57)
                {
                    if (activeTextInputBuffer.count == 1 && activeTextInputBuffer[0] == '0')
                    {
                        activeTextInputBuffer[0] = char(keycodeASCII);
                    }
                    else if (activeTextInputBuffer.count < 8) // just to prevent integers that are too big
                    {
                        activeTextInputBuffer.put(char(keycodeASCII));
                    }
                }
                else if (keycodeASCII == 45 /* minus sign */ && activeTextInputBuffer.count == 0)
                {
                    activeTextInputBuffer.put(char(keycodeASCII));
                }
                else if (keycodeASCII == SDLK_RETURN)
                {
                    bSetInactiveAndReturnValue = true;
                }
                else if (keycodeASCII == SDLK_BACKSPACE)
                {
                    if (activeTextInputBuffer.count > 0)
                    {
                        activeTextInputBuffer.back() = '\0';
                        --activeTextInputBuffer.count;
                    }
                }
            }

            if (MouseWentDown() && !MouseInside(rect))
            {
                bSetInactiveAndReturnValue = true;
            }
        }
        else if (IsHovered(id))
        {
            if (MouseWentDown())
            {
                std::string intValueAsString = std::to_string(*v);
                activeTextInputBuffer.memset_zero();
                memcpy(activeTextInputBuffer.data, intValueAsString.c_str(), intValueAsString.size());
                activeTextInputBuffer.count = (int)intValueAsString.size();
                SetActive(id);
            }
        }

        if (bSetInactiveAndReturnValue)
        {
            int inputtedInteger = 0;
            bool inputIsNotEmpty = activeTextInputBuffer.count > 0;
            bool onlyInputIsMinusSign = activeTextInputBuffer.count == 1 && activeTextInputBuffer[0] == '-';
            if (inputIsNotEmpty && !onlyInputIsMinusSign)
            {
                inputtedInteger = std::stoi(activeTextInputBuffer.data);
            }
            *v = inputtedInteger;
            activeTextInputBuffer.reset_count();
            activeTextInputBuffer.memset_zero();
            SetActive(null_ui_id);
        }

        if (MouseInside(rect))
        {
            RequestSetHovered(id);
        }

        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = IsActive(id) ? vec4(0.f, 0.f, 0.f, 1.f) : vec4(0.2f, 0.2f, 0.2f, 1.f);//vec4(1.f, 1.f, 1.f, 1.f) : vec4(0.8f, 0.8f, 0.8f, 1.f);

        AppendToCurrentDrawRequestsCollection(drawRequest);

        if (IsActive(id))
        {
            if (activeTextInputBuffer.count > 0)
            {
                PrimitiveText(rect.x + rect.w, rect.y + rect.h, GetFontSize(), Align::RIGHT, activeTextInputBuffer.data);
            }
        }
        else
        {
            PrimitiveText(rect.x + rect.w, rect.y + rect.h, GetFontSize(), Align::RIGHT, std::to_string(*v).c_str());
        }
    }

    void PrimitiveFloatInputField(ui_id id, UIRect rect, float* v)
    {
        bool bSetInactiveAndReturnValue = false;

        if (IsActive(id))
        {
            for (int i = 0; i < keyboardInputASCIIKeycodeThisFrame.count; ++i)
            {
                i32 keycodeASCII = keyboardInputASCIIKeycodeThisFrame[i];
                if (48 <= keycodeASCII && keycodeASCII <= 57)
                {
                    if (activeTextInputBuffer.count == 1 && activeTextInputBuffer[0] == '0')
                    {
                        activeTextInputBuffer[0] = char(keycodeASCII);
                    }
                    else if (activeTextInputBuffer.count < 8) // just to prevent floats that are too big
                    {
                        activeTextInputBuffer.put(char(keycodeASCII));
                    }
                }
                else if (keycodeASCII == 45 /* minus sign */
                         && activeTextInputBuffer.count == 0)
                {
                    activeTextInputBuffer.put(char(keycodeASCII));
                }
                else if (keycodeASCII == 46 /* decimal point */
                         && !IsOneOfArray('.', activeTextInputBuffer.data, activeTextInputBuffer.count))
                {
                    activeTextInputBuffer.put(char(keycodeASCII));
                }
                else if (keycodeASCII == SDLK_RETURN)
                {
                    bSetInactiveAndReturnValue = true;
                }
                else if (keycodeASCII == SDLK_BACKSPACE)
                {
                    if (activeTextInputBuffer.count > 0)
                    {
                        activeTextInputBuffer.back() = '\0';
                        --activeTextInputBuffer.count;
                    }
                }
            }

            if (MouseWentDown() && !MouseInside(rect))
            {
                bSetInactiveAndReturnValue = true;
            }
        }
        else if (IsHovered(id))
        {
            if (MouseWentDown())
            {
                std::string floatValueAsString = std::to_string(*v);
                RemoveCharactersFromEndOfString(floatValueAsString, '0');
                if (floatValueAsString.back() == '.') floatValueAsString.push_back('0');
                activeTextInputBuffer.memset_zero();
                memcpy(activeTextInputBuffer.data, floatValueAsString.c_str(), floatValueAsString.size());
                activeTextInputBuffer.count = (int)floatValueAsString.size();
                SetActive(id);
            }
        }

        if (bSetInactiveAndReturnValue)
        {
            float inputtedFloat = 0;
            bool inputIsNotEmpty = activeTextInputBuffer.count > 0;
            bool onlyInputIsMinusSignOrDot = activeTextInputBuffer.count == 1
                                             && ISANYOF2(activeTextInputBuffer[0], '-', '.');
            if (inputIsNotEmpty && !onlyInputIsMinusSignOrDot)
            {
                inputtedFloat = std::stof(activeTextInputBuffer.data);
            }
            *v = inputtedFloat;
            activeTextInputBuffer.reset_count();
            activeTextInputBuffer.memset_zero();
            SetActive(null_ui_id);
        }

        if (MouseInside(rect))
        {
            RequestSetHovered(id);
        }

        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = IsActive(id) ? vec4(0.f, 0.f, 0.f, 1.f) : vec4(0.2f, 0.2f, 0.2f, 1.f);

        AppendToCurrentDrawRequestsCollection(drawRequest);

        if (IsActive(id))
        {
            if (activeTextInputBuffer.count > 0)
            {
                PrimitiveText(rect.x + rect.w, rect.y + rect.h, GetFontSize(), Align::RIGHT, activeTextInputBuffer.data);
            }
        }
        else
        {
            char cbuf[32];
            stbsp_sprintf(cbuf, "%.2f", *v);
            PrimitiveText(rect.x + rect.w, rect.y + rect.h, GetFontSize(), Align::RIGHT, cbuf);
        }
    }

    void PrimitiveCheckbox(ui_id id, UIRect rect, int inset, bool *value, vec4 background, vec4 foreground)
    {
        if (*value)
        {
            *value = !PrimitiveButton(id, rect, background, background, background);
            PrimitivePanel(UIRect(rect.x+inset,rect.y+inset,rect.w-inset*2,rect.h-inset*2), foreground);
        }
        else
        {
            *value = PrimitiveButton(id, rect, background, background, background);
        }
    }

    bool PrimitiveLabelledButton(UIRect rect, const char* label, Align textAlignment)
    {
        int ascenderTextSize = GetFontSize();
        float yTextPaddingRatio = (1.f - (float(ascenderTextSize) / float(rect.h))) / 2.f;
        ivec2 textPadding = ivec2(10, (int) roundf(rect.h * yTextPaddingRatio));
        int textX = rect.x + textPadding.x;
        if (textAlignment == Align::CENTER)
        {
            textX = (int) ((rect.w / 2) + rect.x);
        }
        else if (textAlignment == Align::RIGHT)
        {
            textX = (int) rect.x + rect.w - textPadding.x;
        }

        bool buttonValue = PrimitiveButton(FreshID(), rect, style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor);
        PrimitiveText(textX, rect.y + rect.h - textPadding.y, ascenderTextSize, textAlignment, label);
        return buttonValue;
    }

    void BeginWindow(UIRect windowRect, vec4 bgcolor, int depth)
    {
        if (lastAddedElementDimension != ivec2())
            Window_CommitLastElementDimension();

        if (depth > -1)
            GUIDraw_PushDrawCollection(windowRect, depth);
        else
            GUIDraw_PushDrawCollection(windowRect, (u8)WINDOWSTACK.size());

        WindowData windata;
        windata.zoneId = FreshID();
        windata.zoneRect = windowRect;
        windata.paddingFromLeft = 2;
        windata.topLeftXOffset = windata.paddingFromLeft;
        windata.topLeftYOffset = 2;
        WINDOWSTACK.push(windata);

        PrimitivePanel(windowRect, bgcolor);

        lastAddedElementDimension = ivec2();
    }

    void EndWindow()
    {
        lastAddedElementDimension = ivec2();

        if (!WINDOWSTACK.empty())
        {
            // If no other element is hovered at the end of this window, then check if window is hovered
            if (!anyWindowHovered && MouseInside(WINDOWSTACK.top().zoneRect))
                anyWindowHovered = true;
            // Pop draw requests collection and from window stack
            GUIDraw_PopDrawCollection();
            WINDOWSTACK.pop();
        }
        else
        {
            ASSERT(0);
        }
    }

    void Window_GetWidthHeight(int *w, int *h)
    {
        *w = CurrentWindowOrGridItem()->zoneRect.w;
        *h = CurrentWindowOrGridItem()->zoneRect.h;
    }

    void Window_GetCurrentOffsets(int *x, int *y)
    {
        *x = CurrentWindowOrGridItem()->zoneRect.x + CurrentWindowOrGridItem()->topLeftXOffset;
        *y = CurrentWindowOrGridItem()->zoneRect.y + CurrentWindowOrGridItem()->topLeftYOffset;
    }

    void Window_StageLastElementDimension(int x, int y)
    {
        if (flag_HorizontalMode)
            lastAddedElementDimension = ivec2(x, GM_max(lastAddedElementDimension.y, y));
        else
            lastAddedElementDimension = ivec2(GM_max(lastAddedElementDimension.x, x), y);
    }

    void Window_CommitLastElementDimension()
    {
        if (flag_HorizontalMode)
        {
            CurrentWindowOrGridItem()->topLeftXOffset += lastAddedElementDimension.x;
        }
        else
        {
            CurrentWindowOrGridItem()->topLeftXOffset = CurrentWindowOrGridItem()->paddingFromLeft;
            CurrentWindowOrGridItem()->topLeftYOffset += lastAddedElementDimension.y;
        }
    }

    void EditorText(const char* text)
    {
        Window_CommitLastElementDimension();

        int x;
        int y;
        Window_GetCurrentOffsets(&x, &y);

        int sz = GetFontSize();

        PrimitiveText(x + style_paddingLeft, y + sz + style_paddingTop, sz, Align::LEFT, text);

        float textW;
        float textH;
        vtxt_get_text_bounding_box_info(&textW, &textH, text, style_textFont.ptr, sz);
        Window_StageLastElementDimension(style_paddingLeft + (int)textW + style_paddingRight, 
            sz + style_paddingTop + style_paddingBottom);
    }

    void EditorSpacer(int x, int y)
    {
        Window_CommitLastElementDimension();
        Window_StageLastElementDimension(x, y);
    }

    void EditorImage(u32 glTextureId, ivec2 size)
    {
        Window_CommitLastElementDimension();

        int x;
        int y;
        Window_GetCurrentOffsets(&x, &y);
        x += style_paddingLeft;
        y += style_paddingTop;
        UIRect rect = UIRect(x, y, size.x, size.y);

        PrimitivePanel(rect, glTextureId);

        Window_StageLastElementDimension(style_paddingLeft + size.x + style_paddingRight, 
                                         style_paddingTop + size.y + style_paddingBottom);
    }

    bool EditorImageButton(u32 glTextureId, ivec2 size)
    {
        EditorImage(glTextureId, size);

        int x;
        int y;
        Window_GetCurrentOffsets(&x, &y);
        x += style_paddingLeft;
        y += style_paddingTop;
        UIRect rect = UIRect(x, y, size.x, size.y);

        ui_id buttonId = FreshID();
        bool result = Behaviour_Button(buttonId, rect);

        if (IsActive(buttonId))
            PrimitivePanel(rect, vec4(1.f,1.f,1.f,0.4f));
        else if (IsHovered(buttonId))
            PrimitivePanel(rect, vec4(1.f,1.f,1.f,0.2f));

        return result;
    }

    bool EditorLabelledButton(const char* label, int minwidth)
    {
        Window_CommitLastElementDimension();

        int labelTextSize = GetFontSize();
        float textW;
        float textH;
        vtxt_get_text_bounding_box_info(&textW, &textH, label, style_textFont.ptr, labelTextSize);

        int buttonX;
        int buttonY;
        Window_GetCurrentOffsets(&buttonX, &buttonY);
        buttonX += style_paddingLeft;
        buttonY += style_paddingTop;
        int buttonW = GM_max((int) textW + 4, minwidth);
        int buttonH = labelTextSize + 4;

        UIRect buttonRect = UIRect(buttonX, buttonY, buttonW, buttonH);
        bool result = PrimitiveLabelledButton(buttonRect, label, Align::CENTER);

        Window_StageLastElementDimension(style_paddingLeft + buttonW + style_paddingRight, 
            style_paddingTop + buttonH + style_paddingBottom);

        return result;
    }

    void EditorIncrementableIntegerField(const char* label, int* v, int increment)
    {
        Window_CommitLastElementDimension();

        int x;
        int y;
        Window_GetCurrentOffsets(&x, &y);

        int w = 50;
        int h = GetFontSize() + 4;

        PrimitivePanel(UIRect(x, y, w-2, h), vec4(0.4f, 0.4f, 0.4f, 1.f));
        PrimitiveIntegerInputField(FreshID(), UIRect(x + 1, y + 1, w-4, h - 2), v);
        if (PrimitiveButton(FreshID(), UIRect(x + w, y + 1, 10, (h / 2) - 1), style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor))
        {
            (*v) += increment;
        }
        if (PrimitiveButton(FreshID(), UIRect(x + w, y + (h / 2) + 1, 10, (h / 2) - 1), style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor))
        {
            (*v) -= increment;
        }
        PrimitiveText(x + w + 12, y + h, GetFontSize(), Align::LEFT, label);

        // TODO stage width
        Window_StageLastElementDimension(0, style_paddingTop + h + style_paddingBottom);
    }

    void EditorIncrementableFloatField(const char* label, float* v, float increment)
    {
        Window_CommitLastElementDimension();

        int x, y;
        Window_GetCurrentOffsets(&x, &y);

        int w = 50;
        int h = GetFontSize()+4;

        PrimitivePanel(UIRect(x, y, w-2, h), vec4(0.4f, 0.4f, 0.4f, 1.f));
        PrimitiveFloatInputField(FreshID(), UIRect(x + 1, y + 1, w-4, h - 2), v);
        if (PrimitiveButton(FreshID(), UIRect(x + w, y + 1, 10, (h / 2) - 1), style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor))
        {
            (*v) += increment;
        }
        if (PrimitiveButton(FreshID(), UIRect(x + w, y + (h / 2) + 1, 10, (h / 2) - 1), style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor))
        {
            (*v) -= increment;
        }
        PrimitiveText(x + w + 12, y + h, GetFontSize(), Align::LEFT, label);

        // TODO stage width
        Window_StageLastElementDimension(0, style_paddingTop + h + style_paddingBottom);
    }

    void EditorCheckbox(const char *label, bool *value)
    {
        Window_CommitLastElementDimension();
        int x, y;
        Window_GetCurrentOffsets(&x, &y);
        int w, h;
        w = style_paddingLeft;
        h = style_paddingTop;

        // Element 0: checkbox
        PrimitiveCheckbox(FreshID(), UIRect(x+w,y+h,12,12), 2, 
            value, vec4(0.9f,0.9f,0.9f,1), vec4(0,0,0,1));
        w += 12 + style_paddingRight;
        h += 12 + style_paddingBottom;

        // Element 1: text
        w += style_paddingLeft;
        int sz = GetFontSize();
        PrimitiveText(x+w, y+h-style_paddingBottom, sz, Align::LEFT, label);
        float textW;
        float textH;
        vtxt_get_text_bounding_box_info(&textW, &textH, label, style_textFont.ptr, sz);
        w += int(textW) + style_paddingRight;

        Window_StageLastElementDimension(w, h);
    }

    bool EditorSelectableRect(vec4 colorRGBA, bool *selected, int id)
    {
        Window_CommitLastElementDimension();
        int x, y;
        Window_GetCurrentOffsets(&x, &y);

        PrimitivePanel(UIRect(x,y,12,12), colorRGBA);

        Window_StageLastElementDimension(12, 12);

        return Behaviour_Button(id, UIRect(x,y,12,12));
    }

    bool EditorSelectable(const char *label, bool *selected)
    {
        Window_CommitLastElementDimension();
        int x, y;
        Window_GetCurrentOffsets(&x, &y);

        UIRect selectableRegion = UIRect(x, y, 64, 19);
        if (*selected)
        {
            PrimitivePanel(selectableRegion, style_buttonActiveColor);
            PrimitiveText(x + 1, y + 10, GetFontSize(), Align::LEFT, label);
            Window_StageLastElementDimension(selectableRegion.w, selectableRegion.h);
        }
        else
        {
            *selected = PrimitiveButton(FreshID(), selectableRegion, 
                style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor, true);
            PrimitiveText(x + 1, y + 10, GetFontSize(), Align::LEFT, label);
            Window_StageLastElementDimension(selectableRegion.w, selectableRegion.h);
            if (*selected)
            {
                return true;
            }
        }
        return false;
    }

    bool EditorSelectable_2(const char *label, bool *selected)
    {
        Window_CommitLastElementDimension();
        int x, y;
        Window_GetCurrentOffsets(&x, &y);
        x += style_paddingLeft;
        y += style_paddingTop;
        int labelTextSize = GetFontSize();
        int w = 40;
        int h = labelTextSize + 4;

        UIRect selectableRegion = UIRect(x, y, w, h);
        if (*selected)
        {
            PrimitivePanel(selectableRegion, vec4(0.7f,0.7f,0.7f,1.0f));
            PrimitiveText(x + 2, y + h - 2, labelTextSize, Align::LEFT, label);
            Window_StageLastElementDimension(selectableRegion.w, selectableRegion.h);
        }
        else
        {
            *selected = PrimitiveButton(FreshID(), selectableRegion, 
                style_buttonNormalColor, vec4(0.4f,0.4f,0.4f,1.0f), vec4(0.7f,0.7f,0.7f,1.0f), true);
            PrimitiveText(x + 2, y + h - 2, labelTextSize, Align::LEFT, label);
            Window_StageLastElementDimension(selectableRegion.w, selectableRegion.h);
            if (*selected)
            {
                return true;
            }
        }
        return false;
    }

    void EditorBeginListBox()
    {

    }
    void EditorEndListBox()
    {

    }

    void EditorBeginHorizontal()
    {
        Window_CommitLastElementDimension();
        flag_HorizontalMode = true;
        Window_StageLastElementDimension(0, 0);
    }

    void EditorEndHorizontal()
    {
        flag_HorizontalMode = false;
        Window_CommitLastElementDimension();
        Window_StageLastElementDimension(0, 0);
    }

    static int gridWidth = -1;
    static int gridHeight = -1;
    static int gridX = -1;
    static int gridY = -1;
    static int gridBaseX = -1;
    static int gridBaseY = -1;
    static int rowMaxHeight = -1;
    void EditorBeginGrid(int gridwidth, int gridheight)
    {
        Window_CommitLastElementDimension();

        Window_GetCurrentOffsets(&gridBaseX, &gridBaseY);

        gridWidth = gridwidth;
        gridHeight = gridheight;
        gridX = 0;
        gridY = 0;
    }
    void EditorBeginGridItem(int itemw, int itemh)
    {
        lastAddedElementDimension = ivec2();
        gridItemData.zoneId = FreshID();
        gridItemData.paddingFromLeft = 0;
        gridItemData.topLeftXOffset = 0;
        gridItemData.topLeftYOffset = 0;

        if (gridX + itemw > gridWidth)
        {
            gridX = 0;
            gridY += rowMaxHeight;
            rowMaxHeight = itemh;
        }
        else
        {
            if (rowMaxHeight < itemh)
                rowMaxHeight = itemh;
        }

        gridItemData.zoneRect = UIRect(gridBaseX + gridX, gridBaseY + gridY, itemw, itemh);

        gridX += itemw;
    }
    void EditorEndGridItem()
    {
        ASSERT(gridItemData.zoneId != null_ui_id);
        gridItemData.zoneId = null_ui_id;
    }
    void EditorEndGrid()
    {
        Window_StageLastElementDimension(style_paddingLeft + gridWidth + style_paddingRight, 
            style_paddingTop + gridHeight + style_paddingBottom);

        ASSERT(gridItemData.zoneId == null_ui_id);
        gridWidth = -1;
        gridHeight = -1;
        gridX = -1;
        gridY = -1;
    }


    struct SpriteColor
    {
        u8 r = 0;
        u8 g = 0;
        u8 b = 0;
        u8 a = 0;
    };

    struct SpriteImage
    {
        SpriteColor *pixels;
        i32 w;
        i32 h;
    };

    void AllocSpriteImage(SpriteImage *image, i32 w, i32 h, bool white)
    {
        image->w = w;
        image->h = h;
        image->pixels = (SpriteColor*) calloc(w * h, sizeof(SpriteColor));

        if (white)
        {
            for (i32 y = 0; y < h; ++y)
            {
                for (i32 x = 0; x < w; ++x)
                {
                    SpriteColor *p = image->pixels + (image->w*y + x);
                    p->r = 0xff;
                    p->g = 0xff;
                    p->b = 0xff;
                    p->a = 0xff;
                }
            }
        }
    }

    void SpriteImageToGPUTexture(GPUTexture *texture, SpriteImage *image)
    {
        if (texture->id == 0)
        {
            CreateGPUTextureFromBitmap(texture, (unsigned char*)image->pixels, image->w, image->h, GL_RGBA, GL_RGBA, GL_NEAREST);
        }
        else
        {
            UpdateGPUTextureFromBitmap(texture, (unsigned char*)image->pixels, image->w, image->h);
        }
    }

    void SetFramePixelColor(SpriteImage *frame, i32 x, i32 y, SpriteColor color)
    {
        if (x >= frame->w || y >= frame->h || x < 0 || y < 0)
            return;
        *(frame->pixels + (frame->w*y + x)) = color;
    }

    void EditorColorPicker(ui_id id, float *hue, float *saturation, float *value, float *opacity)
    {
        Window_CommitLastElementDimension();
        int x, y;
        Window_GetCurrentOffsets(&x, &y);

        static SpriteImage chromaselector;
        static SpriteImage hueselector;
        static SpriteImage alphaselector;
        static GPUTexture chromaselectorgputex;
        static GPUTexture hueselectorgputex;
        static GPUTexture alphaselectorgputex;
        if (chromaselectorgputex.id == 0)
        {
            AllocSpriteImage(&chromaselector, 61, 96, false);
            SpriteImageToGPUTexture(&chromaselectorgputex, &chromaselector);

            AllocSpriteImage(&hueselector, chromaselector.w, 12, false);
            SpriteImageToGPUTexture(&hueselectorgputex, &hueselector);

            AllocSpriteImage(&alphaselector, chromaselector.w, 12, false);
            SpriteImageToGPUTexture(&alphaselectorgputex, &alphaselector);
        }

        const UIRect chromaselectorrect = UIRect(x, y, chromaselector.w, chromaselector.h);
        const UIRect hueselectorrect = UIRect(x, y + chromaselector.h, hueselector.w, hueselector.h);
        const UIRect alphaselectorrect = UIRect(x, y + chromaselector.h + hueselector.h, alphaselector.w, alphaselector.h);

        if (IsActive(id) && MouseWentUp())
        {
            SetActive(null_ui_id);
        }
        if (MouseWentDown() && IsHovered(id))
        {
            SetActive(id);
        }
        if (MouseInside(chromaselectorrect))
        {
            RequestSetHovered(id);
        }

        // todo(kevin): this is all temporary code to quickly hack
        if (IsActive(id + 1) && MouseWentUp())
        {
            SetActive(null_ui_id);
        }
        if (MouseWentDown() && IsHovered(id + 1))
        {
            SetActive(id + 1);
        }
        if (MouseInside(hueselectorrect))
        {
            RequestSetHovered(id + 1);
        }

        if (IsActive(id + 2) && MouseWentUp())
        {
            SetActive(null_ui_id);
        }
        if (MouseWentDown() && IsHovered(id + 2))
        {
            SetActive(id + 2);
        }
        if (MouseInside(alphaselectorrect))
        {
            RequestSetHovered(id + 2);
        }

        if (IsActive(id))
        {
            i32 chromaSMouseX = GM_clamp(MouseXInGUI - chromaselectorrect.x, 0, chromaselectorrect.w - 1);
            i32 chromaVMouseY = GM_clamp(chromaselectorrect.h - (MouseYInGUI - chromaselectorrect.y) - 1, 0, chromaselectorrect.h - 1);
            *saturation = float(chromaSMouseX) / float (chromaselectorrect.w - 1);
            *value = float(chromaVMouseY) / float (chromaselectorrect.h - 1);
        }
        else if (IsActive(id + 1))
        {
            i32 hueselectormousex = GM_clamp(MouseXInGUI - hueselectorrect.x, 0, hueselectorrect.w - 1);
            *hue = float(hueselectormousex) / float(hueselectorrect.w - 1);
        }
        else if (IsActive(id + 2))
        {
            i32 alphaselectormousex = GM_clamp(MouseXInGUI - alphaselectorrect.x, 0, alphaselectorrect.w - 1);
            *opacity = float(alphaselectormousex) / float(alphaselectorrect.w - 1);
        }

        for (i32 i = 0; i < chromaselector.w; ++i)
        {
            for (i32 j = 0; j < chromaselector.h; ++j)
            {
                float isaturation = float(i) / float(chromaselector.w - 1);
                float ivalue = float(j) / float(chromaselector.h - 1);
                vec3 interprgb = HSVToRGB(*hue, isaturation, ivalue);
                SpriteColor c = {
                        (u8)(255.f * interprgb.x),
                        (u8)(255.f * interprgb.y),
                        (u8)(255.f * interprgb.z),
                        255
                };
                *(chromaselector.pixels + chromaselector.w * j + i) = c;
            }
        }
        SpriteColor selectedchromacirclecolor = {0, 0, 0, 200 };
        if (*value < 0.5f)
            selectedchromacirclecolor = { 255,255,255,200 };
        i32 left = i32(*saturation * float(chromaselector.w)) - 2;
        i32 bottom = i32(*value * float(chromaselector.h)) - 2;
        SetFramePixelColor(&chromaselector, (left + 0), (bottom + 1), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 0), (bottom + 2), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 3), (bottom + 1), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 3), (bottom + 2), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 1), (bottom + 0), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 2), (bottom + 0), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 1), (bottom + 3), selectedchromacirclecolor);
        SetFramePixelColor(&chromaselector, (left + 2), (bottom + 3), selectedchromacirclecolor);
        SpriteImageToGPUTexture(&chromaselectorgputex, &chromaselector);

        for (i32 i = 0; i < hueselector.w; ++i)
        {
            float normalizedhuef = float(i)/float(hueselector.w - 1);
            vec3 irgb = HSVToRGB(normalizedhuef, 1.f, 1.f);
            for (i32 j = 0; j < hueselector.h; ++j)
            {
                SetFramePixelColor(&hueselector, i, j, {
                        u8(irgb.x * 255.f),
                        u8(irgb.y * 255.f),
                        u8(irgb.z * 255.f),
                        255
                });
            }
        }
        i32 selectedhuecirclex = i32(*hue * float(hueselector.w));
        i32 selectedhuecircley = hueselector.h / 2;
        SpriteColor selectedhuecirclecolor = {10, 10, 10, 180 };
        SetFramePixelColor(&hueselector, selectedhuecirclex-2, selectedhuecircley-1, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex-2, selectedhuecircley, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex+1, selectedhuecircley-1, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex+1, selectedhuecircley, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex-1, selectedhuecircley-2, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex, selectedhuecircley-2, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex-1, selectedhuecircley+1, selectedhuecirclecolor);
        SetFramePixelColor(&hueselector, selectedhuecirclex, selectedhuecircley+1, selectedhuecirclecolor);
        SpriteImageToGPUTexture(&hueselectorgputex, &hueselector);

        for (i32 i = 0; i < alphaselector.w; ++i)
        {
            float normalizedalpha = float(i) / float(alphaselector.w - 1);
            for (i32 j = 0; j < alphaselector.h; ++j)
            {
                vec3 alphaselectorbg;
                if ((i % 16) < 8 != j < (alphaselector.h / 2))
                    alphaselectorbg = { 0.75f, 0.75f, 0.75f };
                else
                    alphaselectorbg = { 0.50f, 0.50f, 0.50f };

                vec3 alphaselectorfg = HSVToRGB(*hue, *saturation, *value);

                vec3 alphaselectorfinalcolor = Lerp(alphaselectorbg, alphaselectorfg, normalizedalpha);

                SetFramePixelColor(&alphaselector, i, j, {
                        u8(alphaselectorfinalcolor.x * 255.f),
                        u8(alphaselectorfinalcolor.y * 255.f),
                        u8(alphaselectorfinalcolor.z * 255.f),
                        255
                });
            }
        }
        i32 selectedalphacirclex = i32(*opacity * float(alphaselector.w));
        i32 selectedalphacircley = alphaselector.h / 2;
        SpriteColor selectedalphacirclecolor = {10, 10, 10, 180 };
        SetFramePixelColor(&alphaselector, selectedalphacirclex-2, selectedalphacircley-1, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex-2, selectedalphacircley, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex+1, selectedalphacircley-1, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex+1, selectedalphacircley, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex-1, selectedalphacircley-2, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex, selectedalphacircley-2, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex-1, selectedalphacircley+1, selectedalphacirclecolor);
        SetFramePixelColor(&alphaselector, selectedalphacirclex, selectedalphacircley+1, selectedalphacirclecolor);
        SpriteImageToGPUTexture(&alphaselectorgputex, &alphaselector);

        PrimitivePanel(chromaselectorrect, chromaselectorgputex.id);
        PrimitivePanel(hueselectorrect, hueselectorgputex.id);
        PrimitivePanel(alphaselectorrect, alphaselectorgputex.id);

        Window_StageLastElementDimension(chromaselectorrect.w, chromaselector.h + hueselector.h + alphaselectorrect.h);
    }


    static void UpdateALHContainer(ALH *layout)
    {
        const int lx = layout->x;
        const int ly = layout->y;
        const int lw = layout->w;
        const int lh = layout->h;
        const int lc = layout->Count();

        if (lc == 0) return;

        if (layout->vertical)
        {
            int absHeightSum = 0;
            int elemIgnoredCount = 0;

            for (ALH *child : layout->container)
            {
                if (child->xauto == false || child->yauto == false)
                {
                    ++elemIgnoredCount;
                }
                else if (child->hauto == false)
                {
                    absHeightSum += child->h;
                    ++elemIgnoredCount;
                }
            }

            int elemAutoHeight = (lh - absHeightSum) / (lc - elemIgnoredCount);

            int yPosAccum = ly;
            for (int i = 0; i < lc; ++i)
            {
                ALH *child = layout->container.at(i);
                if (child->xauto == false || child->yauto == false) continue;

                child->x = lx;
                child->y = yPosAccum;
                child->w = child->wauto ? lw : child->w;
                child->h = child->hauto ? elemAutoHeight : child->h;

                yPosAccum += child->h;

                if (i == lc - 1 && yPosAccum < lh)
                    child->h += lh - yPosAccum;
            }
        }
        else
        {
            int absWidthSum = 0;
            int elemIgnoredCount = 0;

            for (ALH *child : layout->container)
            {
                if (child->xauto == false || child->yauto == false)
                {
                    ++elemIgnoredCount;
                }
                else if (child->wauto == false)
                {
                    absWidthSum += child->w;
                    ++elemIgnoredCount;
                }
            }

            int elemAutoWidth = (lw - absWidthSum) / (lc - elemIgnoredCount);

            int xPosAccum = lx;
            for (int i = 0; i < lc; ++i)
            {
                ALH *child = layout->container.at(i);
                if (child->xauto == false || child->yauto == false) continue;

                child->x = xPosAccum;
                child->y = ly;
                child->w = child->wauto ? elemAutoWidth : child->w;
                child->h = child->hauto ? lh : child->h;

                xPosAccum += child->w;

                if (i == lc - 1 && xPosAccum < lw)
                    child->w += lw - xPosAccum;
            }
        }

        for (ALH *child : layout->container)
        {
            UpdateALHContainer(child);
        }
    }

    void UpdateMainCanvasALH(ALH *layout)
    {
        layout->x = 0;
        layout->y = 0;
        layout->w = BackbufferWidth;
        layout->h = BackbufferHeight;
        UpdateALHContainer(layout);
    }

    ALH *NewALH(bool vertical)
    {
        return NewALH(-1, -1, -1, -1, vertical);
    }

    ALH *NewALH(int absX, int absY, int absW, int absH, bool vertical)
    {
        ALH *alh = new ALH();

        alh->x = absX;
        alh->xauto = alh->x < 0;
        alh->y = absY;
        alh->yauto = alh->y < 0;
        alh->w = absW;
        alh->wauto = alh->w < 0;
        alh->h = absH;
        alh->hauto = alh->h < 0;

        alh->vertical = vertical;

        return alh;
    }

    void DeleteALH(ALH *layout)
    {
        // TODO DeleteALH
    }




    void Init()
    {
        drawRequestsFrameStorageBuffer.Init(1000000);

        hoveredUI = null_ui_id;
        activeUI = null_ui_id;

        // __default_font = FontCreateFromTTFFile(data_path("Baskic8.otf"), 32, true);
        // __fonts[1] = FontCreateFromTTFFile(data_path("Baskic8.otf"), 16, true);
        // __fonts[2] = FontCreateFromTTFFile(data_path("EndlessBossBattle.ttf"), 16, true);
        // s_Fonts[5] = FontCreateFromTTFFile(data_path("PressStart2P.ttf"), 16, true);
        // s_Fonts[0] = FontCreateFromTTFFile(data_path("BitFontMaker2Tes.ttf"), 12, true);
        // s_Fonts[1] = FontCreateFromTTFFile(data_path("BitFontMaker2Tes.ttf"), 13, true);
        // s_Fonts[2] = FontCreateFromTTFFile(data_path("BitFontMaker2Tes.ttf"), 14, true);
        // s_Fonts[3] = FontCreateFromTTFFile(data_path("BitFontMaker2Tes.ttf"), 15, true);
        // s_Fonts[4] = FontCreateFromTTFFile(data_path("BitFontMaker2Tes.ttf"), 16, true);

        BitmapHandle bm_curses6x9;
        ReadImage(bm_curses6x9, wd_path("Kevin6x9.png").c_str());
        for (u32 y = 0; y < bm_curses6x9.height; ++y)
        {
            for (u32 x = 0; x < bm_curses6x9.width; ++x)
            {
                unsigned char *bitmapData = (unsigned char *)bm_curses6x9.memory;
                unsigned char *pixelData = bitmapData + (y * 3 * bm_curses6x9.width + x * 3);
                if (pixelData[0] == 255 && pixelData[1] != 255 && pixelData[2] == 255)
                {
                    pixelData[0] = 0;
                    pixelData[1] = 0;
                    pixelData[2] = 0;
                }
                else
                {
                    pixelData[0] = 255;
                    pixelData[1] = 0;
                    pixelData[2] = 0;
                }
            }
        }
        GPUTexture tex_0;
        CreateGPUTextureFromBitmap(&tex_0, (unsigned char *) bm_curses6x9.memory, bm_curses6x9.width, bm_curses6x9.height, GL_RED, GL_RGB);
        s_Fonts[6] = FontCreateFromBitmap(tex_0, 3); // NOTE(Kevin): 2023-12-22 THIS VALUE IS BEING COPIED HARDCODED INTO CODE EDITOR MAKE SURE TO CHANGE AS WELL
        s_DefaultFont = s_Fonts[6];

        style_textFont = s_DefaultFont;

        GUIDraw_InitResources();
    }

    void NewFrame()
    {
        if (!anyElementHovered)
        {
            hoveredUI = null_ui_id;
        }
        else
        {
            // Process array of hovered ui element ids this frame and pick the one with highest depth i.e. closest to screen
            ASSERT(hoveredThisFrame.count > 0);
            u64 highestDepthElement = 0;
            for (int i = 0; i < hoveredThisFrame.count; ++i)
            {
                u64 info = hoveredThisFrame[i];
                if ((0xff000000 & info) >= (0xff000000 & highestDepthElement))
                    highestDepthElement = info;
            }
            hoveredUI = 0x00ffffff & highestDepthElement;
            hoveredThisFrame.reset_count();
        }
        ASSERT(hoveredThisFrame.count == 0);

        anyElementHovered = false;
        anyWindowHovered = false;

        if (activeUI == null_ui_id)
            anyElementActive = false;

        keyboardInputASCIIKeycodeThisFrame.reset_count();
        keyboardInputASCIIKeycodeThisFrame.memset_zero();
        freshIdCounter = 0;
        __reservedTextMemoryIndexer = 0;

        drawRequestsFrameStorageBuffer.ArenaOffset = 0;
        GUIDraw_NewFrame();
    }

    void Draw()
    {
        GUIDraw_DrawEverything();
    }

    void ProcessSDLEvent(const SDL_Event event)
    {
        switch (event.type)
        {
            case SDL_EVENT_MOUSE_MOTION:
            {
                // NOTE(Kevin): For game GUI where window aspect ratio does not match game aspect ratio, must
                //              map from window mouse pos to gui canvas mouse pos
                MouseXInGUI = int(float(MousePos.x) * (float(RenderTargetGUI.width) / float(BackbufferWidth)));
                MouseYInGUI = int(float(MousePos.y) * (float(RenderTargetGUI.height) / float(BackbufferHeight)));
            }break;
            case SDL_EVENT_KEY_DOWN:
            {
                SDL_KeyboardEvent keyevent = event.key;
                SDL_Keycode keycodeASCII = keyevent.key;
                keycodeASCII = ShiftASCII(keycodeASCII, keyevent.mod & (SDL_KMOD_LSHIFT | SDL_KMOD_RSHIFT));
                keyboardInputASCIIKeycodeThisFrame.put(keycodeASCII);
            }break;
        }
    }

}


static GPUShader main_ui_shader;
static const char* main_ui_shader_vs =
        "#version 330 core\n"
        "uniform mat4 matrixOrtho;\n"
        "layout (location = 0) in vec2 pos;\n"
        "layout (location = 1) in vec2 uv;\n"
        "out vec2 texUV;\n"
        "out vec2 fragPos;\n"
        "void main() {\n"
        "    gl_Position = matrixOrtho * vec4(pos, 0.0, 1.0);\n"
        "    texUV = uv;\n"
        "    fragPos = pos;\n"
        "}\n";
static const char* main_ui_shader_fs =
        "#version 330 core\n"
        "uniform sampler2D textureSampler0;\n"
        "uniform bool useColour = false;\n"
        "uniform vec4 uiColour;\n"
        "uniform ivec4 windowMask;\n"
        "in vec2 texUV;\n"
        "in vec2 fragPos;\n"
        "out vec4 colour;\n"
        "void main() {\n"
        "    vec4 fmask = vec4(windowMask);\n"
        "    bool maskxokay = fmask.x <= fragPos.x && fragPos.x < (fmask.x + fmask.z);\n"
        "    bool maskyokay = fmask.y <= fragPos.y && fragPos.y < (fmask.y + fmask.w);\n"
        "    if (!maskxokay || !maskyokay) {"
        "        colour = vec4(0.0, 0.0, 0.0, 0.0);"
        "    } else if (useColour) {\n"
        "        colour = uiColour;\n"
        "    } else {\n"
        "        colour = texture(textureSampler0, texUV);\n"
        "    }\n"
        "}\n";

static GPUShader rounded_corner_rect_shader;
static const char* rounded_corner_rect_shader_vs =
        "#version 330 core\n"
        "uniform mat4 matrixOrtho;\n"
        "layout (location = 0) in vec2 pos;\n"
        "layout (location = 1) in vec2 uv;\n"
        "out vec2 fragPos;\n"
        "void main() {\n"
        "    fragPos = pos;\n"
        "    gl_Position = matrixOrtho * vec4(pos, 0.0, 1.0);\n"
        "}\n";
static const char* rounded_corner_rect_shader_fs =
        "#version 330 core\n"
        "uniform ivec4 rect;\n"
        "uniform int cornerRadius;\n"
        "uniform vec4 uiColour;\n"
        "uniform ivec4 windowMask;\n"
        "in vec2 fragPos;\n"
        "out vec4 colour;\n"
        "void main() {\n"
        "    vec4 frect = vec4(rect);\n"
        "    float fradius = float(cornerRadius);\n"
        "    bool xokay = (frect.x + fradius) < fragPos.x && fragPos.x < (frect.x + frect.z - fradius);\n"
        "    bool yokay = (frect.y + fradius) < fragPos.y && fragPos.y < (frect.y + frect.w - fradius);\n"
        "\n"
        "    vec4 fmask = vec4(windowMask);\n"
        "    bool maskxokay = fmask.x <= fragPos.x && fragPos.x < (fmask.x + fmask.z);\n"
        "    bool maskyokay = fmask.y <= fragPos.y && fragPos.y < (fmask.y + fmask.w);\n"
        "    if (!maskxokay || !maskyokay) {"
        "        colour = vec4(0.0, 0.0, 0.0, 0.0);"
        "    } else if (xokay || yokay) { \n"
        "        colour = uiColour;\n"
        "    } else {\n"
        "        vec2 cornerPoint;\n"
        "        if (fragPos.x < frect.x + fradius && fragPos.y < frect.y + fradius) { // top left\n"
        "            cornerPoint = vec2(frect.x + fradius, frect.y + fradius);\n"
        "        } else if (fragPos.x < frect.x + fradius && fragPos.y > frect.y + frect.w - fradius) { // bottom left\n"
        "            cornerPoint = vec2(frect.x + fradius, frect.y + frect.w - fradius);\n"
        "        } else if (fragPos.x > frect.x + frect.z - fradius && fragPos.y < frect.y + fradius) { // top right\n"
        "            cornerPoint = vec2(frect.x + frect.z - fradius, frect.y + fradius);\n"
        "        } else if (fragPos.x > frect.x + frect.z - fradius && fragPos.y > frect.y + frect.w - fradius) { // bottom right\n"
        "            cornerPoint = vec2(frect.x + frect.z - fradius, frect.y + frect.w - fradius);\n"
        "        }\n"
        "        if (distance(cornerPoint, fragPos) < fradius) {\n"
        "            colour = uiColour;\n"
        "        } else {\n"
        "            colour = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "        }\n"
        "    }\n"
        "}\n";

static GPUShader text_shader;
static const char* text_shader_vs =
        "#version 330 core\n"
        "uniform mat4 matrixModel;\n"
        "uniform mat4 matrixOrtho;\n"
        "layout (location = 0) in vec2 pos;\n"
        "layout (location = 1) in vec2 uv;\n"
        "out vec2 fragPos;\n"
        "out vec2 texUV;\n"
        "void main() {\n"
        "    fragPos = pos;\n"
        "    gl_Position = matrixOrtho * matrixModel * vec4(pos, 0.0, 1.0);\n"
        "    texUV = uv;\n"
        "}\n";
static const char* text_shader_fs =
        "#version 330 core\n"
        "uniform sampler2D textureSampler0;\n"
        "uniform vec4 uiColour;\n"
        "uniform ivec4 rectMask;\n"
        "uniform ivec4 windowMask;\n"
        "uniform int rectMaskCornerRadius;\n"
        "in vec2 fragPos;\n"
        "in vec2 texUV;\n"
        "out vec4 colour;\n"
        "void main() {\n"
        "    float textAlpha = texture(textureSampler0, texUV).x;\n"
        "\n"
        "    vec4 fmask = vec4(windowMask);\n"
        "    bool maskxokay = fmask.x <= fragPos.x && fragPos.x < (fmask.x + fmask.z);\n"
        "    bool maskyokay = fmask.y <= fragPos.y && fragPos.y < (fmask.y + fmask.w);\n"
        "    if (!maskxokay || !maskyokay) {"
        "        colour = vec4(0.0, 0.0, 0.0, 0.0);"
        "    } else if (rectMaskCornerRadius < 0)\n"
        "    {\n"
        "        colour = vec4(uiColour.xyz, uiColour.w * textAlpha);\n"
        "    }\n"
        "    else\n"
        "    {\n"
        "        vec4 frect = vec4(rectMask);\n"
        "        float fradius = float(rectMaskCornerRadius);\n"
        "\n"
        "        bool xbad = fragPos.x < frect.x || (frect.x + frect.z) < fragPos.x;\n"
        "        bool ybad = fragPos.y < frect.y || (frect.y + frect.w) < fragPos.y;\n"
        "\n"
        "        if (xbad || ybad) {\n"
        "            colour = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "        } else {\n"
        "            bool xokay = (frect.x + fradius) < fragPos.x && fragPos.x < (frect.x + frect.z - fradius);\n"
        "            bool yokay = (frect.y + fradius) < fragPos.y && fragPos.y < (frect.y + frect.w - fradius);\n"
        "\n"
        "            if (xokay || yokay) { \n"
        "                colour = vec4(uiColour.xyz, uiColour.w * textAlpha);\n"
        "            } else {\n"
        "                vec2 cornerPoint;\n"
        "                if (fragPos.x < frect.x + fradius && fragPos.y < frect.y + fradius) { // top left\n"
        "                    cornerPoint = vec2(frect.x + fradius, frect.y + fradius);\n"
        "                } else if (fragPos.x < frect.x + fradius && fragPos.y > frect.y + frect.w - fradius) { // bottom left\n"
        "                    cornerPoint = vec2(frect.x + fradius, frect.y + frect.w - fradius);\n"
        "                } else if (fragPos.x > frect.x + frect.z - fradius && fragPos.y < frect.y + fradius) { // top right\n"
        "                    cornerPoint = vec2(frect.x + frect.z - fradius, frect.y + fradius);\n"
        "                } else if (fragPos.x > frect.x + frect.z - fradius && fragPos.y > frect.y + frect.w - fradius) { // bottom right\n"
        "                    cornerPoint = vec2(frect.x + frect.z - fradius, frect.y + frect.w - fradius);\n"
        "                }\n"
        "                if (distance(cornerPoint, fragPos) < fradius) {\n"
        "                    colour = vec4(uiColour.xyz, uiColour.w * textAlpha);\n"
        "                } else {\n"
        "                    colour = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
        "";

static GPUShader colored_text_shader;
static const char* colored_text_shader_vs =
        "#version 330 core\n"
        "uniform mat4 matrixModel;\n"
        "uniform mat4 matrixOrtho;\n"
        "layout (location = 0) in vec2 pos;\n"
        "layout (location = 1) in vec2 uv;\n"
        "layout (location = 2) in vec3 vcol;\n"
        "out vec2 fragPos;\n"
        "out vec2 texUV;\n"
        "out vec3 vertColor;\n"
        "void main() {\n"
        "    fragPos = pos;\n"
        "    gl_Position = matrixOrtho * matrixModel * vec4(pos, 0.0, 1.0);\n"
        "    texUV = uv;\n"
        "    vertColor = vcol;\n"
        "}\n";
static const char* colored_text_shader_fs =
        "#version 330 core\n"
        "uniform sampler2D textureSampler0;\n"
        "uniform ivec4 rectMask;\n"
        "uniform int rectMaskCornerRadius;\n"
        "uniform ivec4 windowMask;\n"
        "in vec2 fragPos;\n"
        "in vec2 texUV;\n"
        "in vec3 vertColor;\n"
        "out vec4 colour;\n"
        "void main() {\n"
        "    float textAlpha = texture(textureSampler0, texUV).x;\n"
        "    \n"
        "    vec4 fmask = vec4(windowMask);\n"
        "    bool maskxokay = fmask.x <= fragPos.x && fragPos.x < (fmask.x + fmask.z);\n"
        "    bool maskyokay = fmask.y <= fragPos.y && fragPos.y < (fmask.y + fmask.w);\n"
        "    if (!maskxokay || !maskyokay) {"
        "        colour = vec4(0.0, 0.0, 0.0, 0.0);"
        "    } else if (rectMaskCornerRadius < 0)\n"
        "    {\n"
        "        colour = vec4(vertColor, textAlpha);\n"
        "    }\n"
        "    else\n"
        "    {\n"
        "        vec4 frect = vec4(rectMask);\n"
        "        float fradius = float(rectMaskCornerRadius);\n"
        "\n"
        "        bool xbad = fragPos.x < frect.x || (frect.x + frect.z) < fragPos.x;\n"
        "        bool ybad = fragPos.y < frect.y || (frect.y + frect.w) < fragPos.y;\n"
        "\n"
        "        if (xbad || ybad) {\n"
        "            colour = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "        } else {\n"
        "            bool xokay = (frect.x + fradius) < fragPos.x && fragPos.x < (frect.x + frect.z - fradius);\n"
        "            bool yokay = (frect.y + fradius) < fragPos.y && fragPos.y < (frect.y + frect.w - fradius);\n"
        "\n"
        "            if (xokay || yokay) { \n"
        "                colour = vec4(vertColor, textAlpha);\n"
        "            } else {\n"
        "                vec2 cornerPoint;\n"
        "                if (fragPos.x < frect.x + fradius && fragPos.y < frect.y + fradius) { // top left\n"
        "                    cornerPoint = vec2(frect.x + fradius, frect.y + fradius);\n"
        "                } else if (fragPos.x < frect.x + fradius && fragPos.y > frect.y + frect.w - fradius) { // bottom left\n"
        "                    cornerPoint = vec2(frect.x + fradius, frect.y + frect.w - fradius);\n"
        "                } else if (fragPos.x > frect.x + frect.z - fradius && fragPos.y < frect.y + fradius) { // top right\n"
        "                    cornerPoint = vec2(frect.x + frect.z - fradius, frect.y + fradius);\n"
        "                } else if (fragPos.x > frect.x + frect.z - fradius && fragPos.y > frect.y + frect.w - fradius) { // bottom right\n"
        "                    cornerPoint = vec2(frect.x + frect.z - fradius, frect.y + frect.w - fradius);\n"
        "                }\n"
        "                if (distance(cornerPoint, fragPos) < fradius) {\n"
        "                    colour = vec4(vertColor, textAlpha);\n"
        "                } else {\n"
        "                    colour = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
        "";

static GPUMeshIndexed s_ui_mesh;
static GPUMeshIndexed s_text_mesh;
static GPUMeshIndexed s_colored_text_mesh;

#define MAX_DRAWCOLLECTIONS_ALLOWED 16

namespace GUI
{
    struct DrawCollectionMetaData
    {
        UIRect windowMask;
        int depth = 0;
    };

    static UIRect activeWindowMask;
    static c_array<std::vector<UIDrawRequest*>, MAX_DRAWCOLLECTIONS_ALLOWED + 1> DRAWQSTORAGE;
    static c_array<DrawCollectionMetaData, MAX_DRAWCOLLECTIONS_ALLOWED + 1> DRAWQUEUE_METADATA;
    static std::stack<std::vector<UIDrawRequest*>*> DRAWREQCOLLECTIONSTACK;

    void GUIDraw_InitResources()
    {
        GLCreateShaderProgram(main_ui_shader, main_ui_shader_vs, main_ui_shader_fs);
        GLCreateShaderProgram(rounded_corner_rect_shader, rounded_corner_rect_shader_vs, rounded_corner_rect_shader_fs);
        GLCreateShaderProgram(text_shader, text_shader_vs, text_shader_fs);
        GLCreateShaderProgram(colored_text_shader, colored_text_shader_vs, colored_text_shader_fs);
        CreateGPUMeshIndexed(&s_ui_mesh, nullptr, nullptr, 0, 0, 2, 2, 0, GL_DYNAMIC_DRAW);
        CreateGPUMeshIndexed(&s_text_mesh, nullptr, nullptr, 0, 0, 2, 2, 0, GL_DYNAMIC_DRAW);
        CreateGPUMeshIndexed(&s_colored_text_mesh, nullptr, nullptr, 0, 0, 2, 2, 3, GL_DYNAMIC_DRAW);

        ASSERT(DRAWQSTORAGE.count == 0);
        ASSERT(DRAWQUEUE_METADATA.count == 0);
        ASSERT(DRAWREQCOLLECTIONSTACK.empty());

        DRAWQSTORAGE.count = 1; // base collection
        DRAWQUEUE_METADATA.put({UIRect(0, 0, 9999, 9999), 0});
        DRAWREQCOLLECTIONSTACK.push(&DRAWQSTORAGE.back());
    }

    void GUIDraw_NewFrame()
    {
        // Clear all collections
        for (int i = 0; i < DRAWQSTORAGE.count; ++i)
            DRAWQSTORAGE[i].clear();
        // clear draw queue
        DRAWQSTORAGE.count = 1;
        DRAWQUEUE_METADATA.count = 1;
        if (DRAWREQCOLLECTIONSTACK.size() > 1)
        {
            LogError("GUI BeginWindow and EndWindow don't match.");
            ASSERT(0);
        }
    }

    void GUIDraw_DrawEverything()
    {
        i32 kevGuiScreenWidth = RenderTargetGUI.width;
        i32 kevGuiScreenHeight = RenderTargetGUI.height;
        mat4 projectionMatrix = ProjectionMatrixOrthographicNoZ(0.f, (float)kevGuiScreenWidth, (float)kevGuiScreenHeight, 0.f);

        UseShader(main_ui_shader);
        GLBindMatrix4fv(main_ui_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        UseShader(rounded_corner_rect_shader);
        GLBindMatrix4fv(rounded_corner_rect_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        UseShader(text_shader);
        GLBindMatrix4fv(text_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        UseShader(colored_text_shader);
        GLBindMatrix4fv(colored_text_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        // Draw base collection
        activeWindowMask = DRAWQUEUE_METADATA[0].windowMask;
        std::vector<UIDrawRequest *> &baseDrawQueue = DRAWQSTORAGE[0];
        for (auto drawCall : baseDrawQueue)
            drawCall->Draw();

        // could sort so its O(n) but realistically how many windows am I going to have...
        int highestDepth = 0;
        for (int i = 0; i < DRAWQUEUE_METADATA.count; ++i)
            highestDepth = GM_max(highestDepth, DRAWQUEUE_METADATA[i].depth);

        // Draw collections of depth 0 to highestDepth except for base collection
        for (int depth = 0; depth <= highestDepth; ++depth)
        {
            for (int i = 1; i < DRAWQSTORAGE.count; ++i)
            {
                if (DRAWQUEUE_METADATA[i].depth == depth)
                {
                    activeWindowMask = DRAWQUEUE_METADATA[i].windowMask;
                    std::vector<UIDrawRequest*>& drawQueue = DRAWQSTORAGE[i];
                    for (auto drawCall : drawQueue)
                        drawCall->Draw();
                }
            }
        }
    }

    void GUIDraw_PushDrawCollection(UIRect windowMask, int depth)
    {
        ASSERT(DRAWQSTORAGE.not_at_cap());
        DRAWQSTORAGE.count++;
        depth = GM_min(depth, MAX_DRAWCOLLECTIONS_ALLOWED);
        DRAWQUEUE_METADATA.put({windowMask, depth});
        DRAWREQCOLLECTIONSTACK.push(&DRAWQSTORAGE.back());
    }

    void GUIDraw_PopDrawCollection()
    {
        ASSERT(DRAWREQCOLLECTIONSTACK.size() > 1);
        DRAWREQCOLLECTIONSTACK.pop();
    }

    u8 GetCurrentDrawingDepth()
    {
        return DRAWQUEUE_METADATA.back().depth;
    }

    void AppendToCurrentDrawRequestsCollection(UIDrawRequest *drawRequest)
    {
        DRAWREQCOLLECTIONSTACK.top()->push_back(drawRequest);
    }

    void RectDrawRequest::Draw()
    {
        float left = (float)rect.x;
        float top = (float)rect.y;
        float bottom = (float)rect.y + rect.h;
        float right = (float)rect.x + rect.w;
        float vb[] = { left, top, 0.f, 1.f,
                       left, bottom, 0.f, 0.f,
                       right, bottom, 1.f, 0.f,
                       right, top, 1.f, 1.f, };
        u32 ib[] = { 0, 1, 3, 1, 2, 3 };

        UseShader(main_ui_shader);
        GLBind4i(main_ui_shader, "windowMask", activeWindowMask.x, activeWindowMask.y, activeWindowMask.w, activeWindowMask.h);

        if (textureId != 0)
        {
            GLBind1i(main_ui_shader, "useColour", false);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);
            GLBind1i(main_ui_shader, "textureSampler0", 0);
        }
        else
        {
            GLBind1i(main_ui_shader, "useColour", true);
            GLBind4f(main_ui_shader, "uiColour", color.x, color.y, color.z, color.w);
        }

        RebindGPUMeshIndexedData(&s_ui_mesh, vb, ib, ARRAY_COUNT(vb), ARRAY_COUNT(ib), GL_DYNAMIC_DRAW);
        RenderGPUMeshIndexed(s_ui_mesh);
    }

    void RoundedCornerRectDrawRequest::Draw()
    {
        float left = (float)rect.x;
        float top = (float)rect.y;
        float bottom = (float)rect.y + rect.h;
        float right = (float)rect.x + rect.w;
        float vb[] = { left, top, 0.f, 1.f,
                       left, bottom, 0.f, 0.f,
                       right, bottom, 1.f, 0.f,
                       right, top, 1.f, 1.f, };
        u32 ib[] = { 0, 1, 3, 1, 2, 3 };

        UseShader(rounded_corner_rect_shader);
        GLBind4i(main_ui_shader, "windowMask", activeWindowMask.x, activeWindowMask.y, activeWindowMask.w, activeWindowMask.h);
        GLBind4i(rounded_corner_rect_shader, "rect", rect.x, rect.y, rect.w, rect.h);
        GLBind1i(rounded_corner_rect_shader, "cornerRadius", radius);
        GLBind4f(rounded_corner_rect_shader, "uiColour", color.x, color.y, color.z, color.w);

        RebindGPUMeshIndexedData(&s_ui_mesh, vb, ib, ARRAY_COUNT(vb), ARRAY_COUNT(ib), GL_DYNAMIC_DRAW);
        RenderGPUMeshIndexed(s_ui_mesh);
    }

    void CorneredRectDrawRequest::Draw()
    {
        float left = (float)rect.x;
        float top = (float)rect.y;
        float bottom = (float)rect.y + rect.h;
        float right = (float)rect.x + rect.w;
        float corner = (float)radius;
        float uv0 = normalizedCornerSizeInUV;
        float uv1 = 1.f - uv0;

        float vb[] = { left, top,              0.f, 1.f,
                       left, top + corner,     0.f, uv1,
                       left + corner, top,     uv0, 1.f,
                       left + corner, top + corner, uv0, uv1,

                       right - corner, top,    uv1, 1.f,
                       right - corner, top + corner, uv1, uv1,
                       right, top,             1.f, 1.f,
                       right, top + corner,    1.f, uv1,

                       left, bottom - corner,  0.f, uv0,
                       left, bottom,           0.f, 0.f,
                       left + corner, bottom - corner, uv0, uv0,
                       left + corner, bottom,  uv0, 0.f,

                       right - corner, bottom - corner, uv1, uv0,
                       right - corner, bottom, uv1, 0.f,
                       right, bottom - corner, 1.f, uv0,
                       right, bottom,          1.f, 0.f,
        };
        u32 ib[] = { 0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5, 4, 5, 6, 6, 5, 7, 1, 8, 3, 3, 8, 10, 3, 10, 5,
                     5, 10, 12, 5, 12, 7, 7, 12, 14, 8, 9, 10, 10, 9, 11, 10, 11, 12, 12, 11, 13, 12, 13, 14, 14, 13, 15 };

        UseShader(main_ui_shader);
        GLBind4i(main_ui_shader, "windowMask", activeWindowMask.x, activeWindowMask.y, activeWindowMask.w, activeWindowMask.h);

        if (textureId != 0)
        {
            GLBind1i(main_ui_shader, "useColour", false);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);
            GLBind1i(main_ui_shader, "textureSampler0", 0);
        }
        else
        {
            GLBind1i(main_ui_shader, "useColour", true);
            GLBind4f(main_ui_shader, "uiColour", color.x, color.y, color.z, color.w);
        }

        RebindGPUMeshIndexedData(&s_ui_mesh, vb, ib, ARRAY_COUNT(vb), ARRAY_COUNT(ib), GL_DYNAMIC_DRAW);
        RenderGPUMeshIndexed(s_ui_mesh);
    }

    void TextDrawRequest::Draw()
    {
        vtxt_setflags(VTXT_CREATE_INDEX_BUFFER);
        vtxt_clear_buffer();
        vtxt_move_cursor(x, y);
        switch (alignment)
        {
            case Align::LEFT:{
                vtxt_append_line(text, font.ptr, size);
            }break;
            case Align::CENTER:{
                vtxt_append_line_centered(text, font.ptr, size);
            }break;
            case Align::RIGHT:{
                vtxt_append_line_align_right(text, font.ptr, size);
            }break;
        }
        vtxt_vertex_buffer _txt = vtxt_grab_buffer();
        if (_txt.vertices_array_count <= 0)
        {
            vtxt_clear_buffer();
            return;
        }
        RebindGPUMeshIndexedData(&s_text_mesh, _txt.vertex_buffer, _txt.index_buffer, _txt.vertices_array_count, _txt.indices_array_count, GL_DYNAMIC_DRAW);
        vtxt_clear_buffer();

        mat4 matrixModel = mat4();
        UseShader(text_shader);
        GLBindMatrix4fv(text_shader, "matrixModel", 1, matrixModel.ptr());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, font.textureId);
        GLBind1i(text_shader, "textureSampler0", 0);
        GLBind4f(text_shader, "uiColour", color.x, color.y, color.z, color.w);

        GLBind4i(text_shader, "rectMask", rectMask.x, rectMask.y, rectMask.w, rectMask.h);
        GLBind1i(text_shader, "rectMaskCornerRadius", rectMaskCornerRadius);

        GLBind4i(text_shader, "windowMask", activeWindowMask.x, activeWindowMask.y, activeWindowMask.w, activeWindowMask.h);

        RenderGPUMeshIndexed(s_text_mesh);
    }

    void PipCodeDrawRequest::Draw()
    {
        vtxt_setflags(VTXT_CREATE_INDEX_BUFFER);
        vtxt_clear_buffer();
        vtxt_move_cursor(x, y);
        vtxt_append_line_vertex_color_hack(text, font.ptr, size, (float*)CodeCharIndexToColor);
        vtxt_vertex_buffer _txt = vtxt_grab_buffer();
        _txt.vertices_array_count = _txt.vertex_count * 7;

        if (_txt.vertices_array_count > 0)
        {
            RebindGPUMeshIndexedData(&s_colored_text_mesh, _txt.vertex_buffer, _txt.index_buffer, _txt.vertices_array_count, _txt.indices_array_count, GL_DYNAMIC_DRAW);

            mat4 matrixModel = mat4();
            UseShader(colored_text_shader);
            GLBindMatrix4fv(colored_text_shader, "matrixModel", 1, matrixModel.ptr());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, font.textureId);
            GLBind1i(colored_text_shader, "textureSampler0", 0);

            GLBind4i(colored_text_shader, "rectMask", rectMask.x, rectMask.y, rectMask.w, rectMask.h);
            GLBind1i(colored_text_shader, "rectMaskCornerRadius", rectMaskCornerRadius);

            GLBind4i(colored_text_shader, "windowMask", activeWindowMask.x, activeWindowMask.y, activeWindowMask.w, activeWindowMask.h);

            RenderGPUMeshIndexed(s_colored_text_mesh);
        }

        vtxt_clear_buffer();
    }
}

#undef ISANYOF1
#undef ISANYOF2
#undef ISANYOF3
#undef ISANYOF4

