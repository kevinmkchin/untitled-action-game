// NOTE(Kevin): This file is amalgamation of GUI.H, GUI.CPP, GUI_DRAWING.H, GUI_DRAWING.CPP from my pip/mesa
//              fantasy console project. Some things like Auto Layout Handle and PipCode/colored text drawing
//              may be useless for this game but who knows.


#define null_ui_id (-1)
typedef i64 ui_id;

#define ISANYOF1(a, x) ((a) == (x))
#define ISANYOF2(a, x, y) ((a) == (x) || (a) == (y))
#define ISANYOF3(a, x, y, z) ((a) == (x) || (a) == (y) || (a) == (z))
#define ISANYOF4(a, x, y, z, w) ((a) == (x) || (a) == (y) || (a) == (z) || (a) == (w))

namespace Gui
{
    extern NiceArray<SDL_Keycode, 32> keyboardInputASCIIKeycodeThisFrame;
    extern vec4 style_textColor;

    enum class Align
    {
        Left,
        Center,
        Right
    };

    struct Font
    {
        vtxt_font* ptr = nullptr;
        u32 textureId = 0;
    };

    struct UIRect
    {
        UIRect() : x(0), y(0), w(0), h(0) {};
        UIRect(int _x, int _y, int _w, int _h) : x(_x), y(_y), w(_w), h(_h) {};
        UIRect(struct ALH *layout);
        int x;
        int y;
        int w;
        int h;
    };

    /* Application API */
    void Init(); // on program start
    // void NewFrame(); // before each frame NOTE(Kevin): just call this as part of ProcessSDLEvent automatically
    void ProcessSDLEvent(const SDL_Event evt); // update input state
    void Draw(); // flush draw requests to GPU

    /* Immediate-mode state helpers */
    bool IsActive(ui_id id);
    bool IsHovered(ui_id id);
    void SetActive(ui_id id);
    void SetHovered(ui_id id);
    bool MouseWentUp();
    bool MouseWentDown();
    bool MouseInside(const UIRect& rect);
    // IsMouseInsideWindow?

    /* Behaviours
    * "building block" behaviours for making interactible GUI elements */
    bool Behaviour_Button(ui_id id, UIRect rect);

    bool ImageButton(UIRect rect, u32 normalTexId, u32 hoveredTexId, u32 activeTexId);
    extern vec3 CodeCharIndexToColor[];
    void PipCode(int x, int y, int size, const char* text);


    /* Primitive "building block" GUI elements with the most parameters
    * */
    void PrimitivePanel(UIRect rect, vec4 colorRGBA);
    void PrimitivePanel(UIRect rect, int cornerRadius, vec4 colorRGBA);
    void PrimitivePanel(UIRect rect, u32 glTextureId);
    void PrimitivePanel(UIRect rect, int cornerRadius, u32 glTextureId = 0, float normalizedCornerSizeInUV = 0.3f);
    bool PrimitiveButton(ui_id id, UIRect rect, vec4 normalColor, vec4 hoveredColor, vec4 activeColor, bool activeColorOnClickReleaseFrame = false);
    void PrimitiveText(int x, int y, int size, Align alignment, const char* text);
    void PrimitiveTextFmt(int x, int y, int size, Align alignment, const char* textFmt, ...);
    void PrimitiveTextMasked(int x, int y, int size, Align alignment, const char* text, UIRect mask, int maskCornerRadius);
    void PrimitiveIntegerInputField(ui_id id, UIRect rect, int* v);
    void PrimitiveFloatInputField(ui_id id, UIRect rect, float* v);
    //void PrimitiveCheckbox(const char* label, );
    bool PrimitiveLabelledButton(UIRect rect, const char* label, Align textAlignment);


    /* Windows
    * For aligning GUI elements like a DearImGui window
    * TODO Can be set to stop clicks from going through (if i click inside zone, then click doesn't go through to zones or program behind?)
    * TODO think about:
    * Can be set to capture focus?
    * Can be set to collapse?
    * */
    void BeginWindow(UIRect windowRect, vec4 bgcolor = vec4(0.05f, 0.05f, 0.05f, 0.5f), int depth = -1);
    void EndWindow();
    void Window_GetWidthHeight(int *w, int *h);
    void Window_GetCurrentOffsets(int *x, int *y);
    void Window_StageLastElementDimension(int x, int y);
    void Window_CommitLastElementDimension();
    // Essentially workflow is like this:
    // When Editor GUI element function called:
    // - Window_CommitLastElementDimension to shift offsets by the last added element
    // - GetXY and draw stuff
    // - call Window_StageLastElementDimension to cache the dimensions of this added element

    /* GUI elements that go inside windows
    * */
    void EditorText(const char *text);
    // TODO void EditorImage(u32 glTextureId, ivec2 size, vec2 uv0 = vec2(0.f,0.f), vec2 uv1 = vec2(1.f,1.f));
    // TODO void EditorImageButton(u32 glTextureId, ivec2 size, vec2 uv0 = vec2(0.f,0.f), vec2 uv1 = vec2(1.f,1.f));
    bool EditorLabelledButton(const char *label);
    void EditorIncrementableIntegerField(const char *label, int *v, int increment = 1);
    void EditorIncrementableFloatField(const char *label, float *v, float increment = 0.1f);

    // EditorSelectable returns true once when it gets selected
    bool EditorSelectable(const char *label, bool *selected);
    bool EditorSelectableRect(vec4 colorRGBA, bool *selected, int id);

    void EditorBeginListBox();
    void EditorEndListBox();

    void EditorBeginHorizontal();
    void EditorEndHorizontal();

    void EditorColorPicker(ui_id id, float *hue, float *saturation, float *value, float *opacity);

    void EditorSpacer(int x, int y);


    struct ALH
    {
        // if layout has absolute x y then it is not auto layouted
        int x;
        int y;
        bool xauto = true;
        bool yauto = true;

        int w;
        int h;
        bool wauto = true;
        bool hauto = true;

        bool vertical = true;

        void Insert(ALH *layout, int index)
        {
            container.insert(container.begin() + index, layout);
        }

        void Insert(ALH *layout)
        {
            container.push_back(layout);
        }

        void Replace(int index, ALH *layout)
        {
            ASSERT(index < (int)container.size());
            container.at(index) = layout;
        }

        int Count()
        {
            return int(container.size());
        }

        std::vector<ALH*> container;
    };

    void UpdateMainCanvasALH(ALH *layout);
    ALH *NewALH(bool vertical);
    ALH *NewALH(int absX, int absY, int absW, int absH, bool vertical);
    void DeleteALH(ALH *layout);
}

namespace Gui {

    struct UIDrawRequest;

    void GUIDraw_InitResources();
    void GUIDraw_DrawEverything();
    void GUIDraw_NewFrame();

    void GUIDraw_PushDrawCollection(UIRect windowMask, int depth = -1);
    void GUIDraw_PopDrawCollection();
    void AppendToCurrentDrawRequestsCollection(UIDrawRequest *drawRequest);

    struct UIDrawRequest
    {
        vec4 color;

        virtual void Draw() = 0;
    };

    struct RectDrawRequest : UIDrawRequest
    {
        UIRect rect;
        GLuint textureId = 0;

        void Draw() final;
    };

    struct RoundedCornerRectDrawRequest : UIDrawRequest
    {
        UIRect rect;
        int radius = 10;

        void Draw() final;
    };

    struct CorneredRectDrawRequest : UIDrawRequest
    {
        UIRect rect;
        int radius = 10;

        GLuint textureId = 0;
        float normalizedCornerSizeInUV = 0.3f; // [0,1] with 0.5 being half way across texture

        void Draw() final;
    };

    struct TextDrawRequest : UIDrawRequest
    {
        const char* text = "";
        int size = 8;
        int x = 0;
        int y = 0;
        Align alignment = Align::Left;
        Font font;

        UIRect rectMask = UIRect(0, 0, 9999, 9999);
        int rectMaskCornerRadius = -1;

        void Draw() final;
    };

    struct PipCodeDrawRequest : UIDrawRequest
    {
        const char* text = "";
        int size = 8;
        int x = 0;
        int y = 0;
        Font font;

        UIRect rectMask = UIRect(0, 0, 9999, 9999);
        int rectMaskCornerRadius = -1;

        void Draw() final;
    };

}


static ivec2 lastAddedElementDimension;
static bool flag_HorizontalMode = false;

static ui_id freshIdCounter = 0;
static ui_id FreshID()
{
    return freshIdCounter++;
}

namespace Gui
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


    static NiceArray<vtxt_font, 10> s_vtxtLoadedFonts;
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

        s_vtxtLoadedFonts.PushBack(fontHandle);
        fontToReturn.ptr = &s_vtxtLoadedFonts.Back();
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

        s_vtxtLoadedFonts.PushBack(fontHandle);
        bitmapFont.ptr = &s_vtxtLoadedFonts.Back();
        return bitmapFont;
    }

    UIRect::UIRect(struct ALH *layout) : x(layout->x), y(layout->y), w(layout->w), h(layout->h) {};

    static char __reservedTextMemory[16000000];
    static u32 __reservedTextMemoryIndexer = 0;
    NiceArray<SDL_Keycode, 32> keyboardInputASCIIKeycodeThisFrame;
    static NiceArray<char, 128> activeTextInputBuffer;

    static ui_id hoveredUI = null_ui_id;
    static ui_id activeUI = null_ui_id;
    static bool anyElementHoveredThisFrame = false;


    static MemoryLinearBuffer drawRequestsFrameStorageBuffer;
#define MESAIMGUI_NEW_DRAW_REQUEST(type) new (MEMORY_LINEAR_ALLOCATE(&drawRequestsFrameStorageBuffer, type)) type()

    struct WindowData
    {
        ui_id zoneId = null_ui_id;
        UIRect zoneRect;
        int paddingFromLeft = 0;
        int topLeftXOffset = 0;
        int topLeftYOffset = 0;
    };
    static std::stack<WindowData> WINDOWSTACK;
    WindowData *CurrentWindow()
    {
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
    }

    void SetHovered(ui_id id)
    {
        anyElementHoveredThisFrame = true;
        hoveredUI = id;
    }


    static int mouseX_InGuiTarget = 0;
    static int mouseY_InGuiTarget = 0;

    bool MouseWentUp()
    {
        return mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT);
    }

    bool MouseWentDown()
    {
        return mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT);
    }

    bool MouseInside(const UIRect& rect)
    {
        int left = rect.x;
        int top = rect.y;
        int right = left + rect.w;
        int bottom = top + rect.h;
        if (left <= mouseX_InGuiTarget && mouseX_InGuiTarget < right
            && top <= mouseY_InGuiTarget && mouseY_InGuiTarget < bottom)
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
            SetHovered(id);
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
                i32 keycodeASCII = keyboardInputASCIIKeycodeThisFrame.At(i);
                if (48 <= keycodeASCII && keycodeASCII <= 57)
                {
                    if (activeTextInputBuffer.count == 1 && activeTextInputBuffer.At(0) == '0')
                    {
                        activeTextInputBuffer.At(0) = char(keycodeASCII);
                    }
                    else if (activeTextInputBuffer.count < 9) // just to prevent integers that are too big
                    {
                        activeTextInputBuffer.PushBack(char(keycodeASCII));
                    }
                }
                else if (keycodeASCII == 45 /* minus sign */ && activeTextInputBuffer.count == 0)
                {
                    activeTextInputBuffer.PushBack(char(keycodeASCII));
                }
                else if (keycodeASCII == SDLK_RETURN)
                {
                    bSetInactiveAndReturnValue = true;
                }
                else if (keycodeASCII == SDLK_BACKSPACE)
                {
                    if (activeTextInputBuffer.count > 0)
                    {
                        activeTextInputBuffer.Back() = '\0';
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
                activeTextInputBuffer.ResetToZero();
                memcpy(activeTextInputBuffer.data, intValueAsString.c_str(), intValueAsString.size());
                activeTextInputBuffer.count = (int)intValueAsString.size();
                SetActive(id);
            }
        }

        if (bSetInactiveAndReturnValue)
        {
            int inputtedInteger = 0;
            bool inputIsNotEmpty = activeTextInputBuffer.count > 0;
            bool onlyInputIsMinusSign = activeTextInputBuffer.count == 1 && activeTextInputBuffer.At(0) == '-';
            if (inputIsNotEmpty && !onlyInputIsMinusSign)
            {
                inputtedInteger = std::stoi(activeTextInputBuffer.data);
            }
            *v = inputtedInteger;
            activeTextInputBuffer.ResetCount();
            activeTextInputBuffer.ResetToZero();
            SetActive(null_ui_id);
        }

        if (MouseInside(rect))
        {
            SetHovered(id);
        }

        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = IsActive(id) ? vec4(0.f, 0.f, 0.f, 1.f) : vec4(0.2f, 0.2f, 0.2f, 1.f);//vec4(1.f, 1.f, 1.f, 1.f) : vec4(0.8f, 0.8f, 0.8f, 1.f);

        AppendToCurrentDrawRequestsCollection(drawRequest);

        if (IsActive(id))
        {
            if (activeTextInputBuffer.count > 0)
            {
                PrimitiveText(rect.x + rect.w, rect.y + rect.h, 9, Align::Right, activeTextInputBuffer.data);
            }
        }
        else
        {
            PrimitiveText(rect.x + rect.w, rect.y + rect.h, 9, Align::Right, std::to_string(*v).c_str());
        }
    }

    void PrimitiveFloatInputField(ui_id id, UIRect rect, float* v)
    {
        bool bSetInactiveAndReturnValue = false;

        if (IsActive(id))
        {
            for (int i = 0; i < keyboardInputASCIIKeycodeThisFrame.count; ++i)
            {
                i32 keycodeASCII = keyboardInputASCIIKeycodeThisFrame.At(i);
                if (48 <= keycodeASCII && keycodeASCII <= 57)
                {
                    if (activeTextInputBuffer.count == 1 && activeTextInputBuffer.At(0) == '0')
                    {
                        activeTextInputBuffer.At(0) = char(keycodeASCII);
                    }
                    else if (activeTextInputBuffer.count < 9) // just to prevent floats that are too big
                    {
                        activeTextInputBuffer.PushBack(char(keycodeASCII));
                    }
                }
                else if (keycodeASCII == 45 /* minus sign */
                         && activeTextInputBuffer.count == 0)
                {
                    activeTextInputBuffer.PushBack(char(keycodeASCII));
                }
                else if (keycodeASCII == 46 /* decimal point */
                         && !IsOneOfArray('.', activeTextInputBuffer.data, activeTextInputBuffer.count))
                {
                    activeTextInputBuffer.PushBack(char(keycodeASCII));
                }
                else if (keycodeASCII == SDLK_RETURN)
                {
                    bSetInactiveAndReturnValue = true;
                }
                else if (keycodeASCII == SDLK_BACKSPACE)
                {
                    if (activeTextInputBuffer.count > 0)
                    {
                        activeTextInputBuffer.Back() = '\0';
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
                activeTextInputBuffer.ResetToZero();
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
                                             && ISANYOF2(activeTextInputBuffer.At(0), '-', '.');
            if (inputIsNotEmpty && !onlyInputIsMinusSignOrDot)
            {
                inputtedFloat = std::stof(activeTextInputBuffer.data);
            }
            *v = inputtedFloat;
            activeTextInputBuffer.ResetCount();
            activeTextInputBuffer.ResetToZero();
            SetActive(null_ui_id);
        }

        if (MouseInside(rect))
        {
            SetHovered(id);
        }

        RectDrawRequest *drawRequest = MESAIMGUI_NEW_DRAW_REQUEST(RectDrawRequest);
        drawRequest->rect = rect;
        drawRequest->color = IsActive(id) ? vec4(0.f, 0.f, 0.f, 1.f) : vec4(0.2f, 0.2f, 0.2f, 1.f);

        AppendToCurrentDrawRequestsCollection(drawRequest);

        if (IsActive(id))
        {
            if (activeTextInputBuffer.count > 0)
            {
                PrimitiveText(rect.x + rect.w, rect.y + rect.h, 9, Align::Right, activeTextInputBuffer.data);
            }
        }
        else
        {
            char cbuf[32];
            stbsp_sprintf(cbuf, "%.2f", *v);
            PrimitiveText(rect.x + rect.w, rect.y + rect.h, 9, Align::Right, cbuf);
        }
    }



    bool PrimitiveLabelledButton(UIRect rect, const char* label, Align textAlignment)
    {
        int ascenderTextSize = style_textFont.ptr->font_height_px;
        float yTextPaddingRatio = (1.f - (float(ascenderTextSize) / float(rect.h))) / 2.f;
        ivec2 textPadding = ivec2(10, (int) roundf(rect.h * yTextPaddingRatio));
        int textX = rect.x + textPadding.x;
        if (textAlignment == Align::Center)
        {
            textX = (int) ((rect.w / 2) + rect.x);
        }
        else if (textAlignment == Align::Right)
        {
            textX = (int) rect.x + rect.w - textPadding.x;
        }

        bool buttonValue = PrimitiveButton(FreshID(), rect, style_buttonNormalColor, style_buttonHoveredColor, style_buttonActiveColor);
        PrimitiveText(textX, rect.y + rect.h - textPadding.y, ascenderTextSize, textAlignment, label);
        return buttonValue;
    }

    void BeginWindow(UIRect windowRect, vec4 bgcolor, int depth)
    {
        GUIDraw_PushDrawCollection(windowRect, depth);

        WindowData windata;
        windata.zoneId = FreshID();
        windata.zoneRect = windowRect;
        windata.paddingFromLeft = 5;
        windata.topLeftXOffset = windata.paddingFromLeft;
        windata.topLeftYOffset = 5;
        WINDOWSTACK.push(windata);

        PrimitivePanel(windowRect, bgcolor);

        lastAddedElementDimension = ivec2();
    }

    void EndWindow()
    {
        if (!WINDOWSTACK.empty())
        {
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
        *w = CurrentWindow()->zoneRect.w;
        *h = CurrentWindow()->zoneRect.h;
    }

    void Window_GetCurrentOffsets(int *x, int *y)
    {
        *x = CurrentWindow()->zoneRect.x + CurrentWindow()->topLeftXOffset;
        *y = CurrentWindow()->zoneRect.y + CurrentWindow()->topLeftYOffset;
    }

    void Window_StageLastElementDimension(int x, int y)
    {
        lastAddedElementDimension = ivec2(x,y);
    }

    void Window_CommitLastElementDimension()
    {
        if (flag_HorizontalMode)
        {
            CurrentWindow()->topLeftXOffset += lastAddedElementDimension.x;
        }
        else
        {
            CurrentWindow()->topLeftXOffset = CurrentWindow()->paddingFromLeft;
            CurrentWindow()->topLeftYOffset += lastAddedElementDimension.y;
        }
    }

    void EditorText(const char* text)
    {
        Window_CommitLastElementDimension();

        int x;
        int y;
        Window_GetCurrentOffsets(&x, &y);

        int sz = style_textFont.ptr->font_height_px;

        PrimitiveText(x + style_paddingLeft, y + sz + style_paddingTop, sz, Align::Left, text);

        Window_StageLastElementDimension(0, sz + style_paddingTop + style_paddingBottom);
    }

    void EditorSpacer(int x, int y)
    {
        Window_CommitLastElementDimension();
        Window_StageLastElementDimension(x, y);
    }

    bool EditorLabelledButton(const char* label)
    {
        Window_CommitLastElementDimension();

        int labelTextSize = style_textFont.ptr->font_height_px;
        float textW;
        float textH;
        vtxt_get_text_bounding_box_info(&textW, &textH, label, style_textFont.ptr, labelTextSize);

        int buttonX;
        int buttonY;
        Window_GetCurrentOffsets(&buttonX, &buttonY);
        buttonX += style_paddingLeft;
        buttonY += style_paddingTop;
        int buttonW = GM_max((int) textW + 4, 50);
        int buttonH = labelTextSize + 4;

        UIRect buttonRect = UIRect(buttonX, buttonY, buttonW, buttonH);
        bool result = PrimitiveLabelledButton(buttonRect, label, Align::Center);

        Window_StageLastElementDimension(0, style_paddingTop + buttonH + style_paddingBottom);

        return result;
    }

    void EditorIncrementableIntegerField(const char* label, int* v, int increment)
    {
        Window_CommitLastElementDimension();

        int x;
        int y;
        Window_GetCurrentOffsets(&x, &y);

        int w = 50;
        int h = 9 + 4;
        int paddingAbove = style_paddingTop;
        int paddingBelow = style_paddingBottom;

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
        PrimitiveText(x + w + 12, y + h, 9, Align::Left, label);

        Window_StageLastElementDimension(0, paddingAbove + h + paddingBelow);
    }

    void EditorIncrementableFloatField(const char* label, float* v, float increment)
    {
        Window_CommitLastElementDimension();

        int x, y;
        Window_GetCurrentOffsets(&x, &y);

        int w = 50;
        int h = 9+4;
        int paddingAbove = style_paddingTop;
        int paddingBelow = style_paddingBottom;

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
        PrimitiveText(x + w + 12, y + h, 9, Align::Left, label);

        Window_StageLastElementDimension(0, paddingAbove + h + paddingBelow);
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

        UIRect selectableRegion = UIRect(x, y, 100, 11);
        if (*selected)
        {
            PrimitivePanel(selectableRegion, vec4(0,0,0,0.4f));
            PrimitiveText(x + 1, y + 10, 9, Align::Left, label);
            Window_StageLastElementDimension(0, 11);
        }
        else
        {
            *selected = PrimitiveButton(FreshID(), selectableRegion, vec4(), vec4(0,0,0,0.2f), vec4(0,0,0,0.4f), true);
            PrimitiveText(x + 1, y + 10, 9, Align::Left, label);
            Window_StageLastElementDimension(0, 11);
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
            SetHovered(id);
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
            SetHovered(id + 1);
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
            SetHovered(id + 2);
        }

        if (IsActive(id))
        {
            i32 chromaSMouseX = GM_clamp(mouseX_InGuiTarget - chromaselectorrect.x, 0, chromaselectorrect.w - 1);
            i32 chromaVMouseY = GM_clamp(chromaselectorrect.h - (mouseY_InGuiTarget - chromaselectorrect.y) - 1, 0, chromaselectorrect.h - 1);
            *saturation = float(chromaSMouseX) / float (chromaselectorrect.w - 1);
            *value = float(chromaVMouseY) / float (chromaselectorrect.h - 1);
        }
        else if (IsActive(id + 1))
        {
            i32 hueselectormousex = GM_clamp(mouseX_InGuiTarget - hueselectorrect.x, 0, hueselectorrect.w - 1);
            *hue = float(hueselectormousex) / float(hueselectorrect.w - 1);
        }
        else if (IsActive(id + 2))
        {
            i32 alphaselectormousex = GM_clamp(mouseX_InGuiTarget - alphaselectorrect.x, 0, alphaselectorrect.w - 1);
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
        layout->w = BACKBUFFER_WIDTH;
        layout->h = BACKBUFFER_HEIGHT;
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
        MemoryLinearInitialize(&drawRequestsFrameStorageBuffer, 1000000);

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
        ReadImage(bm_curses6x9, data_path("Kevin6x9.png").c_str());
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
        if (!anyElementHoveredThisFrame)
        {
            hoveredUI = null_ui_id;
        }
        anyElementHoveredThisFrame = false;
        keyboardInputASCIIKeycodeThisFrame.ResetCount();
        keyboardInputASCIIKeycodeThisFrame.ResetToZero();
        freshIdCounter = 0;
        __reservedTextMemoryIndexer = 0;

        drawRequestsFrameStorageBuffer.arenaOffset = 0;
        GUIDraw_NewFrame();
    }

    void Draw()
    {
        GUIDraw_DrawEverything();
    }

    void ProcessSDLEvent(const SDL_Event event)
    {
        Gui::NewFrame();

        switch (event.type)
        {
            case SDL_MOUSEMOTION:
            {
                // NOTE(Kevin): For game GUI where window aspect ratio does not match game aspect ratio, must
                //              map from window mouse pos to gui canvas mouse pos
                mouseX_InGuiTarget = int(float(mousepos.x) * (float(renderTargetGUI.width) / float(BACKBUFFER_WIDTH)));
                mouseY_InGuiTarget = int(float(mousepos.y) * (float(renderTargetGUI.height) / float(BACKBUFFER_HEIGHT)));
            }break;
            case SDL_KEYDOWN:
            {
                SDL_KeyboardEvent keyevent = event.key;
                SDL_Keycode keycodeASCII = keyevent.keysym.sym;
                keycodeASCII = ModifyASCIIBasedOnModifiers(keycodeASCII, keyevent.keysym.mod & (KMOD_LSHIFT | KMOD_RSHIFT));
                keyboardInputASCIIKeycodeThisFrame.PushBack(keycodeASCII);
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

static GPUMesh s_ui_mesh;
static GPUMesh s_text_mesh;
static GPUMesh s_colored_text_mesh;

#define MAX_DRAWCOLLECTIONS_ALLOWED 8

namespace Gui
{
    struct DrawCollectionMetaData
    {
        UIRect windowMask;
        int depth = 0;
    };

    static UIRect activeWindowMask;
    static NiceArray<std::vector<UIDrawRequest*>, MAX_DRAWCOLLECTIONS_ALLOWED + 1> DRAWQSTORAGE;
    static NiceArray<DrawCollectionMetaData, MAX_DRAWCOLLECTIONS_ALLOWED + 1> DRAWQUEUE_METADATA;
    static std::stack<std::vector<UIDrawRequest*>*> DRAWREQCOLLECTIONSTACK;

    void GUIDraw_InitResources()
    {
        GLCreateShaderProgram(main_ui_shader, main_ui_shader_vs, main_ui_shader_fs);
        GLCreateShaderProgram(rounded_corner_rect_shader, rounded_corner_rect_shader_vs, rounded_corner_rect_shader_fs);
        GLCreateShaderProgram(text_shader, text_shader_vs, text_shader_fs);
        GLCreateShaderProgram(colored_text_shader, colored_text_shader_vs, colored_text_shader_fs);
        CreateGPUMesh(&s_ui_mesh, nullptr, nullptr, 0, 0, 2, 2, 0, GL_DYNAMIC_DRAW);
        CreateGPUMesh(&s_text_mesh, nullptr, nullptr, 0, 0, 2, 2, 0, GL_DYNAMIC_DRAW);
        CreateGPUMesh(&s_colored_text_mesh, nullptr, nullptr, 0, 0, 2, 2, 3, GL_DYNAMIC_DRAW);

        ASSERT(DRAWQSTORAGE.count == 0);
        ASSERT(DRAWQUEUE_METADATA.count == 0);
        ASSERT(DRAWREQCOLLECTIONSTACK.empty());

        DRAWQSTORAGE.count++;
        DRAWQUEUE_METADATA.PushBack({ UIRect(0,0,9999,9999), 0 });
        DRAWREQCOLLECTIONSTACK.push(&DRAWQSTORAGE.Back());
    }

    void GUIDraw_NewFrame()
    {
        // clear draw queue
        DRAWQSTORAGE.count = 1;
        DRAWQUEUE_METADATA.count = 1;
        if (DRAWREQCOLLECTIONSTACK.size() > 1)
        {
            LogError("GUI BeginWindow and EndWindow don't match.");
        }
    }

    void GUIDraw_DrawEverything()
    {
        i32 kevGuiScreenWidth = renderTargetGUI.width;
        i32 kevGuiScreenHeight = renderTargetGUI.height;
        mat4 projectionMatrix = ProjectionMatrixOrthographicNoZ(0.f, (float)kevGuiScreenWidth, (float)kevGuiScreenHeight, 0.f);

        UseShader(main_ui_shader);
        GLBindMatrix4fv(main_ui_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        UseShader(rounded_corner_rect_shader);
        GLBindMatrix4fv(rounded_corner_rect_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        UseShader(text_shader);
        GLBindMatrix4fv(text_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        UseShader(colored_text_shader);
        GLBindMatrix4fv(colored_text_shader, "matrixOrtho", 1, projectionMatrix.ptr());

        // could sort so its O(n) but realistically how many windows am I going to have...
        int highestDepth = 0;
        for (int i = 0; i < DRAWQUEUE_METADATA.count; ++i)
            highestDepth = GM_max(highestDepth, DRAWQUEUE_METADATA.At(i).depth);

        // Draw collections of depth 0 to highestDepth except for base collection
        for (int depth = 0; depth <= highestDepth; ++depth)
        {
            for (int i = 1; i < DRAWQSTORAGE.count; ++i)
            {
                if (DRAWQUEUE_METADATA.At(i).depth == depth)
                {
                    activeWindowMask = DRAWQUEUE_METADATA.At(i).windowMask;
                    std::vector<UIDrawRequest*>& drawQueue = DRAWQSTORAGE.At(i);
                    for (auto drawCall : drawQueue)
                        drawCall->Draw();
                }
            }
        }

        // Draw base collection
        activeWindowMask = DRAWQUEUE_METADATA.At(0).windowMask;
        std::vector<UIDrawRequest*>& baseDrawQueue = DRAWQSTORAGE.At(0);
        for (auto drawCall : baseDrawQueue)
            drawCall->Draw();

        // Clear all collections
        for (int i = 0; i < DRAWQSTORAGE.count; ++i)
            DRAWQSTORAGE.At(i).clear();
    }

    void GUIDraw_PushDrawCollection(UIRect windowMask, int depth)
    {
        ASSERT(DRAWQSTORAGE.NotAtCapacity());
        DRAWQSTORAGE.count++;
        if (depth < 0)
            depth = (int)DRAWREQCOLLECTIONSTACK.size();
        else
            depth = GM_min(depth, MAX_DRAWCOLLECTIONS_ALLOWED);
        DRAWQUEUE_METADATA.PushBack({ windowMask, depth });
        DRAWREQCOLLECTIONSTACK.push(&DRAWQSTORAGE.Back());
    }

    void GUIDraw_PopDrawCollection()
    {
        ASSERT(DRAWREQCOLLECTIONSTACK.size() > 1);
        DRAWREQCOLLECTIONSTACK.pop();
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

        RebindGPUMeshData(&s_ui_mesh, vb, ib, ARRAY_COUNT(vb), ARRAY_COUNT(ib), GL_DYNAMIC_DRAW);
        RenderGPUMesh(s_ui_mesh);
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

        RebindGPUMeshData(&s_ui_mesh, vb, ib, ARRAY_COUNT(vb), ARRAY_COUNT(ib), GL_DYNAMIC_DRAW);
        RenderGPUMesh(s_ui_mesh);
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

        RebindGPUMeshData(&s_ui_mesh, vb, ib, ARRAY_COUNT(vb), ARRAY_COUNT(ib), GL_DYNAMIC_DRAW);
        RenderGPUMesh(s_ui_mesh);
    }

    void TextDrawRequest::Draw()
    {
        vtxt_setflags(VTXT_CREATE_INDEX_BUFFER);
        vtxt_clear_buffer();
        vtxt_move_cursor(x, y);
        switch (alignment)
        {
            case Align::Left:{
                vtxt_append_line(text, font.ptr, size);
            }break;
            case Align::Center:{
                vtxt_append_line_centered(text, font.ptr, size);
            }break;
            case Align::Right:{
                vtxt_append_line_align_right(text, font.ptr, size);
            }break;
        }
        vtxt_vertex_buffer _txt = vtxt_grab_buffer();
        if (_txt.vertices_array_count > 0)
            RebindGPUMeshData(&s_text_mesh, _txt.vertex_buffer, _txt.index_buffer, _txt.vertices_array_count, _txt.indices_array_count, GL_DYNAMIC_DRAW);
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

        RenderGPUMesh(s_text_mesh);
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
            RebindGPUMeshData(&s_colored_text_mesh, _txt.vertex_buffer, _txt.index_buffer, _txt.vertices_array_count, _txt.indices_array_count, GL_DYNAMIC_DRAW);

            mat4 matrixModel = mat4();
            UseShader(colored_text_shader);
            GLBindMatrix4fv(colored_text_shader, "matrixModel", 1, matrixModel.ptr());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, font.textureId);
            GLBind1i(colored_text_shader, "textureSampler0", 0);

            GLBind4i(colored_text_shader, "rectMask", rectMask.x, rectMask.y, rectMask.w, rectMask.h);
            GLBind1i(colored_text_shader, "rectMaskCornerRadius", rectMaskCornerRadius);

            GLBind4i(colored_text_shader, "windowMask", activeWindowMask.x, activeWindowMask.y, activeWindowMask.w, activeWindowMask.h);

            RenderGPUMesh(s_colored_text_mesh);
        }

        vtxt_clear_buffer();
    }
}

#undef ISANYOF1
#undef ISANYOF2
#undef ISANYOF3
#undef ISANYOF4

