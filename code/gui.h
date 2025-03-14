#pragma once

/* NOTES

Decouple DrawReqCollection concept from windows concept
DrawReqCollection simply has a mask and a depth
maybe even let user keep reference to a DrawReqCollection handle and they
can add more elements to it later in the frame


*/


#define null_ui_id 0
typedef u64 ui_id;

namespace GUI
{
    // extern NiceArray<SDL_Keycode, 32> keyboardInputASCIIKeycodeThisFrame;
    // extern vec4 style_textColor;

    // Use to prevent mouse events from registering through UI or windows
    extern bool anyElementHovered;
    extern bool anyElementActive;
    extern bool anyWindowHovered;
    // Mouse position in GUI render target resolution
    extern int MouseXInGUI;
    extern int MouseYInGUI;

    enum Align
    {
        LEFT,
        CENTER,
        RIGHT
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
    void NewFrame(); // before each frame
    void ProcessSDLEvent(const SDL_Event evt); // update input state
    void Draw(); // flush draw requests to GPU

    /* Immediate-mode state helpers */
    bool IsActive(ui_id id);
    bool IsHovered(ui_id id);
    void SetActive(ui_id id);
    void RequestSetHovered(ui_id id);
    bool MouseWentUp();
    bool MouseWentDown();
    bool MouseInside(const UIRect& rect);
    // IsMouseInsideWindow?

    /* Utility helpers */
    int GetFontSize();

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
    void PrimitiveCheckbox(ui_id id, UIRect rect, int inset, bool *value, vec4 background, vec4 foreground);
    bool PrimitiveLabelledButton(UIRect rect, const char* label, Align textAlignment);


    /* Windows
    * For aligning GUI elements like a DearImGui window
    * Check anyWindowHovered to prevent propagating mouse clicks to underlying program
    *   depth of -1 means the window will set its depth to the window stack count (windows can begin in other windows)
    * TODO think about:
    * Can be set to capture focus?
    * Can be set to collapse?
    * */
    void BeginWindow(UIRect windowRect, vec4 bgcolor = vec4(0.05f, 0.05f, 0.05f, 1.0f), int depth = -1);
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
    void EditorImage(u32 glTextureId, ivec2 size);//, vec2 uv0 = vec2(0.f,0.f), vec2 uv1 = vec2(1.f,1.f));
    bool EditorImageButton(u32 glTextureId, ivec2 size);//, vec2 uv0 = vec2(0.f,0.f), vec2 uv1 = vec2(1.f,1.f));
    bool EditorLabelledButton(const char *label, int minwidth = 50);
    void EditorIncrementableIntegerField(const char *label, int *v, int increment = 1);
    void EditorIncrementableFloatField(const char *label, float *v, float increment = 0.1f);
    void EditorCheckbox(const char *label, bool *value);

    // EditorSelectable returns true once when it gets selected
    bool EditorSelectable(const char *label, bool *selected);
    // I forgot why i did this:
    bool EditorSelectable_2(const char *label, bool *selected);
    bool EditorSelectableRect(vec4 colorRGBA, bool *selected, int id);

    void EditorBeginListBox();
    void EditorEndListBox();

    void EditorBeginHorizontal();
    void EditorEndHorizontal();

    // Grids
    void EditorBeginGrid(int gridwidth, int gridheight);
    void EditorBeginGridItem(int itemw, int itemh);
    void EditorEndGridItem();
    void EditorEndGrid();

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


    // some helpers from GUI drawing code
    u8 GetCurrentDrawingDepth();
}

namespace GUI {

    // GUI rendering code

    struct UIDrawRequest;

    void GUIDraw_InitResources();
    void GUIDraw_DrawEverything();
    void GUIDraw_NewFrame();

    void GUIDraw_PushDrawCollection(UIRect windowMask, int depth);
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
        Align alignment = Align::LEFT;
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
