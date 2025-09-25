#include "shooter_renderer_opengl.h"
#include "shooter_shared.h"

#define GL3W_IMPLEMENTATION
#include "../ext/gl3w.h"

internal void OpenGLInit(open_gl *OpenGL)
{

}

internal void OpenGLEndFrame(open_gl *OpenGL)
{
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
}

#ifdef SHOOTER_PLATFORM_WIN32

#define WGL_CONTEXT_MAJOR_VERSION_ARB           0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB           0x2092
#define WGL_CONTEXT_LAYER_PLANE_ARB             0x2093
#define WGL_CONTEXT_FLAGS_ARB                   0x2094
#define WGL_CONTEXT_PROFILE_MASK_ARB            0x9126

#define WGL_CONTEXT_DEBUG_BIT_ARB               0x0001
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB  0x0002

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB        0x00000001
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002

#define WGL_DRAW_TO_WINDOW_ARB                  0x2001
#define WGL_ACCELERATION_ARB                    0x2003
#define WGL_SUPPORT_OPENGL_ARB                  0x2010
#define WGL_DOUBLE_BUFFER_ARB                   0x2011
#define WGL_PIXEL_TYPE_ARB                      0x2013

#define WGL_TYPE_RGBA_ARB                       0x202B
#define WGL_FULL_ACCELERATION_ARB               0x2027

#define WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB        0x20A9

typedef HGLRC WINAPI wgl_create_context_attribs_arb(HDC hDC, HGLRC hShareContext, const int *attribList);

typedef BOOL WINAPI wgl_choose_pixel_format_arb(HDC hdc,
                                                const int *piAttribIList,
                                                const FLOAT *pfAttribFList,
                                                UINT nMaxFormats,
                                                int *piFormats,
                                                UINT *nNumFormats);

typedef BOOL WINAPI wgl_swap_interval_ext(int interval);

typedef const char * WINAPI wgl_get_extensions_string_ext(void);

global wgl_create_context_attribs_arb *wglCreateContextAttribsARB;
global wgl_choose_pixel_format_arb *wglChoosePixelFormatARB;
global wgl_swap_interval_ext *wglSwapIntervalEXT;
global wgl_get_extensions_string_ext *wglGetExtensionsStringEXT;

internal void
Win32SetPixelFormat(open_gl *OpenGL, HDC WindowDC)
{
    int SuggestedPixelFormatIndex = 0;
    GLuint ExtendedPick = 0;

    if (wglChoosePixelFormatARB)
    {
        int IntAttribList[] =
        {
            WGL_DRAW_TO_WINDOW_ARB, GL_TRUE, // 0
            WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB, // 1
            WGL_SUPPORT_OPENGL_ARB, GL_TRUE, // 2
            WGL_DOUBLE_BUFFER_ARB, GL_TRUE, // 3
            WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB, // 4
            WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB, GL_TRUE, // 5
            0,
        };
        
        if (!OpenGL->SupportsSRGBFramebuffer)
        {
            IntAttribList[10] = 0;
        }

        wglChoosePixelFormatARB(WindowDC, IntAttribList, 0, 1, &SuggestedPixelFormatIndex, &ExtendedPick);
    }
    
    if (!ExtendedPick)
    {
        PIXELFORMATDESCRIPTOR DesiredPixelFormat = {};
        DesiredPixelFormat.nSize = sizeof(DesiredPixelFormat);
        DesiredPixelFormat.nVersion = 1;
        DesiredPixelFormat.iPixelType = PFD_TYPE_RGBA;
        DesiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL|PFD_DRAW_TO_WINDOW|PFD_DOUBLEBUFFER;
        DesiredPixelFormat.cColorBits = 32;
        DesiredPixelFormat.cAlphaBits = 8;
        DesiredPixelFormat.cDepthBits = 24;
        DesiredPixelFormat.iLayerType = PFD_MAIN_PLANE;
        
        SuggestedPixelFormatIndex = ChoosePixelFormat(WindowDC, &DesiredPixelFormat);
    }

    PIXELFORMATDESCRIPTOR SuggestedPixelFormat;
    // NOTE(Kevin): Technically you do not need to call DescribePixelFormat here,
    // as SetPixelFormat doesn't actually need it to be filled out properly.
    DescribePixelFormat(WindowDC, SuggestedPixelFormatIndex, sizeof(SuggestedPixelFormat), &SuggestedPixelFormat);
    SetPixelFormat(WindowDC, SuggestedPixelFormatIndex, &SuggestedPixelFormat);
}

internal void Win32LoadWGLExtensions(open_gl *OpenGL)
{
    WNDCLASSA WindowClass = {};
    WindowClass.lpfnWndProc = DefWindowProcA;
    WindowClass.hInstance = GetModuleHandle(0);
    WindowClass.lpszClassName = "ShooterGameWGLLoader";

    if (RegisterClassA(&WindowClass))
    {
        HWND Window = CreateWindowExA(
            0,
            WindowClass.lpszClassName,
            "Shooter Game",
            0,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            0,
            0,
            WindowClass.hInstance,
            0);
        
        HDC WindowDC = GetDC(Window);

        Win32SetPixelFormat(OpenGL, WindowDC);

        HGLRC OpenGLRC = wglCreateContext(WindowDC);

        if (wglMakeCurrent(WindowDC, OpenGLRC))
        {
            wglChoosePixelFormatARB =
                (wgl_choose_pixel_format_arb *)wglGetProcAddress("wglChoosePixelFormatARB");
            wglCreateContextAttribsARB =
                (wgl_create_context_attribs_arb *)wglGetProcAddress("wglCreateContextAttribsARB");
            wglGetExtensionsStringEXT = 
                (wgl_get_extensions_string_ext *)wglGetProcAddress("wglGetExtensionsStringEXT");

            if (wglGetExtensionsStringEXT)
            {
                char *Extensions = (char *)wglGetExtensionsStringEXT();
                char *At = Extensions;
                while (*At)
                {
                    while (IsWhitespace(*At)) {++At;}
                    char *End = At;
                    while (*End && !IsWhitespace(*End)) {++End;}

                    umm Count = End - At;

                    if (0) {}
                    else if (StringsAreEqual(Count, At, "WGL_EXT_framebuffer_sRGB")) { OpenGL->SupportsSRGBFramebuffer = true; }
                    else if (StringsAreEqual(Count, At, "WGL_ARB_framebuffer_sRGB")) { OpenGL->SupportsSRGBFramebuffer = true; }

                    At = End;
                }
            }
            
            wglMakeCurrent(0, 0);
        }

        wglDeleteContext(OpenGLRC);

        ReleaseDC(Window, WindowDC);

        DestroyWindow(Window);
    }
}

internal void *
Win32RendererAlloc(umm Size)
{
    void *Result = VirtualAlloc(0, Size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    return Result;
}

internal void
PlatformOpenGLSetVSync(open_gl *OpenGL, bool32 VSyncEnabled)
{
    if (!wglSwapIntervalEXT)
    {
        wglSwapIntervalEXT = (wgl_swap_interval_ext *)wglGetProcAddress("wglSwapIntervalEXT");
    }

    if (wglSwapIntervalEXT)
    {
        wglSwapIntervalEXT(VSyncEnabled ? 1 : 0);
    }
}

global int Win32OpenGLAttribs[] =
{
    WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
    WGL_CONTEXT_MINOR_VERSION_ARB, 3,
    WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB
#if HANDMADE_INTERNAL
        |WGL_CONTEXT_DEBUG_BIT_ARB
#endif
    ,
#if 0
    WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
#else
    WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
#endif
    0,
};

internal open_gl *
Win32InitOpenGL(HDC WindowDC /* TODO(Kevin): platform limits?*/)
{
    open_gl *OpenGL = (open_gl *)Win32RendererAlloc(sizeof(open_gl));
    Win32LoadWGLExtensions(OpenGL);
    Win32SetPixelFormat(OpenGL, WindowDC);

    bool32 ModernContext = true;
    HGLRC OpenGLRC = 0;
    if (wglCreateContextAttribsARB)
    {
        OpenGLRC = wglCreateContextAttribsARB(WindowDC, 0, Win32OpenGLAttribs);
    }

    if (!OpenGLRC)
    {
        ModernContext = false;
        OpenGLRC = wglCreateContext(WindowDC);
    }

    Assert(ModernContext);
    // TODO(Kevin): Logging / Messagebox to say hardware does not support required opengl version

    if (wglMakeCurrent(WindowDC, OpenGLRC))
    {
        // OpenGL rendering context ready

#define Win32GetOpenGLFunction(Name) OpenGL->Name = (type_##Name *)wglGetProcAddress(#Name)
        // opengl_info Info = OpenGLGetInfo(OpenGL, ModernContext);
        // if(Info.OpenGL_ARB_framebuffer_object)
        // {
        //     Win32GetOpenGLFunction(glBindFramebuffer);
        //     Win32GetOpenGLFunction(glGenFramebuffers);
        //     Win32GetOpenGLFunction(glFramebufferTexture2D);
        //     Win32GetOpenGLFunction(glCheckFramebufferStatus);
        // }

        // NOTE(Kevin): gl3w load procedures (https://github.com/skaslev/gl3w)
        if (gl3w_init())
        {
            // TODO(Kevin): Logging
            Assert(0);
        }
        // TODO(Kevin): Logging
        // LogMessage("GL_VERSION %s", glGetString(GL_VERSION));
        
        PlatformOpenGLSetVSync(OpenGL, true);
        OpenGLInit(OpenGL);
    }

    return OpenGL;
}

renderer *
Win32LoadRenderer(HDC WindowDC)
{
    renderer *Renderer = (renderer *)Win32InitOpenGL(WindowDC);
    return Renderer;
}

void
Win32EndFrame(renderer *Renderer)
{
    OpenGLEndFrame((open_gl *)Renderer);
    SwapBuffers(wglGetCurrentDC());
}

#endif // SHOOTER_PLATFORM_WIN32
