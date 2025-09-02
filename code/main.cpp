/** NOTES

    Keep It Simple

    Rendering Layer:
        renderer.h/cpp
        primitives.h/cpp
        instanced.h/cpp
        anim.h/cpp
        gpu_resources.h/cpp
        shaders.h/cpp
        gui.h/cpp
        facebatch.h/cpp

        leveleditor.h/cpp

*/

#include "common.h"

#include <gl3w.h>
#include <stb_sprintf.h>
#include <stb_image.h>
#include <stb_truetype.h>
#include <vertext.h>
#if INTERNAL_BUILD
#include <renderdoc_app.h>
static RENDERDOC_API_1_6_0 *RDOCAPI = NULL;
#endif

#include "mem.h"
#include "shaders.h"
#include "game_assets.h"
#include "filedialog.h"
#include "physics.h"
#include "primitives.h"
#include "leveleditor.h"
#include "saveloadlevel.h"
#include "gui.h"
#include "game.h"
#include "nav.h"
#include "renderer.h"

// Application state
static SDL_GLContext SDLGLContext;
static bool ProgramShutdownRequested = false;
float DeltaTime = 0.f;
static float RealDeltaTime = 0.f; // Unscaled and uncapped
static float GameTimeScale = 1.f;
static float CurrentTime = 0.f;
static float TimeSinceStart = 0.f;
static u32 MouseCurrent;
static u32 MousePressed;
static u32 MouseReleased;
static vec2 MouseDelta;
static ivec2 MousePos;
bool KeysCurrent[256] = {0};
static bool KeysPressed[256] = {0};
static bool KeysReleased[256] = {0};
static app_state AppState;

#include "debugmenu.cpp"

static void
TickTime()
{
    static std::chrono::high_resolution_clock::time_point timeAtLastUpdate = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float elapsedMs = (float)(std::chrono::duration_cast<std::chrono::microseconds>(now - timeAtLastUpdate)).count() * 0.001f;
    timeAtLastUpdate = now;
    float deltaTimeInSeconds = elapsedMs * 0.001f; // elapsed time in SECONDS
    float currentTimeInSeconds = (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() * 0.001f);
    CurrentTime = currentTimeInSeconds;
    RealDeltaTime = deltaTimeInSeconds;
    TimeSinceStart += RealDeltaTime;
    AppState.TimeSinceStart += RealDeltaTime;

    static const float CappedDeltaTime = 1.0f / 8.0f;
    DeltaTime = GM_min(RealDeltaTime, CappedDeltaTime);
    DeltaTime = DeltaTime * GameTimeScale;
}

static void
LoadRenderDoc()
{
#if INTERNAL_BUILD
    // === RENDER DOC API ===
    LoadLibrary("renderdoc.dll");
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&RDOCAPI);
        ASSERT(ret == 1);
        // rdoc_api->SetCaptureFilePathTemplate(...);
        LogMessage("Loaded renderdoc.dll");
    }
    else
    {
        LogWarning("Failed to load renderdoc.dll");
    }
#endif
}

static void
InitWindowAndGLContext()
{
    // SDL3
    ASSERT(SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_EVENTS));
    // OpenGL 4.3
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    AppState.BackBufferWidth = 1920;
    AppState.BackBufferHeight = 1080;
    AppState.SDLMainWindow = SDL_CreateWindow(
        "game", AppState.BackBufferWidth, AppState.BackBufferHeight,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    ASSERT(AppState.SDLMainWindow);
    SDL_SetWindowMinimumSize(AppState.SDLMainWindow, 200, 100);
    SDL_GetWindowSizeInPixels(AppState.SDLMainWindow,
        &AppState.BackBufferWidth,
        &AppState.BackBufferHeight);

    SDLGLContext = SDL_GL_CreateContext(AppState.SDLMainWindow);
    ASSERT(SDLGLContext);
    // GL3W
    if (gl3w_init())
    {
        fprintf(stderr, "Failed to initialize OpenGL\n");
        ASSERT(0);
    }
    LogMessage("GL_VERSION %s", glGetString(GL_VERSION));

    SDL_GL_SetSwapInterval(1);
}

static void
InitAudioMixer()
{
    SDL_AudioDeviceID AudioDeviceID = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
    SDL_AudioSpec AudioSpec;
    AudioSpec.format = SDL_AUDIO_S16LE; /**< Audio data format */
    AudioSpec.channels = 2; /**< Number of channels: 1 mono, 2 stereo, etc */
    AudioSpec.freq = 44100; /**< sample rate: sample frames per second */
    if (Mix_OpenAudio(AudioDeviceID, &AudioSpec) == false || Mix_Init(MIX_INIT_OGG) == 0)
        ASSERT(0);
}

static void 
ProcessSDLEvents()
{
    // MOUSE
    float mx, my;
    u32 mousestate = SDL_GetMouseState(&mx, &my); // mousepos only valid when mouse not relative mode
    MousePos.x = (int)mx;
    MousePos.y = (int)my;
    u32 mousechanged = MouseCurrent ^ mousestate;
    MousePressed = mousechanged & mousestate;
    MouseReleased = mousechanged & MouseCurrent;
    MouseCurrent = mousestate;
    SDL_GetRelativeMouseState(&MouseDelta.x, &MouseDelta.y);

    // KEYBOARD
    const bool *keystate = SDL_GetKeyboardState(NULL);
    bool keyschanged[256];
    for (int i=0;i<256;++i)
    {
        keyschanged[i] = KeysCurrent[i] ^ keystate[i];
        KeysPressed[i] = keyschanged[i] & keystate[i];
        KeysReleased[i] = keyschanged[i] & KeysCurrent[i];
        KeysCurrent[i] = keystate[i];
    }

    AppState.MouseCurrent = MouseCurrent;
    AppState.MousePressed = MousePressed;
    AppState.MouseReleased = MouseReleased;
    AppState.MouseDelta = MouseDelta;
    AppState.MousePos = MousePos;
    memcpy(AppState.KeysCurrent, KeysCurrent, 256);
    memcpy(AppState.KeysPressed, KeysPressed, 256);
    memcpy(AppState.KeysReleased, KeysReleased, 256);

    // EVENT HANDLING
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            {
                SDL_GetWindowSizeInPixels(AppState.SDLMainWindow, 
                    &AppState.BackBufferWidth,
                    &AppState.BackBufferHeight);
                UpdateRenderTargetSizes(&AppState);
                break;
            }

            case SDL_EVENT_QUIT:
            {
                ProgramShutdownRequested = true;
                break;
            }

            case SDL_EVENT_KEY_DOWN:
            {
                SDL_Keycode sdlkey = event.key.key;

                if (sdlkey == SDLK_RETURN && SDL_GetModState() & SDL_KMOD_LALT)
                {
                    // SDL_GetFullscreenDisplayModes();
                    if (SDL_GetWindowFlags(AppState.SDLMainWindow) & SDL_WINDOW_FULLSCREEN)
                        SDL_SetWindowFullscreen(AppState.SDLMainWindow, false);
                    else
                        SDL_SetWindowFullscreen(AppState.SDLMainWindow, true);
                    event.type = 0;
                }
                break;
            }
        }

        GUI::ProcessSDLEvent(&AppState, event);

        if (CurrentDebugMode == DEBUG_MODE_CONSOLE)
            SendInputToConsole(event);

    }
}

int main(int argc, char* argv[])
{
    StaticGameMemory.Init(128000000);
    StaticLevelMemory.Init(32000000);
    FrameMemory.Init(32000000);
    JoltPhysicsMemory.Init(128000000);
    SetupConsoleAndBindProcedures();
    LoadRenderDoc();
    InitWindowAndGLContext();
    InitAudioMixer();
    GUI::Init();
    AcquireRenderingResources(&AppState);
    AppState.GameState = new_InGameMemory(game_state)();
    AppState.GameState->AppState = &AppState;
    AppState.PrimitivesRenderer = new_InGameMemory(support_renderer_t)();
    AppState.PrimitivesRenderer->Initialize();
    AppState.LevelEditor = new_InGameMemory(level_editor_t)();
    AppState.LevelEditor->AppState = &AppState;
    AppState.LevelEditor->SupportRenderer = AppState.PrimitivesRenderer;
    Assets.LoadAllResources();
    InitializeGame(&AppState);

    // LevelEditor.LoadMap(wd_path("testing.emf").c_str());
    // BuildGameMap(wd_path(AppState.LevelEditor, "buildtest.map").c_str());
    // LoadLevel(wd_path("buildtest.map").c_str());

    SDL_SetWindowRelativeMouseMode(AppState.SDLMainWindow, true);
    LoadLevel(wd_path("buildtest.map").c_str());
    // SwitchToLevelEditor();

    while (!ProgramShutdownRequested)
    {
        FrameMemory.ArenaOffset = 0;
        TickTime();
        GUI::NewFrame();
        AppState.PrimitivesRenderer->NewFrame();

        ProcessSDLEvents();

#if INTERNAL_BUILD
        if (RDOCAPI && KeysPressed[SDL_SCANCODE_HOME])
            if (RDOCAPI->ShowReplayUI() == 0)
                RDOCAPI->LaunchReplayUI(1, "");
#endif // INTERNAL_BUILD

        ShowDebugConsole();

        if (AppState.LevelEditor->IsActive)
        {
            AppState.LevelEditor->Tick();
            AppState.LevelEditor->Draw();
        }
        else
        {
            DoGameLoop(&AppState);
        }

        RenderFrame(&AppState);
        SDL_GL_SwapWindow(AppState.SDLMainWindow);
        // NOTE(Kevin) 2025-04-02: Even on SDL3 there's microstutters with Windowed VSYNC
        //                         Still feels much better with DwmFlush after SwapBuffers
        // NOTE(Kevin) 2025-04-08: FullscreenEx VSYNC+DwmFlush feels the best. FullscreenEx VSYNC
        //                         without DwmFlush seems to have more latency and stutter/lag.
        //                         Windowed+vsync+dwmflush still Stutters after long stretches
        //                         of no Stutter...
        int SwapInterval = 0;
        if (SDL_GL_GetSwapInterval(&SwapInterval) && SwapInterval == 1)
            DwmFlush();
    }

    AppState.LevelEditor->Close();
    DestroyGame();
    ReleaseRenderingResources();
    AppState.PrimitivesRenderer->Destroy();
    free(StaticGameMemory.Arena);
    free(StaticLevelMemory.Arena);
    SDL_DestroyWindow(AppState.SDLMainWindow);
    SDL_GL_DestroyContext(SDLGLContext);
    SDL_Quit();
    return 0;
}
