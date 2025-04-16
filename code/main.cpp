/** NOTES

    Keep It Simple

    Rendering:
        renderer.h/cpp
        primitives.h/cpp
        instanced.h/cpp
        leveleditor.h/cpp
        gui.h/cpp
        anim.h/cpp
        facebatch.h/cpp
        game.h/cpp
        gpu_resources.h/cpp

*/

#include "common.h"

#include <gl3w.h>
#include <stb_sprintf.h>
#include <stb_image.h>
#include <stb_truetype.h>
#include <vertext.h>
#if INTERNAL_BUILD
#include <renderdoc_app.h>
#endif

#include "mem.h"
#include "debugmenu.h"
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
SDL_Window *SDLMainWindow;
SDL_GLContext SDLGLContext;
bool ProgramShutdownRequested = false;
float DeltaTime = 0.f;
float RealDeltaTime = 0.f; // Unscaled and uncapped
float GameTimeScale = 1.f;
float CurrentTime = 0.f;
float TimeSinceStart = 0.f;
u32 MouseCurrent;
u32 MousePressed;
u32 MouseReleased;
vec2 MouseDelta;
ivec2 MousePos;
bool KeysCurrent[256] = {0};
bool KeysPressed[256] = {0};
bool KeysReleased[256] = {0};
i32 BackbufferWidth = -1;
i32 BackbufferHeight = -1;
char CurrentWorkingDirectory[128];
app_state ApplicationState;

#if INTERNAL_BUILD
RENDERDOC_API_1_6_0 *RDOCAPI = NULL;
#endif // INTERNAL_BUILD


GPUShader Sha_ModelTexturedLit;
GPUShader Sha_Gun;
GPUShader Sha_Hemicube;
GPUShader FinalPassShader;
GPUFrameBuffer RenderTargetGame;
GPUFrameBuffer RenderTargetGUI;
GPUMeshIndexed FinalRenderOutputQuad;

#include "debugmenu.cpp"


static void RenderGUILayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGUI.fbo);
    glViewport(0, 0, RenderTargetGUI.width, RenderTargetGUI.height);
//    glDepthRange(0.00001f, 10.f); I should just be doing painters algorithm
    glClearColor(RGB255TO1(244, 194, 194), 0.0f);
//    glClearDepth(10.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST); // I forgot why the fuck I'm disabling depth test when using glDepthRange

    GUI::Draw(RenderTargetGUI.width, RenderTargetGUI.height);
}

static void FinalRenderToBackBuffer()
{
    UseShader(FinalPassShader);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, BackbufferWidth, BackbufferHeight);
//    glDepthRange(0, 10);
    glClearColor(RGB255TO1(0, 0, 0), 1.f);
//    glClearDepth(1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    // Draw game frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderTargetGame.colorTexId);
    RenderGPUMeshIndexed(FinalRenderOutputQuad);

    // Draw GUI frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderTargetGUI.colorTexId);
    RenderGPUMeshIndexed(FinalRenderOutputQuad);

    //    // Draw Debug UI frame
    //    glActiveTexture(GL_TEXTURE0);
    //    glBindTexture(GL_TEXTURE_2D, debugUILayer.colorTexId);
    //    RenderMesh(screenSizeQuad);

    GLHasErrors();
}

const char* __finalpass_shader_vs =
    "#version 330\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec2 uv;\n"
    "out vec2 texcoord;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(pos, 0, 1.0);\n"
    "    texcoord = uv;\n"
    "}\n";

const char* __finalpass_shader_fs =
    "#version 330\n"
    "uniform sampler2D screen_texture;\n"
    "in vec2 texcoord;\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "    vec4 in_color = texture(screen_texture, texcoord);\n"
    "    if(in_color.w < 0.001)\n"
    "    {\n"
    "        discard;\n"
    "    }\n"
    "    color = in_color;\n"
    "}\n";

static void InitGameRenderer()
{

    // alpha blending func: (srcRGB) * srcA + (dstRGB) * (1 - srcA)  = final color output
    // alpha blending func: (srcA) * a + (dstA) * 1 = final alpha output
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glFrontFace(GL_CCW); // OpenGL default is GL_CCW


    SDL_GetWindowSizeInPixels(SDLMainWindow, &BackbufferWidth, &BackbufferHeight);
    RenderTargetGame.width = BackbufferWidth;
    RenderTargetGame.height = BackbufferHeight;
    CreateGPUFrameBuffer(&RenderTargetGame);
    RenderTargetGUI.width = BackbufferWidth / 2;
    RenderTargetGUI.height = BackbufferHeight / 2;
    CreateGPUFrameBuffer(&RenderTargetGUI);


    GLLoadShaderProgramFromFile(Sha_ModelTexturedLit, 
        shader_path("model_textured.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_Gun, 
        shader_path("guns.vert").c_str(), 
        shader_path("guns.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_Hemicube, 
        shader_path("__patches_id.vert").c_str(), 
        shader_path("__patches_id.frag").c_str());
    GLCreateShaderProgram(FinalPassShader, 
        __finalpass_shader_vs, 
        __finalpass_shader_fs);


    float refQuadVertices[16] = {
        //  x   y    u    v
        -1.f, -1.f, 0.f, 0.f,
        1.f, -1.f, 1.f, 0.f,
        -1.f, 1.f, 0.f, 1.f,
        1.f, 1.f, 1.f, 1.f
    };
    u32 refQuadIndices[6] = {
        0, 1, 3,
        0, 3, 2
    };
    CreateGPUMeshIndexed(&FinalRenderOutputQuad, refQuadVertices, refQuadIndices, 16, 6, 2, 2, 0, GL_STATIC_DRAW);
}


static void TickTime()
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
    ApplicationState.TimeSinceStart += RealDeltaTime;

    static const float CappedDeltaTime = 1.0f / 8.0f;
    DeltaTime = GM_min(RealDeltaTime, CappedDeltaTime);
    DeltaTime = DeltaTime * GameTimeScale;
}

static bool InitializeApplication()
{
    _getcwd(CurrentWorkingDirectory, 128);

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

    ProgramShutdownRequested = false;

    if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_EVENTS) == false) return false;

    // OpenGL 4.6
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDLMainWindow = SDL_CreateWindow("game",
                                     1920,
                                     1080,
                                     SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    ApplicationState.SDLMainWindow = SDLMainWindow;

    SDLGLContext = SDL_GL_CreateContext(SDLMainWindow);

    if (SDLMainWindow == nullptr || SDLGLContext == nullptr) return false;

    if (gl3w_init())
    {
        fprintf(stderr, "Failed to initialize OpenGL\n");
        return false;
    }
    LogMessage("GL_VERSION %s", glGetString(GL_VERSION));

    SDL_SetWindowMinimumSize(SDLMainWindow, 200, 100);
    SDL_GL_SetSwapInterval(1);

    SDL_AudioDeviceID AudioDeviceID = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
    SDL_AudioSpec AudioSpec;
    AudioSpec.format = SDL_AUDIO_S16LE; /**< Audio data format */
    AudioSpec.channels = 2; /**< Number of channels: 1 mono, 2 stereo, etc */
    AudioSpec.freq = 44100; /**< sample rate: sample frames per second */
    if (Mix_OpenAudio(AudioDeviceID, &AudioSpec) == false || Mix_Init(MIX_INIT_OGG) == 0)
        return false;

    GUI::Init();

    return true;
}

static void ProcessSDLEvents()
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

    ApplicationState.MouseCurrent = MouseCurrent;
    ApplicationState.MousePressed = MousePressed;
    ApplicationState.MouseReleased = MouseReleased;
    ApplicationState.MouseDelta = MouseDelta;
    ApplicationState.MousePos = MousePos;
    memcpy(ApplicationState.KeysCurrent, KeysCurrent, 256);
    memcpy(ApplicationState.KeysPressed, KeysPressed, 256);
    memcpy(ApplicationState.KeysReleased, KeysReleased, 256);
    ApplicationState.BackBufferWidth = BackbufferWidth;
    ApplicationState.BackBufferHeight = BackbufferHeight;
    ApplicationState.GUIRenderTargetWidth = RenderTargetGUI.width;
    ApplicationState.GUIRenderTargetHeight = RenderTargetGUI.height;

    // EVENT HANDLING
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            {
                SDL_GetWindowSizeInPixels(SDLMainWindow, &BackbufferWidth, &BackbufferHeight);
                UpdateGPUFrameBufferSize(&RenderTargetGame, BackbufferWidth, BackbufferHeight);
                UpdateGPUFrameBufferSize(&RenderTargetGUI, BackbufferWidth / 2, BackbufferHeight / 2);
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
                    if (SDL_GetWindowFlags(SDLMainWindow) & SDL_WINDOW_FULLSCREEN)
                        SDL_SetWindowFullscreen(SDLMainWindow, false);
                    else
                        SDL_SetWindowFullscreen(SDLMainWindow, true);
                    event.type = 0;
                }
                break;
            }
        }

        GUI::ProcessSDLEvent(&ApplicationState, event);

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

    if (!InitializeApplication()) return -1;

    InitGameRenderer();

    AcquireRenderingResources();
    ApplicationState.GameState = new_InGameMemory(game_state)();
    ApplicationState.GameState->AppState = &ApplicationState;
    ApplicationState.RenderTargetGame = &RenderTargetGame;
    ApplicationState.PrimitivesRenderer = new_InGameMemory(support_renderer_t)();
    ApplicationState.PrimitivesRenderer->Initialize();
    ApplicationState.LevelEditor = new_InGameMemory(level_editor_t)();
    ApplicationState.LevelEditor->AppState = &ApplicationState;
    ApplicationState.LevelEditor->SupportRenderer = ApplicationState.PrimitivesRenderer;
    ApplicationState.LevelEditor->RenderTargetGame = RenderTargetGame;

    Assets.LoadAllResources();

    InitializeGame(&ApplicationState);

    // LevelEditor.LoadMap(wd_path("testing.emf").c_str());
    // BuildGameMap(wd_path(ApplicationState.LevelEditor, "buildtest.map").c_str());
    // LoadLevel(wd_path("buildtest.map").c_str());

    SDL_SetWindowRelativeMouseMode(SDLMainWindow, true);
    LoadLevel(wd_path("buildtest.map").c_str());
    // SwitchToLevelEditor();

    while (!ProgramShutdownRequested)
    {
        // Prepare for next frame
        FrameMemory.ArenaOffset = 0;
        TickTime();
        GUI::NewFrame();
        ApplicationState.PrimitivesRenderer->NewFrame();

        // Poll and process events
        ProcessSDLEvents();

#if INTERNAL_BUILD
        if (RDOCAPI && KeysPressed[SDL_SCANCODE_HOME])
            if (RDOCAPI->ShowReplayUI() == 0)
                RDOCAPI->LaunchReplayUI(1, "");
#endif // INTERNAL_BUILD

        // Process console commands
        ShowDebugConsole();

        // Game logic
        if (ApplicationState.LevelEditor->IsActive)
        {
            ApplicationState.LevelEditor->Tick();
            ApplicationState.LevelEditor->Draw();
        }
        else
        {
            DoGameLoop(&ApplicationState);
        }

        // Draw calls
        RenderGUILayer();
        FinalRenderToBackBuffer();

        // Swap buffers
        SDL_GL_SwapWindow(SDLMainWindow);

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

    ApplicationState.LevelEditor->Close();
    DestroyGame();
    ReleaseRenderingResources();
    ApplicationState.PrimitivesRenderer->Destroy();

    free(StaticGameMemory.Arena);
    free(StaticLevelMemory.Arena);

    SDL_DestroyWindow(SDLMainWindow);
    SDL_GL_DestroyContext(SDLGLContext);
    SDL_Quit();

    return 0;
}
