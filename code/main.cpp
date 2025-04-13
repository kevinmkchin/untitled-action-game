#include "common.h"

#define GL3W_IMPLEMENTATION
#include <gl3w.h>

#define STB_SPRINTF_IMPLEMENTATION
#include <stb_sprintf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#define STB_DS_IMPLEMENTATION
#define STBDS_REALLOC(c,p,s) ::realloc(p,s)
#define STBDS_FREE(c,p)      ::free(p) // ensure global namespace
#include <stb_ds.h>

#define VERTEXT_IMPLEMENTATION
#include <vertext.h>

#if INTERNAL_BUILD
#include <renderdoc_app.h>
#endif

#include "mem.h"
// external
linear_arena_t StaticGameMemory;
linear_arena_t StaticLevelMemory;
linear_arena_t FrameMemory;
manualheap_arena_t JoltPhysicsMemory;

#include "debugmenu.h"
#include "utility.h"
#include "gpu_resources.h"
#include "shaders.h"
#include "resources.h"
#include "anim.h"
#include "game_assets.h"
#include "facebatch.h"
#include "filedialog.h"
#include "physics.h"
#include "physics_debug.h"
#include "cam.h"
#include "winged.h"
#include "lightmap.h"
#include "primitives.h"
#include "levelentities.h"
#include "leveleditor.h"
#include "saveloadlevel.h"
#include "gui.h"
#include "weapons.h"
#include "player.h"
#include "instanced.h"
#include "particles.h"
#include "game.h"
#include "enemy.h"
#include "nav.h"
#include "shader_helpers.h"

// Application state
SDL_Window *SDLMainWindow;
SDL_GLContext SDLGLContext;
bool ProgramShutdownRequested = false;
const float FixedDeltaTime = 1.0f / 60.0f;
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
support_renderer_t SupportRenderer;
app_state ApplicationState;

#if INTERNAL_BUILD
RENDERDOC_API_1_6_0 *RDOCAPI = NULL;
#endif // INTERNAL_BUILD


GPUShader Sha_GameLevel;
GPUShader Sha_ModelTexturedLit;
GPUShader Sha_ModelSkinnedLit;
GPUShader Sha_ModelInstancedLit;
GPUShader Sha_ParticlesDefault;
GPUShader Sha_Gun;
GPUShader Sha_Hemicube;
GPUShader Sha_EditorScene;
GPUShader Sha_EditorWireframe;
GPUShader Sha_EditorFaceSelected;
GPUShader FinalPassShader;
GPUFrameBuffer RenderTargetGame;
GPUFrameBuffer RenderTargetGUI;
GPUMeshIndexed FinalRenderOutputQuad;
float GAMEPROJECTION_NEARCLIP = 4.f; // even 2 works fine to remove z fighting
float GAMEPROJECTION_FARCLIP = 3200.f;


#include "anim.cpp"
#include "gpu_resources.cpp"
#include "shaders.cpp"
#include "resources.cpp"
#include "game_assets.cpp"
#include "physics_debug.cpp"
#include "shader_helpers.cpp"
#include "facebatch.cpp"
#include "filedialog.cpp"
#include "primitives.cpp"
#include "lightmap.cpp"
#include "winged.cpp"
#include "leveleditor.cpp"
#include "saveloadlevel.cpp"
#include "game.cpp"
#include "physics.cpp"
#include "levelentities.cpp"
#include "enemy.cpp"
#include "nav.cpp"
#include "cam.cpp"
#include "player.cpp"
#include "weapons.cpp"
#include "instanced.cpp"
#include "particles.cpp"
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


    GLLoadShaderProgramFromFile(Sha_GameLevel, 
        shader_path("__game_level.vert").c_str(), 
        shader_path("__game_level.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ModelTexturedLit, 
        shader_path("model_textured.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ModelSkinnedLit, 
        shader_path("model_skinned.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ModelInstancedLit, 
        shader_path("model_instanced_lit.vert").c_str(), 
        shader_path("model_instanced_lit.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ParticlesDefault, 
        shader_path("particles.vert").c_str(), 
        shader_path("particles.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_Gun, 
        shader_path("guns.vert").c_str(), 
        shader_path("guns.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_Hemicube, 
        shader_path("__patches_id.vert").c_str(), 
        shader_path("__patches_id.frag").c_str());
    GLCreateShaderProgram(Sha_EditorScene, 
        __editor_scene_shader_vs, 
        __editor_scene_shader_fs);
    GLCreateShaderProgram(Sha_EditorWireframe, 
        __editor_scene_wireframe_shader_vs, 
        __editor_scene_wireframe_shader_fs);
    GLCreateShaderProgram(Sha_EditorFaceSelected, 
        __editor_shader_face_selected_vs, 
        __editor_shader_face_selected_fs);
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

    SupportRenderer.Initialize();
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
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDLMainWindow = SDL_CreateWindow("game",
                                     1920,
                                     1080,
                                     SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

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
    // if (SDL_GL_SetSwapInterval(-1) == -1)
    // {
    //     LogWarning("Hardware does not support adaptive vsync.");
    //     SDL_GL_SetSwapInterval(1);
    // }

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

    Assets.LoadAllResources();

    InitializeGame();

    // LevelEditor.LoadMap(wd_path("testing.emf").c_str());
    // BuildGameMap(wd_path("buildtest.map").c_str());
    // LoadLevel(wd_path("buildtest.map").c_str());

    LoadLevel(wd_path("buildtest.map").c_str());
    // SwitchToLevelEditor();

    while (!ProgramShutdownRequested)
    {
        // Prepare for next frame
        FrameMemory.ArenaOffset = 0;
        TickTime();
        GUI::NewFrame();
        SupportRenderer.NewFrame();

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
        if (LevelEditor.IsActive)
        {
            LevelEditor.Tick();
            LevelEditor.Draw();
        }
        else
        {
            DoGameLoop();
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

    LevelEditor.Close();
    DestroyGame();

    SupportRenderer.Destroy();

    free(StaticGameMemory.Arena);
    free(StaticLevelMemory.Arena);

    SDL_DestroyWindow(SDLMainWindow);
    SDL_GL_DestroyContext(SDLGLContext);
    SDL_Quit();

    return 0;
}
