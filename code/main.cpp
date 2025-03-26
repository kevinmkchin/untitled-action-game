/*

Something fucking cool.
Priorty #1 is building the game, not the engine/tech
Handcrafted with love.


== NOTES ==

    64x64 pixel texture for each 32x32 unit in game looks decent

    Embrace the fact that lighting will be crude and imperfect. The visual artifacts 
    is part of the charm of my game and engine. Something that differentiates it from 
    the perfect crispy lighting of engines like Godot or Unity.

    Ultimately, the game will very much have my identity. From some UI looking crusty,
    or enemy animations being janky, but that personal touch is part of the charm of 
    an indie game like this.


*/

#include <cstdint>
#include <cassert>
#include <fstream>
#include <string>
#include <chrono>
#include <iterator>
#include <algorithm>

#include "BUILDINFO.H"

#if MESA_WINDOWS
#define NOMINMAX
#include <windows.h>
#include <dwmapi.h>
#include <direct.h>
#endif

#define GL3W_IMPLEMENTATION
#include <gl3w.h>
#define MESA_USING_GL3W
#define GL_VERSION_4_3_OR_HIGHER

#include <SDL.h>
#include <SDL_mixer.h>

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
#include <gmath.h>
#if INTERNAL_BUILD
#include <renderdoc_app.h>
#endif

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <stack>
#include <memory>
#include <utility>
#include <thread>


typedef uint8_t       u8;
typedef uint16_t      u16;
typedef uint32_t      u32;
typedef uint64_t      u64;
typedef int8_t        i8;
typedef int16_t       i16;
typedef int32_t       i32;
typedef int64_t       i64;
typedef uint_fast8_t  u8f;
typedef uint_fast16_t u16f;
typedef uint_fast32_t u32f;
typedef int_fast8_t   i8f;
typedef int_fast16_t  i16f;
typedef int_fast32_t  i32f;
typedef i16           bool16;
typedef i32           bool32;


#if (defined _MSC_VER)
#define ASSERT(predicate) if(!(predicate)) { __debugbreak(); }
#else
#define ASSERT(predicate) if(!(predicate)) { __builtin_trap(); }
#endif
#if INTERNAL_BUILD
    #define ASSERTDEBUG(predicate) ASSERT(predicate)
#else
    #define ASSERTDEBUG(predicate)
#endif

inline std::string wd_path() { return std::string(PROJECT_WORKING_DIR); }
inline std::string wd_path(const std::string& name) { return wd_path() + std::string(name); }
inline std::string shader_path() { return wd_path() + "shaders/"; }
inline std::string shader_path(const std::string& name) { return wd_path() + "shaders/" + name; }
inline std::string model_path() { return wd_path() + "models/"; }
inline std::string model_path(const std::string& name) { return wd_path() + "models/" + name; }
inline std::string texture_path() { return wd_path() + "textures/"; }
inline std::string texture_path(const std::string& name) { return wd_path() + "textures/" + name; }
inline std::string sfx_path() { return wd_path() + "sfx/"; }
inline std::string sfx_path(const std::string& name) { return wd_path() + "sfx/" + name; }
inline std::string entity_icons_path() { return wd_path() + "entity_icons/"; }
inline std::string entity_icons_path(const std::string& name) { return wd_path() + "entity_icons/" + name; }

#define ARRAY_COUNT(a) (sizeof(a) / (sizeof(a[0])))

// fflush this shit for CLion
#define LogMessage(...)                        \
    do {                                       \
        fprintf(stdout, __VA_ARGS__);          \
        fprintf(stdout, "\n");                 \
        fflush(stdout);                        \
    } while (false)
#define LogWarning(...)                        \
    do {                                       \
        fprintf(stderr, __VA_ARGS__);          \
        fprintf(stderr, "\n");                 \
        fflush(stderr);                        \
    } while (false)
#define LogError(...)                          \
    do {                                       \
        fprintf(stderr, __VA_ARGS__);          \
        fprintf(stderr, "\n");                 \
        fflush(stderr);                        \
    } while (false)


// HMMM TODO i should just do 32 units = 1 metre. 
// let 1 unit = 1 inch, this approximates 32 units to 0.82 metres
#define STANDARD_LENGTH_IN_GAME_UNITS 32
#define GAME_UNIT_TO_SI_UNITS 0.0254f // 0.8128m / 32 units
#define SI_UNITS_TO_GAME_UNITS 39.37f // 1 / 0.0254
// sqrt(32^2 + 32^2) = 45.254833996 ~= 45
#define STANDARD_LENGTH_DIAGONAL 45
#define THIRTYTWO STANDARD_LENGTH_IN_GAME_UNITS
#define WORLD_LIMIT 32000
#define WORLD_LIMIT_F 32000.f

#include "mem.h"
#include "utility.h"
#include "resources.h"
#include "anim.h"
#include "game_assets.h"
#include "shaders.h"
#include "facebatch.h"
#include "filedialog.h"
#include "physics.h"
#include "physics_debug.h"
#include "primitives.h"
#include "cam.h"
#include "winged.h"
#include "lightmap.h"
#include "levelentities.h"
#include "leveleditor.h"
#include "saveloadlevel.h"
#include "gui.h"
#include "weapons.h"
#include "player.h"
#include "game.h"
#include "enemy.h"
#include "nav.h"
#include "debugmenu.h"


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
u8 KeysCurrent[256] = {0};
u8 KeysPressed[256] = {0};
u8 KeysReleased[256] = {0};

i32 BackbufferWidth = -1;
i32 BackbufferHeight = -1;

char CurrentWorkingDirectory[128];

#if INTERNAL_BUILD
RENDERDOC_API_1_6_0 *RDOCAPI = NULL;
#endif


GPUShader GameLevelShader;
GPUShader GameModelTexturedShader;
GPUShader GameModelSkinnedShader;
GPUShader GunShader;
GPUShader PatchesIDShader;
GPUShader EditorShader_Scene;
GPUShader EditorShader_Wireframe;
GPUShader EditorShader_FaceSelected;
GPUShader FinalPassShader;
GPUFrameBuffer RenderTargetGame;
GPUFrameBuffer RenderTargetGUI;
GPUMeshIndexed FinalRenderOutputQuad;
float GAMEPROJECTION_NEARCLIP = 4.f; // even 2 works fine to remove z fighting
float GAMEPROJECTION_FARCLIP = 32000.f;


#include "mem.cpp"
#include "utility.cpp"
#include "anim.cpp"
#include "resources.cpp"
#include "game_assets.cpp"
#include "physics_debug.cpp"
#include "shaders.cpp"
#include "facebatch.cpp"
#include "filedialog.cpp"
#include "primitives.cpp"
#include "lightmap.cpp"
#include "winged.cpp"
#include "leveleditor.cpp"
#include "saveloadlevel.cpp"
#include "game.cpp"
#include "physics.cpp"
#include "gui.cpp"
#include "levelentities.cpp"
#include "enemy.cpp"
#include "nav.cpp"
#include "cam.cpp"
#include "player.cpp"
#include "weapons.cpp"
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

    GUI::Draw();
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


    SDL_GL_GetDrawableSize(SDLMainWindow, &BackbufferWidth, &BackbufferHeight);
    RenderTargetGame.width = BackbufferWidth;
    RenderTargetGame.height = BackbufferHeight;
    CreateGPUFrameBuffer(&RenderTargetGame);
    RenderTargetGUI.width = BackbufferWidth / 2;
    RenderTargetGUI.height = BackbufferHeight / 2;
    CreateGPUFrameBuffer(&RenderTargetGUI);


    GLLoadShaderProgramFromFile(GameLevelShader, 
        shader_path("__game_level.vert").c_str(), 
        shader_path("__game_level.frag").c_str());
    GLLoadShaderProgramFromFile(GameModelTexturedShader, 
        shader_path("model_textured.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    GLLoadShaderProgramFromFile(GameModelSkinnedShader, 
        shader_path("model_skinned.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    GLLoadShaderProgramFromFile(GunShader, 
        shader_path("guns.vert").c_str(), 
        shader_path("guns.frag").c_str());
    GLLoadShaderProgramFromFile(PatchesIDShader, 
        shader_path("__patches_id.vert").c_str(), 
        shader_path("__patches_id.frag").c_str());
    GLCreateShaderProgram(EditorShader_Scene, 
        __editor_scene_shader_vs, 
        __editor_scene_shader_fs);
    GLCreateShaderProgram(EditorShader_Wireframe, 
        __editor_scene_wireframe_shader_vs, 
        __editor_scene_wireframe_shader_fs);
    GLCreateShaderProgram(EditorShader_FaceSelected, 
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

    SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "system"); // https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/LearnWin32/dpi-and-device-independent-pixels.md#dwm-scaling
    SDL_SetHint(SDL_HINT_WINDOWS_DPI_SCALING, "0"); // https://github.com/libsdl-org/SDL/commit/ab81a559f43abc0858c96788f8e00bbb352287e8

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) return false;

    // OpenGL 4.6
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDLMainWindow = SDL_CreateWindow("game",
                                     SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                     2560,
                                     1440,
                                     SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    SDLGLContext = SDL_GL_CreateContext(SDLMainWindow);

    if (SDLMainWindow == nullptr || SDLGLContext == nullptr) return false;

#ifdef MESA_USING_GL3W
    if (gl3w_init())
    {
        fprintf(stderr, "Failed to initialize OpenGL\n");
        return false;
    }
    LogMessage("GL_VERSION %s", glGetString(GL_VERSION));
#endif

    SDL_SetWindowMinimumSize(SDLMainWindow, 200, 100);
    SDL_GL_SetSwapInterval(0);
    // if (SDL_GL_SetSwapInterval(-1) == -1)
    // {
    //     LogWarning("Hardware does not support adaptive vsync.");
    //     SDL_GL_SetSwapInterval(1);
    // }

    if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0 || Mix_Init(MIX_INIT_OGG) < 0)
        return false;

    GUI::Init();

    return true;
}

static void ProcessSDLEvents()
{
    // MOUSE
    ivec2 prevmp = MousePos;
    u32 mousestate = SDL_GetMouseState(&MousePos.x, &MousePos.y); // mousepos only valid when mouse not relative mode
    u32 mousechanged = MouseCurrent ^ mousestate;
    MousePressed = mousechanged & mousestate;
    MouseReleased = mousechanged & MouseCurrent;
    MouseCurrent = mousestate;
    ivec2 md;
    SDL_GetRelativeMouseState(&md.x, &md.y);
    MouseDelta.x = (float)md.x;
    MouseDelta.y = (float)md.y;

    // KEYBOARD
    const u8 *keystate = SDL_GetKeyboardState(NULL);
    u8 keyschanged[256];
    for (int i=0;i<256;++i)
    {
        keyschanged[i] = KeysCurrent[i] ^ keystate[i];
        KeysPressed[i] = keyschanged[i] & keystate[i];
        KeysReleased[i] = keyschanged[i] & KeysCurrent[i];
        KeysCurrent[i] = keystate[i];
    }

    // EVENT HANDLING
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_WINDOWEVENT:
            {
                switch (event.window.event) 
                {
                    case SDL_WINDOWEVENT_SIZE_CHANGED:
                    case SDL_WINDOWEVENT_RESIZED:
                    {
                        SDL_GL_GetDrawableSize(SDLMainWindow, &BackbufferWidth, &BackbufferHeight);
                        UpdateGPUFrameBufferSize(&RenderTargetGame, BackbufferWidth, BackbufferHeight);
                        UpdateGPUFrameBufferSize(&RenderTargetGUI, BackbufferWidth / 2, BackbufferHeight / 2);
                        break;
                    }
                }
                break;
            }

            case SDL_QUIT:
            {
                ProgramShutdownRequested = true;
                break;
            }

            case SDL_KEYDOWN:
            {
                SDL_Keycode sdlkey = event.key.keysym.sym;

                if (sdlkey == SDLK_RETURN && SDL_GetModState() & KMOD_LALT)
                {
                    if (SDL_GetWindowFlags(SDLMainWindow) & SDL_WINDOW_FULLSCREEN_DESKTOP)
                        SDL_SetWindowFullscreen(SDLMainWindow, 0);
                    else
                        SDL_SetWindowFullscreen(SDLMainWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);
                    event.type = 0;
                }
                break;
            }
        }

        GUI::ProcessSDLEvent(event);
    }
}

static void ApplicationLoop()
{
    if (KeysPressed[SDL_SCANCODE_GRAVE]) 
    {
        DebugMenuActive = !DebugMenuActive;
        if (DebugMenuActive)
            SDL_SetRelativeMouseMode(SDL_FALSE);
        else
            SDL_SetRelativeMouseMode(SDL_TRUE);
    }

    DisplayDebugMenu();
}

static void ApplicationEnd()
{
    SupportRenderer.Destroy();

    SDL_DestroyWindow(SDLMainWindow);
    SDL_GL_DeleteContext(SDLGLContext);
    SDL_Quit();
}

int main(int argc, char* argv[])
{
    if (!InitializeApplication()) return -1;

    InitGameRenderer();

    StaticGameMemory.Init(256000000);
    StaticLevelMemory.Init(32000000);

    Assets.LoadAllResources();

    // RDOCAPI->LaunchReplayUI(1, "");

    srand(100);

    InitializeGame();

    // LevelEditor.LoadMap(wd_path("playground_0.emf").c_str());
    // BuildGameMap(wd_path("buildtest2.map").c_str());
    // LoadLevel(wd_path("buildtest2.map").c_str());

    LoadLevel(wd_path("testing.map").c_str());
    // LevelEditor.Open();

    while (!ProgramShutdownRequested)
    {
        TickTime();
        GUI::NewFrame();
        ProcessSDLEvents();

#if INTERNAL_BUILD
        if (RDOCAPI && KeysPressed[SDL_SCANCODE_HOME])
            if (RDOCAPI->ShowReplayUI() == 0)
                RDOCAPI->LaunchReplayUI(1, "");
#endif // INTERNAL_BUILD

        ApplicationLoop();

        if (LevelEditor.IsActive)
        {
            LevelEditor.Tick();
            LevelEditor.Draw();
        }
        else
        {
            DoGameLoop();
        }

        RenderGUILayer();
        FinalRenderToBackBuffer();

        SDL_GL_SwapWindow(SDLMainWindow);
    }

    LevelEditor.Close();
    DestroyGame();

    ApplicationEnd();

    free(StaticGameMemory.Arena);
    free(StaticLevelMemory.Arena);

    return 0;
}
