/*

Something fucking cool.
Priorty #1 is building the game, not the engine/tech
Handcrafted with love.

I could port the GL code to Vulkan

TODO:

- try simple irradiance caching?

= I can start making and importing proper character models and animations
= I can start making proper textures

- Enemy moves and shoots at player
    - Try using straight path instead of smooth path
    - State machine - mesh changes color when enemy state changes from patrol to chase to shoot to melee
    - sound (audio system)

- Player has a gun and shoots at enemy



- move gmath as subrepo of this repo so i can keep tests updated
- Probably scale down vectors and units before passing into Jolt Physics
- Port over dropdown console?


- list all point entities in the scene
- entity placement window with dropdown
- entity properties window
- reset data when switching between game and map editor

- show lines visualizing total translation when moving things. up and down axis as well
- srgb gamma correction bull shit for editor texture that are not lit

- save/load texture database
- Anisotropic filtering https://www.khronos.org/opengl/wiki/Sampler_Object#Anisotropic_filtering

- move volume mode/tool (don't need whole tool...maybe make brush tool part of this mode? like trenchbroom)
- multi select vertices need to be fixed
- Toggle element translation along axis
- Finish Winged brep editing atomics
- Edge select
- Edge loop


EPICS:
World - I want a crispy fucking visually nice to look at world being created and loaded in game

    DONE Refactor FaceBatch rendering and lighting code 

    Refine Hemicube GI
    - try simple irradiance caching - interpolate w/o irrad caching doesn't work well for large patch sizes.
    - larger hemicube atlas for download (meh, one by one is good for modifications for now)
    - Texels too far from face polygon should be marked as ignore for hemicube. worst case distance: magnitude(1.5*texelsize, 0.5*texelsize)

    Face creation and manipulation by default over cuboid volumes in level editor
    - volume is wasteful; source 2 defaults to faces; best workflow is extrude faces and extrude edges

Physics - I want a crispy bug-free physics simulation with the world

    DONE Hook player controller into Jolt. Move around world simulated via Jolt.

THEN, we have a fucking engine. A fucking starting point for the game.

View-space projected blood decals

Get some things to shoot on the screen (navmesh/pathfinding, skeletal animations)
- Enemy billboards two behaviour:
    - walk towards player
    - stop and shoot after split second to shoot projectile at player, projectile hurts player health
- Player holds 3D gun model, shoot hitscan at enemies, hitscan hurts enemy health
    - Swap to a second 3D gun model, shoot shotgun hitscan at enemies
    - Swap to a third 3D gun model, shoot hitscan with different stats at enemies

Particle effects


== OTHER ==

(poly draw) Draw poly on plane then raise to create volume tool

SDL3:
Can set FramesInFlight https://wiki.libsdl.org/SDL3/SDL_SetGPUAllowedFramesInFlight but 
not sure how it works. Try upgrading to SDL3 first...or to latest SDL2...ugh might still 
just have to rewrite platforms code. Rewrite platforms code in Win32 and remove SDL - FPS
mouselook feels janky (lag + stutter when low framerate), yaw and pitch calculation as 
euler might be dumb. Just write directly to a rotation matrix?

Non-convex faces:
I _could_ replace my shitty fanning triangulation with CDT library (https://github.com/artem-ogre/CDT)
with constraint that all vertices of a face must lie on the same plane and enforce that constraint in
the editor. Then I could have 

== BUGS ==
- (2024-09-24 T480) periodic frame rate drops down to ~59/60fps then back up when in
                    fullscreen mode (both SDL_WINDOW_FULLSCREEN and SDL_WINDOW_FULLSCREEN_DESKTOP)
                    I don't think it has to do with TickTime - Changing to use SDL_GetTicks64 no 
                    effect. Look into https://wiki.libsdl.org/SDL2/SDL_GetWindowDisplayMode
                    Also, setting the window size to the screen size when creating window makes 
                    SDL enter fullscreen?

== NOTES ==

    64x64 pixel texture for each 32x32 unit in game looks decent

    Embrace the fact that lighting will be crude and not perfect. The visual artifacts 
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

#include "BUILDINFO.H"

#if MESA_WINDOWS
#include <windows.h>
#include <dwmapi.h>
#include <direct.h>
#endif

#if MESA_WINDOWS
    #define GL3W_IMPLEMENTATION
    #include <gl3w.h>
    #define MESA_USING_GL3W
#elif MESA_MACOSX
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
#endif

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
inline std::string data_path() { return wd_path() + "data/"; }
inline std::string data_path(const std::string& name) { return wd_path() + "data/" + name; }

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


// let 1 unit = 1 inch, this approximates 32 units to 0.82 metres
#define STANDARD_LENGTH_IN_GAME_UNITS 32
// sqrt(32^2 + 32^2) = 45.254833996 ~= 45
#define STANDARD_LENGTH_DIAGONAL 45
#define THIRTYTWO STANDARD_LENGTH_IN_GAME_UNITS


#include "utility.h"
#include "anim.h"
#include "resources.h"
#include "shaders.h"
#include "facebatch.h"
#include "filedialog.h"
#include "physics.h"
#include "physics_debug.h"
#include "primitives.h"
#include "winged.h"
#include "lightmap.h"
#include "levelentities.h"
#include "leveleditor.h"
#include "saveloadlevel.h"
#include "gui.h"
#include "player.h"
#include "game.h"
#include "enemy.h"
#include "nav.h"



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


level_editor_t LevelEditor;


GPUShader GameLevelShader;
GPUShader GameAnimatedCharacterShader;
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



// MIXER
Mix_Chunk *Mixer_LoadChunk(const char *filepath)
{
    Mix_Chunk *chunk = Mix_LoadWAV(filepath);
    if (chunk == NULL)
        printf("Failed to load sound effect! SDL_mixer error: %s\n", Mix_GetError());
    return chunk;
}


#include "utility.cpp"
#include "anim.cpp"
#include "resources.cpp"
#include "physics.cpp"
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
#include "gui.cpp"
#include "levelentities.cpp"
#include "enemy.cpp"
#include "nav.cpp"
#include "player.cpp"


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


    GLLoadShaderProgramFromFile(GameLevelShader, shader_path("__game_level.vert").c_str(), shader_path("__game_level.frag").c_str());
    GLLoadShaderProgramFromFile(GameAnimatedCharacterShader, shader_path("game_animated_character.vert").c_str(), shader_path("game_animated_character.frag").c_str());
    GLLoadShaderProgramFromFile(PatchesIDShader, shader_path("__patches_id.vert").c_str(), shader_path("__patches_id.frag").c_str());
    GLCreateShaderProgram(EditorShader_Scene, __editor_scene_shader_vs, __editor_scene_shader_fs);
    GLCreateShaderProgram(EditorShader_Wireframe, __editor_scene_wireframe_shader_vs, __editor_scene_wireframe_shader_fs);
    GLCreateShaderProgram(EditorShader_FaceSelected, __editor_shader_face_selected_vs, __editor_shader_face_selected_fs);
    GLCreateShaderProgram(FinalPassShader, __finalpass_shader_vs, __finalpass_shader_fs);


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

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
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

static void ApplicationSwitchToLevelEditor()
{
    // todo clean up game memory
    // todo close game

    GameLoopCanRun = false;

    LevelEditor.Open();
}

static void ApplicationBuildLevelAndPlay()
{
    if (!LevelEditor.IsActive)
    {
        LogWarning("ApplicationBuildLevelAndPlay called when level editor is not active.");
        return;
    }

    std::string path = SaveGameMapDialog();
    if (path.empty())
        return;

    if (BuildGameMap(path.c_str()) == false)
    {
        LogError("Failed to build to %s", path.c_str());
        return;
    }

    LevelEditor.Close();

    // todo open game

    LoadLevel(path.c_str());

    // todo set game to be active

    GameLoopCanRun = true;
}

static void ApplicationLoop()
{
    // TODO actually I want option to let the game keep running while
    // debug menu is open
    static bool ShowDebugMenu = false;
    if (KeysPressed[SDL_SCANCODE_GRAVE]) 
    {
        ShowDebugMenu = !ShowDebugMenu;
        if (ShowDebugMenu)
        {
            GameLoopCanRun = false;
            SDL_SetRelativeMouseMode(SDL_FALSE);
        }
        else
        {
            GameLoopCanRun = true;
            SDL_SetRelativeMouseMode(SDL_TRUE);
        }
    }
    if (ShowDebugMenu)
    {
        GUI::BeginWindow(GUI::UIRect(32, 32, 200, 300));
        GUI::EditorText("== Menu ==");
        GUI::EditorSpacer(0, 10);
        if (GUI::EditorLabelledButton("OPEN LEVEL EDITOR"))
        {
            ApplicationSwitchToLevelEditor();
            ShowDebugMenu = false;
        }
        GUI::EditorSpacer(0, 10);
        if (GUI::EditorLabelledButton("BUILD LEVEL AND PLAY"))
        {
            ApplicationBuildLevelAndPlay();
            ShowDebugMenu = false;
        }
        GUI::EditorSpacer(0, 10);
        GUI::EditorLabelledButton("PLAY playground1.map");
        GUI::EditorLabelledButton("PLAY playground2.map");
        GUI::EditorLabelledButton("PLAY house.map");
        GUI::EndWindow();

        GUI::PrimitiveText(RenderTargetGUI.width/2-13, RenderTargetGUI.height/2, GUI::GetFontSize(), GUI::LEFT, "PAUSED");
    }
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

    Assets.LoadAllResources();

    // RDOCAPI->LaunchReplayUI(1, "");

    srand(100);

    InitializeGame();

    LevelEditor.LoadMap(wd_path("LightTest.emf").c_str());
    BuildGameMap(wd_path("buildtest.map").c_str());
    LoadLevel(wd_path("buildtest.map").c_str());

    // LoadLevel(wd_path("playground_0.map").c_str());
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

    return 0;
}
