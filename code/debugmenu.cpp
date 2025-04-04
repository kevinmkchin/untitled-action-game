#include "debugmenu.h"
#include "game.h"
#include <SDL3/SDL_clipboard.h>

// todo
// - paste from clipboard


// external
DEBUG_MODES CurrentDebugMode = DEBUG_MODE_OFF;
bool DebugDrawLevelColliderFlag = false;
bool DebugDrawEnemyCollidersFlag = false;
bool DebugDrawProjectileCollidersFlag = false;
#if INTERNAL_BUILD
bool DebugShowNumberOfPhysicsBodies = true;
bool DebugShowGameMemoryUsage = true;
#else
bool DebugShowNumberOfPhysicsBodies = false;
bool DebugShowGameMemoryUsage = false;
#endif
bool DebugDrawNavMeshFlag = false;
bool DebugDrawEnemyPathingFlag = false;
bool DebugEnemyBehaviourActive = true;
bool FlyCamActive = false;

noclip::console ConsoleBackend;

// internal
enum CONSOLE_SHOWING_STATES
{
    DEBUG_CONSOLE_HIDDEN,
    DEBUG_CONSOLE_SHOWING,
    DEBUG_CONSOLE_SHOWN,
    DEBUG_CONSOLE_HIDING
};

static CONSOLE_SHOWING_STATES ConsoleShowingState = DEBUG_CONSOLE_HIDDEN;
static constexpr float ConsoleHDefault = 400.f;
static float ConsoleY = 0.f;
static float ConsoleH = ConsoleHDefault;
static constexpr float ConsoleScrollSpd = 2400.f;
static fixed_array<char> ConsoleInputBuf;
static constexpr int ConsoleInputBufMax = 4000;
static u32 ConsoleInputBufCursor = 0;
static bool FlushConsoleCommands = false;
static fixed_array<char> ConsoleOutputBuf;
static constexpr int ConsoleOutputBufMax = 4000;

// these are self-contained valid procedures that 
// run immediately after rendering

void SwitchToLevelEditor()
{
    // todo clean up game memory
    // todo close game

    LevelEditor.Open();

    SDL_SetWindowRelativeMouseMode(SDLMainWindow, false);
    GameLoopCanRun = false;
    // CurrentDebugMode = DEBUG_MODE_OFF;
}

void BuildLevelAndPlay()
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

    LoadLevel(path.c_str());

    GameLoopCanRun = true;
    // CurrentDebugMode = DEBUG_MODE_OFF;
}

void SetDebugMode(int Mode)
{
    CurrentDebugMode = (DEBUG_MODES)GM_max(0, GM_min(u32(Mode), 2));
    if (CurrentDebugMode != DEBUG_MODE_CONSOLE)
        ConsoleShowingState = DEBUG_CONSOLE_HIDDEN;
}

void dd_col_level(int v)
{
    DebugDrawLevelColliderFlag = v != 0;
}

void dd_col_monsters(int v)
{
    DebugDrawEnemyCollidersFlag = v != 0;
}

void dd_col_projectiles(int v)
{
    DebugDrawProjectileCollidersFlag = v != 0;
}

void show_num_bodies(int v)
{
    DebugShowNumberOfPhysicsBodies = v != 0;
}

void show_memory(int v)
{
    DebugShowGameMemoryUsage = v != 0;
}

void dd_navmesh(int v)
{
    DebugDrawNavMeshFlag = v != 0;
}

void dd_navpath(int v)
{
    DebugDrawEnemyPathingFlag = v != 0;
}

void monsters_chase(int v)
{
    DebugEnemyBehaviourActive = v != 0;
}

void proc_noclip()
{
    FlyCamActive = !FlyCamActive;
}

void SetupConsoleAndBindProcedures()
{
    // Prepare input buffer
    ConsoleInputBuf = fixed_array<char>(ConsoleInputBufMax, MemoryType::Game);
    ConsoleInputBuf.put(0);
    ConsoleInputBufCursor = 0;

    ConsoleOutputBuf = fixed_array<char>(ConsoleOutputBufMax, MemoryType::Game);
    ConsoleOutputBuf.setlen(ConsoleOutputBufMax);

    ConsoleBackend.bind_cmd("debug", SetDebugMode);
    ConsoleBackend.bind_cmd("dd_col_level", dd_col_level);
    ConsoleBackend.bind_cmd("dd_col_monsters", dd_col_monsters);
    ConsoleBackend.bind_cmd("dd_col_projectiles", dd_col_projectiles);
    ConsoleBackend.bind_cmd("show_num_bodies", show_num_bodies);
    ConsoleBackend.bind_cmd("show_memory", show_memory);
    ConsoleBackend.bind_cmd("dd_navmesh", dd_navmesh);
    ConsoleBackend.bind_cmd("dd_navpath", dd_navpath);
    ConsoleBackend.bind_cmd("monsters_chase", monsters_chase);
    ConsoleBackend.bind_cmd("noclip", proc_noclip);
}

void SendInputToConsole(const SDL_Event event)
{
    if (ConsoleInputBufCursor >= ConsoleInputBuf.lenu())
        ConsoleInputBufCursor = ConsoleInputBuf.lenu()-1;

    switch (event.type)
    {
        case SDL_EVENT_KEY_DOWN:
        {
            SDL_KeyboardEvent keyevent = event.key;
            SDL_Keycode keycode = keyevent.key;

            switch(keycode)
            {
                case SDLK_RETURN:
                {
                    FlushConsoleCommands = true;
                    if(ConsoleInputBuf.lenu() < ConsoleInputBuf.cap())
                    {
                        ConsoleInputBuf.ins(ConsoleInputBufCursor, '\n');
                        ++ConsoleInputBufCursor;
                    }
                    break;
                }
                case SDLK_BACKSPACE:
                {
                    if (keyevent.mod & (SDL_KMOD_CTRL) && ConsoleInputBufCursor > 0)
                    {
                        do {
                            --ConsoleInputBufCursor;
                            ConsoleInputBuf.del(ConsoleInputBufCursor);
                        } while (ConsoleInputBufCursor > 0 && ConsoleInputBuf[ConsoleInputBufCursor-1] != 32);
                    }
                    if (ConsoleInputBufCursor > 0)
                    {
                        --ConsoleInputBufCursor;
                        ConsoleInputBuf.del(ConsoleInputBufCursor);
                    }
                    break;
                }
                case SDLK_PAGEUP:
                {
                    // for(int i=0;i<10;++i)
                    // {
                    //     console_scroll_up();
                    // }
                    break;
                }
                case SDLK_PAGEDOWN:
                {
                    // for(int i=0;i<10;++i)
                    // {
                    //     console_scroll_down();
                    // }
                    break;
                }
                case SDLK_LEFT:
                {
                    break;
                }
                case SDLK_RIGHT:
                {
                    break;
                }
                case SDLK_UP:
                {
                    break;
                }
                case SDLK_DOWN:
                {
                    break;
                }
            }

            if (!(keyevent.mod & (SDL_KMOD_CTRL | SDL_KMOD_ALT)))
            {
                const int ASCII_SPACE = 32;
                const int ASCII_TILDE = 126;
                keycode = ShiftASCII(keycode, KeysCurrent[SDL_SCANCODE_LSHIFT] | KeysCurrent[SDL_SCANCODE_RSHIFT]);
                if((ASCII_SPACE <= keycode && keycode < ASCII_TILDE) && keycode != '`')
                {
                    if(ConsoleInputBuf.lenu() < ConsoleInputBuf.cap())
                    {
                        ConsoleInputBuf.ins(ConsoleInputBufCursor, keycode);
                        ++ConsoleInputBufCursor;
                    }
                }
            }

            break;
        }
    }
}

static void DisplayDebugConsole()
{
    GUI::PrimitivePanel(GUI::UIRect(0, (int)floorf(-ConsoleH+ConsoleY), 4000, (int)ConsoleH), 
        vec4(0.1f, 0.1f, 0.1f, 0.9f));
    if (ConsoleInputBuf.lenu() > 1)
    {
        GUI::PrimitiveText(GUI::GetFontSize(), int(ConsoleY) - (GUI::GetFontSize() / 2),
            GUI::GetFontSize(), GUI::Align::LEFT, false, ConsoleInputBuf.data);
    }
    if (ConsoleOutputBuf.lenu() > 1)
    {
        GUI::PrimitiveText(GUI::GetFontSize(), int(ConsoleY) - 5*(GUI::GetFontSize() / 2),
            GUI::GetFontSize(), GUI::Align::LEFT, true, ConsoleOutputBuf.data);
    }

    if (u32(TimeSinceStart / 0.85f) % 2 == 1)
    {
        GUI::UIRect CursorRect = GUI::UIRect(GUI::GetFontSize() + 1 + ConsoleInputBufCursor * 6,
            int(ConsoleY) - (GUI::GetFontSize() / 2) - GUI::GetFontSize() - 4, 2, GUI::GetFontSize() + 3);
        GUI::PrimitivePanel(CursorRect, vec4(1.f, 1.f, 1.f, 1.f));
    }

}

static void DisplayDebugMenu()
{
    GUI::BeginWindow(GUI::UIRect(32, 32, 200, 300));
    GUI::EditorText("== Menu ==");
    GUI::EditorSpacer(0, 10);

    if (!LevelEditor.IsActive)
    {
        GUI::EditorText("-- Game --");
        bool DebugPausedFlag = !GameLoopCanRun;
        GUI::EditorCheckbox("Paused", &DebugPausedFlag);
        GameLoopCanRun = !DebugPausedFlag;
        GUI::EditorCheckbox("Noclip", &FlyCamActive);
        GUI::EditorCheckbox("Show memory usage", &DebugShowGameMemoryUsage);
        GUI::EditorCheckbox("Draw level collider", &DebugDrawLevelColliderFlag);
        GUI::EditorCheckbox("Draw enemy colliders", &DebugDrawEnemyCollidersFlag);
        GUI::EditorCheckbox("Draw projectile colliders", &DebugDrawProjectileCollidersFlag);
        GUI::EditorCheckbox("Show num physics bodies", &DebugShowNumberOfPhysicsBodies);
        GUI::EditorCheckbox("Show nav mesh", &DebugDrawNavMeshFlag);
        GUI::EditorCheckbox("Show enemy pathing", &DebugDrawEnemyPathingFlag);
        GUI::EditorCheckbox("Enemy behaviour active", &DebugEnemyBehaviourActive);
        if (GUI::EditorLabelledButton("Print JoltPhysicsMemory usage"))
            JoltPhysicsMemory.DebugPrint();
        if (GUI::EditorLabelledButton("Open level editor"))
            SwitchToLevelEditor();
        GUI::EditorSpacer(0, 10);
    }
    else
    {
        GUI::EditorText("-- Level Edit --");
        if (GUI::EditorLabelledButton("BUILD LEVEL AND PLAY"))
        {
            BuildLevelAndPlay();
        }
        GUI::EditorSpacer(0, 10);
    }

    GUI::EditorLabelledButton("PLAY playground1.map");
    GUI::EditorLabelledButton("PLAY playground2.map");
    GUI::EditorLabelledButton("PLAY house.map");

    GUI::EndWindow();
}

void ShowDebugConsole()
{
    if (KeysCurrent[SDL_SCANCODE_LCTRL] && KeysPressed[SDL_SCANCODE_P])
    {
        GameLoopCanRun = !GameLoopCanRun;
    }

    static DEBUG_MODES LastDebugMode = DEBUG_MODE_CONSOLE;
    if (KeysPressed[SDL_SCANCODE_GRAVE])
    {
        switch (CurrentDebugMode)
        {
        case DEBUG_MODE_OFF:
            if (KeysCurrent[SDL_SCANCODE_LCTRL])
                LastDebugMode = DEBUG_MODE_CONSOLE;
            if (LastDebugMode == DEBUG_MODE_MENU)
            {
                CurrentDebugMode = DEBUG_MODE_MENU;
                LastDebugMode = DEBUG_MODE_OFF;
            }
            else if (LastDebugMode == DEBUG_MODE_CONSOLE || LastDebugMode == DEBUG_MODE_OFF)
            {
                GameLoopCanRun = false;
                ConsoleShowingState = DEBUG_CONSOLE_SHOWING;
                LastDebugMode = DEBUG_MODE_OFF;
                CurrentDebugMode = DEBUG_MODE_CONSOLE;
            }
            break;
        case DEBUG_MODE_MENU:
            LastDebugMode = DEBUG_MODE_MENU;
            CurrentDebugMode = DEBUG_MODE_OFF;
            break;
        case DEBUG_MODE_CONSOLE:
            ConsoleShowingState = DEBUG_CONSOLE_HIDING;
            LastDebugMode = DEBUG_MODE_CONSOLE;
            CurrentDebugMode = DEBUG_MODE_OFF;
            break;
        }
    }

    ConsoleH = fminf(ConsoleHDefault, (float)RenderTargetGUI.height-60);
    switch (ConsoleShowingState)
    {
    case DEBUG_CONSOLE_HIDDEN:
        ConsoleY = 0.f;
        break;
    case DEBUG_CONSOLE_SHOWING:
        if (ConsoleY < 0.f)
        {
            ConsoleY = 0.f;
        }
        if (ConsoleY < ConsoleH)
        {
            ConsoleY += ConsoleScrollSpd * DeltaTime;
        }
        if (ConsoleY >= ConsoleH)
        {
            ConsoleY = ConsoleH;
            ConsoleShowingState = DEBUG_CONSOLE_SHOWN;
        }
        break;
    case DEBUG_CONSOLE_SHOWN:
        break;
    case DEBUG_CONSOLE_HIDING:
        if (ConsoleY > 0.f)
        {
            GameLoopCanRun = true;
            ConsoleY -= ConsoleScrollSpd * DeltaTime;
        }
        if (ConsoleY <= 0.f)
        {
            ConsoleShowingState = DEBUG_CONSOLE_HIDDEN;
        }
        break;
    }

    switch (CurrentDebugMode)
    {
    case DEBUG_MODE_OFF:
        if (!LevelEditor.IsActive)
            SDL_SetWindowRelativeMouseMode(SDLMainWindow, true);
        break;
    case DEBUG_MODE_MENU:
        SDL_SetWindowRelativeMouseMode(SDLMainWindow, false);
        DisplayDebugMenu();
        break;
    case DEBUG_MODE_CONSOLE:
        if (FlushConsoleCommands)
        {
            FlushConsoleCommands = false;
            ConsoleBackend.execute(ConsoleInputBuf.data, std::cout);
            ConsoleOutputBuf.insn(0, ConsoleInputBuf.lenu()-1);
            memcpy(ConsoleOutputBuf.data, ConsoleInputBuf.data, ConsoleInputBuf.lenu()-1);
            memset(ConsoleInputBuf.data, 0, ConsoleInputBuf.lenu()-1);
            ConsoleInputBuf.setlen(1);
            ConsoleInputBufCursor = 0;
        }
        break;
    }

    if (ConsoleY > 0.f)
        DisplayDebugConsole();

}
