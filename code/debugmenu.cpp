#include "debugmenu.h"
#include "game.h"
#include <SDL_clipboard.h>

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

// internal
enum CONSOLE_SHOWING_STATES
{
    DEBUG_CONSOLE_NOTMOVING,
    DEBUG_CONSOLE_SHOWING,
    DEBUG_CONSOLE_HIDING
};

static CONSOLE_SHOWING_STATES ConsoleShowingState = DEBUG_CONSOLE_NOTMOVING;
static constexpr float ConsoleHDefault = 400.f;
static float ConsoleY = 0.f;
static float ConsoleH = ConsoleHDefault;
static constexpr float ConsoleScrollSpd = 2400.f;
static fixed_array<char> ConsoleInputBuf;
static constexpr int ConsoleInputBufMax = 4000;
static u32 ConsoleInputBufCursor = 0;


// these are self-contained valid procedures that 
// run immediately after rendering

void SwitchToLevelEditor()
{
    // todo clean up game memory
    // todo close game

    LevelEditor.Open();

    SDL_SetRelativeMouseMode(SDL_FALSE);
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

void SetDebugMode(DEBUG_MODES Mode)
{
    CurrentDebugMode = (DEBUG_MODES)GM_max(0, GM_min(u32(Mode), 2));
}

void RegisterConsoleCommands()
{
    ConsoleInputBuf = fixed_array<char>(ConsoleInputBufMax, MemoryType::Game);
    ConsoleInputBuf.put(0);
    ConsoleInputBufCursor = 0;
}

void SendInputToConsole(const SDL_Event event)
{
    if (ConsoleInputBufCursor >= ConsoleInputBuf.lenu())
        ConsoleInputBufCursor = ConsoleInputBuf.lenu()-1;

    switch (event.type)
    {
        case SDL_KEYDOWN:
        {
            SDL_KeyboardEvent keyevent = event.key;
            SDL_Keycode keycode = keyevent.keysym.sym;

            switch(keycode)
            {
                case SDLK_RETURN:
                {
                    // console_command(console_input_buffer);
                    // memset(console_input_buffer, 0, console_input_buffer_count);
                    ConsoleInputBufCursor = 0;
                    ConsoleInputBuf.deln(0, ConsoleInputBuf.lenu()-1);
                    break;
                }
                case SDLK_BACKSPACE:
                {
                    if (keyevent.keysym.mod & (KMOD_CTRL) && ConsoleInputBufCursor > 0)
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

            if (!(keyevent.keysym.mod & (KMOD_CTRL | KMOD_ALT)))
            {
                const int ASCII_SPACE = 32;
                const int ASCII_TILDE = 126;
                keycode = ShiftASCII(keycode, keyevent.keysym.mod & KMOD_SHIFT);
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

static void ProcessConsoleCommands()
{

}

static void DisplayDebugConsole()
{
    GUI::PrimitivePanel(GUI::UIRect(0, (int)floorf(-ConsoleH+ConsoleY), 4000, (int)ConsoleH), 
        vec4(0.1f, 0.1f, 0.1f, 0.9f));
    if (ConsoleInputBuf.lenu() > 1)
    {
        GUI::PrimitiveText(GUI::GetFontSize(), int(ConsoleY) - (GUI::GetFontSize() / 2),
            GUI::GetFontSize(), GUI::Align::LEFT, ConsoleInputBuf.data);
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
            if (LastDebugMode == DEBUG_MODE_MENU)
            {
                CurrentDebugMode = DEBUG_MODE_MENU;
                LastDebugMode = DEBUG_MODE_OFF;
            }
            else if (LastDebugMode == DEBUG_MODE_CONSOLE)
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
            ConsoleShowingState = DEBUG_CONSOLE_NOTMOVING;
        }
        break;
    case DEBUG_CONSOLE_NOTMOVING:
        break;
    case DEBUG_CONSOLE_HIDING:
        if (ConsoleY > 0.f)
        {
            GameLoopCanRun = true;
            ConsoleY -= ConsoleScrollSpd * DeltaTime;
        }
        if (ConsoleY <= 0.f)
        {
            ConsoleShowingState = DEBUG_CONSOLE_NOTMOVING;
        }
        break;
    }

    switch (CurrentDebugMode)
    {
    case DEBUG_MODE_OFF:
        if (!LevelEditor.IsActive)
            SDL_SetRelativeMouseMode(SDL_TRUE);
        break;
    case DEBUG_MODE_MENU:
        SDL_SetRelativeMouseMode(SDL_FALSE);
        DisplayDebugMenu();
        break;
    case DEBUG_MODE_CONSOLE:
        ProcessConsoleCommands();
        break;
    }

    if (ConsoleY > 0.f)
        DisplayDebugConsole();

}
