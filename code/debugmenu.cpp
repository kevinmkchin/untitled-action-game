#include "debugmenu.h"
#include "game.h"

bool DebugMenuActive = false;
bool DebugDrawLevelColliderFlag = false;
bool DebugDrawEnemyCollidersFlag = false;
bool DebugDrawProjectileCollidersFlag = false;
bool DebugShowNumberOfPhysicsBodies = false;
bool DebugDrawNavMeshFlag = false;
bool DebugDrawEnemyPathingFlag = false;
bool FlyCamActive = false;

static void DebugMenu_SwitchToLevelEditor()
{
    // todo clean up game memory
    // todo close game

    GameLoopCanRun = false;

    LevelEditor.Open();

    DebugMenuActive = false;
}

static void DebugMenu_BuildLevelAndPlay()
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
    DebugMenuActive = false;
}

void DisplayDebugMenu()
{
    if (KeysPressed[SDL_SCANCODE_P])
    {
        GameLoopCanRun = !GameLoopCanRun;
    }

    if (DebugMenuActive)
    {
        SDL_SetRelativeMouseMode(SDL_FALSE);

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
            GUI::EditorCheckbox("Draw level collider", &DebugDrawLevelColliderFlag);
            GUI::EditorCheckbox("Draw enemy colliders", &DebugDrawEnemyCollidersFlag);
            GUI::EditorCheckbox("Draw projectile colliders", &DebugDrawProjectileCollidersFlag);
            GUI::EditorCheckbox("Show num physics bodies", &DebugShowNumberOfPhysicsBodies);
            GUI::EditorCheckbox("Show nav mesh", &DebugDrawNavMeshFlag);
            GUI::EditorCheckbox("Show enemy pathing", &DebugDrawEnemyPathingFlag);
            if (GUI::EditorLabelledButton("Open level editor"))
                DebugMenu_SwitchToLevelEditor();
            GUI::EditorSpacer(0, 10);
        }
        else
        {
            GUI::EditorText("-- Level Edit --");
            if (GUI::EditorLabelledButton("BUILD LEVEL AND PLAY"))
            {
                DebugMenu_BuildLevelAndPlay();
            }
            GUI::EditorSpacer(0, 10);
        }

        GUI::EditorLabelledButton("PLAY playground1.map");
        GUI::EditorLabelledButton("PLAY playground2.map");
        GUI::EditorLabelledButton("PLAY house.map");

        GUI::EndWindow();
    }

    if (!LevelEditor.IsActive && !GameLoopCanRun)
        GUI::PrimitiveText(RenderTargetGUI.width/2-26, RenderTargetGUI.height/2, GUI::GetFontSize()*2, GUI::LEFT, "PAUSED");
}
