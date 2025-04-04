#pragma once

// debug console/menu api
enum DEBUG_MODES : u32
{
    DEBUG_MODE_OFF,
    DEBUG_MODE_MENU,
    DEBUG_MODE_CONSOLE,
};

// procedures
void SwitchToLevelEditor();
void BuildLevelAndPlay();

// interface
void SetupConsoleAndBindProcedures();
void SendInputToConsole(const SDL_Event event);
void ShowDebugConsole();

extern DEBUG_MODES CurrentDebugMode;
extern bool DebugDrawLevelColliderFlag;
extern bool DebugDrawEnemyCollidersFlag;
extern bool DebugDrawProjectileCollidersFlag;
extern bool DebugShowNumberOfPhysicsBodies;
extern bool DebugShowGameMemoryUsage;
extern bool DebugDrawNavMeshFlag;
extern bool DebugDrawEnemyPathingFlag;
extern bool DebugEnemyBehaviourActive;
extern bool FlyCamActive;
