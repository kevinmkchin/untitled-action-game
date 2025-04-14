#pragma once

#include "pch.h"

#include <stb_sprintf.h>
#include <gmath.h>
#include <gl3w.h>
#define GL_VERSION_4_3_OR_HIGHER

#include "BUILDINFO.H"

#if INTERNAL_BUILD
    #define ASSERTDEBUG(predicate) pch_ASSERT(predicate)
#else
    #define ASSERTDEBUG(predicate)
#endif
#define ASSERT(predicate) pch_ASSERT(predicate)

#define ARRAY_COUNT(a) (sizeof(a) / (sizeof(a[0])))

std::string wd_path();
std::string wd_path(const std::string& name);
std::string shader_path();
std::string shader_path(const std::string& name);
std::string model_path();
std::string model_path(const std::string& name);
std::string texture_path();
std::string texture_path(const std::string& name);
std::string sfx_path();
std::string sfx_path(const std::string& name);
std::string entity_icons_path();
std::string entity_icons_path(const std::string& name);

#define PRINT_TO_INGAME_CONSOLE
inline u32 CharBufLen(char *Buf);
void LogMessage(const char *fmt, ...);
void LogWarning(const char *fmt, ...);
void LogError(const char *fmt, ...);

// let 1 unit = 1 inch, this approximates 32 units to 0.82 metres
#define STANDARD_LENGTH_IN_GAME_UNITS 32
#define GAME_UNIT_TO_SI_UNITS 0.0254f // 0.8128m / 32 units
#define SI_UNITS_TO_GAME_UNITS 39.37f // 1 / 0.0254
// sqrt(32^2 + 32^2) = 45.254833996 ~= 45
#define STANDARD_LENGTH_DIAGONAL 45
#define THIRTYTWO STANDARD_LENGTH_IN_GAME_UNITS
#define WORLD_LIMIT 32000
#define WORLD_LIMIT_F 32000.f

struct app_state
{
    SDL_Window *SDLMainWindow;

    float TimeSinceStart = 0.f;
    u32 MouseCurrent;
    u32 MousePressed;
    u32 MouseReleased;
    vec2 MouseDelta;
    ivec2 MousePos;
    bool KeysCurrent[256] = {0};
    bool KeysPressed[256] = {0};
    bool KeysReleased[256] = {0};
    i32 BackBufferWidth = -1;
    i32 BackBufferHeight = -1;

    i32 GUIRenderTargetWidth;
    i32 GUIRenderTargetHeight;
    struct support_renderer_t *PrimitivesRenderer;
    struct level_editor_t *LevelEditor;
};

extern vec2 MouseDelta;
extern bool KeysCurrent[256];
extern float DeltaTime;
constexpr float FixedDeltaTime = 1.0f / 60.0f;

