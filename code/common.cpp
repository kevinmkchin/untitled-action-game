#include "common.h"
#include "debugmenu.h"

#define GL3W_IMPLEMENTATION
#include <gl3w.h>
#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_SPRINTF_IMPLEMENTATION
#include <stb_sprintf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#define VERTEXT_IMPLEMENTATION
#include <vertext.h>

std::string wd_path() { return std::string(PROJECT_WORKING_DIR); }
std::string wd_path(const std::string& name) { return wd_path() + std::string(name); }
std::string shader_path() { return wd_path() + "shaders/"; }
std::string shader_path(const std::string& name) { return wd_path() + "shaders/" + name; }
std::string model_path() { return wd_path() + "models/"; }
std::string model_path(const std::string& name) { return wd_path() + "models/" + name; }
std::string texture_path() { return wd_path() + "textures/"; }
std::string texture_path(const std::string& name) { return wd_path() + "textures/" + name; }
std::string sfx_path() { return wd_path() + "sfx/"; }
std::string sfx_path(const std::string& name) { return wd_path() + "sfx/" + name; }
std::string entity_icons_path() { return wd_path() + "entity_icons/"; }
std::string entity_icons_path(const std::string& name) { return wd_path() + "entity_icons/" + name; }

u32 CharBufLen(char *Buf)
{
    u32 Len = 0;
    while(*Buf++ != '\0')
        ++Len;
    return Len;
}

void LogMessage(const char *fmt, ...)
{
    va_list ArgPtr;
    static char Message[1024];
    va_start(ArgPtr, fmt);
    stbsp_vsnprintf(Message, 1024, fmt, ArgPtr);
    va_end(ArgPtr);
#ifdef PRINT_TO_INGAME_CONSOLE
    AppendToConsoleOutputBuf(Message, CharBufLen(Message), true);
#endif // PRINT_TO_INGAME_CONSOLE
    fprintf(stdout, Message);
    fprintf(stdout, "\n");
    fflush(stdout);
}

void LogWarning(const char *fmt, ...)
{
    va_list ArgPtr;
    static char Message[1024];
    va_start(ArgPtr, fmt);
    stbsp_vsnprintf(Message, 1024, fmt, ArgPtr);
    va_end(ArgPtr);
#ifdef PRINT_TO_INGAME_CONSOLE
    AppendToConsoleOutputBuf(Message, CharBufLen(Message), true);
#endif // PRINT_TO_INGAME_CONSOLE
    fprintf(stderr, Message);
    fprintf(stderr, "\n");
    fflush(stderr);
}

void LogError(const char *fmt, ...)
{
    va_list ArgPtr;
    static char Message[1024];
    va_start(ArgPtr, fmt);
    stbsp_vsnprintf(Message, 1024, fmt, ArgPtr);
    va_end(ArgPtr);
#ifdef PRINT_TO_INGAME_CONSOLE
    AppendToConsoleOutputBuf(Message, CharBufLen(Message), true);
#endif // PRINT_TO_INGAME_CONSOLE
    fprintf(stderr, Message);
    fprintf(stderr, "\n");
    fflush(stderr);
}
