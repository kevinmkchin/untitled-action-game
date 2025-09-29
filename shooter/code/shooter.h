#pragma once

#include "shooter_types.h"

// the game is the services to the operating system level to produce the graphics
// and sound necessary to play the game

// write the os code be tangled and weird and do all the os specific things it needs to do
// and all the game needs to do is respond to a few very specific requests e.g. give me the
// things you want to draw, give me the sound you want to play, and heres the user's input

// basically, isolate a few specific locations in the code where the platform layer wants services
// from the game, and isolate a few specific locations in the game where the game wants services
// from the platform layer.

struct thread_context
{
    int Placeholder;
};

/*
    NOTE(Kevin): Services that the game provides to the platform layer.
*/

struct game_sound_output_buffer
{
    // 2 channel 16-bit int PCM 
    int SamplesPerSecond;
    int SampleCount;
    void *Samples;
};

struct game_button_state
{
    int HalfTransitionCount;
    bool32 EndedDown;
};

struct game_controller_input
{
    bool32 IsConnected;
    bool32 IsAnalog;
    real32 StickAverageX;
    real32 StickAverageY;

    union
    {
        game_button_state Buttons[12];
        struct
        {
            game_button_state MoveUp;
            game_button_state MoveDown;
            game_button_state MoveLeft;
            game_button_state MoveRight;

            game_button_state ActionUp; // double tap stick in direction
            game_button_state ActionDown;
            game_button_state ActionLeft;
            game_button_state ActionRight;

            game_button_state LeftShoulder;
            game_button_state RightShoulder;

            game_button_state Start;
            game_button_state Back;

            // NOTE(Kevin): All buttons must be added above this line
            game_button_state Terminator; 
        };
    };
};

struct game_input
{
    game_button_state MouseButtons[5];
    int32 MouseX, MouseY, MouseZ;

    real32 dtForFrame;

    game_controller_input Controllers[5];
};
inline game_controller_input *GetController(game_input *Input, int unsigned ControllerIndex)
{
    Assert(ControllerIndex < ArrayCount(Input->Controllers));
    game_controller_input *Result = &Input->Controllers[ControllerIndex];
    return Result;
}

#if HANDMADE_INTERNAL
/*  IMPORTANT(Kevin):
    These are NOT for doing anything in the shipping game - they are blocking
    and the write does not protect against lost data.
*/
struct debug_read_file_result
{
    uint32 ContentsSize;
    void *Contents;
};

#define DEBUG_PLATFORM_FREE_FILE_MEMORY(name) void name(thread_context *Thread, void *BitmapMemory)
typedef DEBUG_PLATFORM_FREE_FILE_MEMORY(debug_platform_free_file_memory);

#define DEBUG_PLATFORM_READ_ENTIRE_FILE(name) debug_read_file_result name(thread_context *Thread, char *Filename)
typedef DEBUG_PLATFORM_READ_ENTIRE_FILE(debug_platform_read_entire_file);

#define DEBUG_PLATFORM_WRITE_ENTIRE_FILE(name) bool32 name(thread_context *Thread, char *Filename, uint32 MemorySize, void *Memory)
typedef DEBUG_PLATFORM_WRITE_ENTIRE_FILE(debug_platform_write_entire_file);

#endif

struct platform_api
{
#if HANDMADE_INTERNAL
    debug_platform_free_file_memory *DEBUGPlatformFreeFileMemory;
    debug_platform_read_entire_file *DEBUGPlatformReadEntireFile;
    debug_platform_write_entire_file *DEBUGPlatformWriteEntireFile;
#endif
};

struct game_memory
{
    bool32 IsInitialized;

    uint64 PermanentStorageSize;
    void *PermanentStorage; // NOTE(Kevin): REQUIRED to be cleared to zero at startup

    uint64 TransientStorageSize;
    void *TransientStorage; // NOTE(Kevin): REQUIRED to be cleared to zero at startup

    platform_api PlatformAPI;
};

// PLATFORM PROVIDES FOUR THINGS - timing, controller/keyboard input, sound buffer to use, renderer to use
#define GAME_UPDATE_AND_RENDER(name) void name(thread_context *Thread, game_memory *Memory, game_input *Input, struct renderer *RendererAPI)
typedef GAME_UPDATE_AND_RENDER(game_update_and_render); 

#define GAME_GET_SOUND_SAMPLES(name) void name(thread_context *Thread, game_memory *Memory, game_sound_output_buffer *SoundBuffer)
typedef GAME_GET_SOUND_SAMPLES(game_get_sound_samples); 

//
// platform layer does not need to know about these:
//

#define Minimum(A, B) ((A < B) ? (A) : (B))
#define Maximum(A, B) ((A > B) ? (A) : (B))

struct memory_arena
{
    memory_index Size;
    uint8 *Base;
    memory_index Used;
};

internal void
InitializeArena(memory_arena *Arena, memory_index Size, uint8 *Base)
{
    Arena->Size = Size;
    Arena->Base = Base;
    Arena->Used = 0;
}

#define PushStruct(Arena, type) (type *)PushSize_(Arena, sizeof(type))
#define PushArray(Arena, Count, type) (type *)PushSize_(Arena, (Count)*sizeof(type))
void *
PushSize_(memory_arena *Arena, memory_index Size)
{
    Assert((Arena->Used + Size) <= Arena->Size);
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += Size;

    return Result;
}

#include "gmath.h"
#include "shooter_renderer.h"
#include "shooter_intrinsics.h"

struct loaded_bitmap
{
    int32 Width;
    int32 Height;
    uint32 *Pixels;
};

struct game_state
{
    memory_arena WorldArena;
    memory_arena ScratchArena;

    real32 tSine;
    loaded_bitmap Backdrop;
};
