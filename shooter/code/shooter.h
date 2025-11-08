#pragma once

#include "shooter_types.h"
#include "shooter_colliders.h"

#define SHOOTER_COMPILE_WITH_EDITOR 1

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
    s32 Placeholder;
};

/*
    NOTE(Kevin): Services that the game provides to the platform layer.
*/

struct game_sound_output_buffer
{
    // 2 channel 16-bit integer PCM 
    s32 SamplesPerSecond;
    s32 SampleCount;
    void *Samples;
};

struct game_button_state
{
    s32 HalfTransitionCount;
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
    int32 MouseX, MouseY;
    vec2 RawMouseDelta;
    vec2 ScrollDelta; // # chars, # lines
    // TODO(Kevin): Key presses can probably come in a big array here? for GUI and things

    real32 dtForFrame;

    game_controller_input Controllers[5];
};
inline game_controller_input *GetController(game_input *Input, u32 ControllerIndex)
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

#define PLATFORM_CAPTURE_MOUSECURSOR(name) void name(bool32 Capture)
typedef PLATFORM_CAPTURE_MOUSECURSOR(platform_capture_mousecursor);

struct platform_api
{
#if HANDMADE_INTERNAL
    debug_platform_free_file_memory *DEBUGPlatformFreeFileMemory;
    debug_platform_read_entire_file *DEBUGPlatformReadEntireFile;
    debug_platform_write_entire_file *DEBUGPlatformWriteEntireFile;
#endif
    platform_capture_mousecursor *CaptureMouseCursor;
};

struct game_memory
{
    bool32 IsInitialized;

    uint64 PermanentStorageSize;
    void *PermanentStorage; // NOTE(Kevin): REQUIRED to be cleared to zero at startup

    uint64 TransientStorageSize;
    void *TransientStorage; // NOTE(Kevin): REQUIRED to be cleared to zero at startup

    uint64 EditorMemoryStorageSize;
    void *EditorMemoryStorage; // NOTE(Kevin): REQUIRED to be cleared to zero at startup

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
#define STANDARD_LENGTH_IN_GAME_UNITS 32

#include "gmath.h"
#include "shooter_memory.h"
#include "shooter_renderer.h"
#include "shooter_intrinsics.h"
#include "shooter_meshes.h"
#if SHOOTER_COMPILE_WITH_EDITOR
#include "shooter_editor.h"
#endif

enum entity_type
{
    Entity_Unspecified,
    Entity_Fiend,
    Entity_Shooter,
    Entity_Fireball,
};

enum enemy_state
{
    EnemyState_Idle,
    EnemyState_Patrol,
    EnemyState_Chasing,
    EnemyState_Lunging,
    EnemyState_Shooting,
};

typedef u32 entity_handle;
struct entity
{
    entity_type Type;
    bool32 Alive;
    bool32 Visible;

    vec3 P;
    vec2 dHorizontalP; // u/s
    real32 dVerticalP; // u/s
    bool32 IsSupported; // by a floor
    quat Orientation;

    // TODO(Kevin): Should I just store a primitive collider "type" like capsule, cube, etc?
    //              How would custom point clouds work then?
    vec3 *ColliderPoints;
    u32 ColliderPointsCount;
    vec3 ColliderScale;

    vec3           RenderScale;
    mesh_handle    RenderMesh;
    texture_handle RenderAlbedo;
    texture_handle RenderLightmap;
    bool32         QueryLightCache;

    real32 AliveSeconds;

    enemy_state State;
    real32 tShoot;

    real32 HealthPoints;

    vec3 PlayerWalkDirectionRight;
    vec3 PlayerWalkDirectionForward;
};

struct loaded_bitmap
{
    int32 Width;
    int32 Height;
    uint32 *Pixels;
};

struct camera
{
    vec3 P;
    vec3 EulerRot; // Roll, Yaw, Pitch
    vec3 Forward;
    vec3 Right;
    vec3 Up;
    quat Orientation;

    // Used for ViewMatrix calculation i.e. rendering
    // This lets me shake or tilt the camera for rendering only
    vec3 ViewPosition;
    vec3 ViewForward;
    vec3 ViewUp;

    real32 StrafeTilt;
};

struct game_map
{
    vec3 PlayerStartPosition;
    vec3 PlayerStartRotation; // maybe should be "PlayerStartFacing"

    int32 LightmapW, LightmapH;
    real32 *LightmapData;

    vec3 ColliderPoints[5000];
    u32 ColliderPointsCount;
    flatpoly_collider Colliders[500];
    u32 CollidersCount;
};
// Serialize(game_map *GameMap, , bool32 IsReading);

struct game_state
{
    memory_arena PermanentArena;
    memory_arena FrameArena;

    game_map Map;

    // NOTE(Kevin): If I wanted I could add a layer of indirection where referenced entity index
    // is used to look up the real entity index -> that would let me coalesce the sparse entity
    // storage for less traversal if I want to support very many entities.
    entity Entities[1000];
    u32 EntitiesCount;
    entity_handle FreeEntityHandles[1000];
    u32 FreeEntityHandlesCount;

    camera Camera;

    entity_handle PlayerEntityHandle;

    PNUV2_vertex Cylinder8SidedVB[CylinderMesh8SidedVertexCount];
    u32 Cylinder8SidedIB[CylinderMesh8SidedIndexCount];

    vec3 Cylinder8SidedPointCloud[CylinderMesh8SidedPointCount];
    vec3 CubePointCloud[8];

    mesh_handle CubeMesh;
    mesh_handle CylinderMesh;
    texture_handle TexWireframeRed;

    real32 PlayerMaxUnitsTravelledPerSecond;
    real32 PlayerUnitsTravelledPerSecond;
    bool32 PlayerViewBobGoingDown;
    real32 PlayerViewBobOffset;
    real32 dPlayerViewBobOffset;

    real32 tSine;
    loaded_bitmap Backdrop;

    bool32 EditorMode;
#if SHOOTER_COMPILE_WITH_EDITOR
    editor Editor;
#endif
};
