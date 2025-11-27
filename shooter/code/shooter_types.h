#ifndef SHOOTER_TYPES_H
#define SHOOTER_TYPES_H

/*
    NOTE(Kevin):

    HANDMADE_INTERNAL:
        0 - Build for public release
        1 - Build for developer only

    HANDMADE_SLOW:
        0 - No slow code allowed!
        1 - Slow code welcome.
*/

//
// NOTE(Kevin): Compilers
//
#ifndef COMPILER_MSVC
#define COMPILER_MSVC 0
#endif

#ifndef COMPILER_LLVM
#define COMPILER_LLVM 0
#endif

#if !COMPILER_MSVC && !COMPILER_LLVM
#if _MSC_VER
#undef COMPILER_MSVC
#define COMPILER_MSVC 1
#else
// TODO(Kevin): More compilers
#undef COMPILER_LLVM
#define COMPILER_LLVM 1
#endif
#endif

//
// NOTE(Kevin): Types
//
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef int32 bool32;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef float real32;
typedef double real64;

typedef size_t memory_index;
typedef uintptr_t umm; // Unsigned Machine-sized integer for pointer arithmetic
typedef intptr_t smm; // Signed Machine-sized integer for pointer arithmetic

typedef int8 s8;
typedef int8 b8;
typedef int16 s16;
typedef int32 s32;
typedef int64 s64;
typedef bool32 b32;

typedef uint8 u8;
typedef uint16 u16;
typedef uint32 u32;
typedef uint64 u64;

typedef real32 f32;
typedef real64 f64;

struct buffer
{
    umm Count;
    uint8 *Data;
};

// NOTE(Kevin): These macros conflict with definitions in iostream.h if I ever use that
#define internal static             // static functions are internal to the translation unit
#define global static               // global static variable
#define local_persist static        // local static variable

#define Pi32 3.14159265359f

#include <stb_sprintf.h>

#if HANDMADE_SLOW
// TODO(Kevin): Complete assertion macro
#define Assert(Expression) if (!(Expression)) { *(int *)0=0; }
#else
#define Assert(Expression)
#endif

#define Kilobytes(Value) ((Value)*1024LL)
#define Megabytes(Value) (Kilobytes(Value)*1024LL)
#define Gigabytes(Value) (Megabytes(Value)*1024LL)
#define Terabytes(Value) (Gigabytes(Value)*1024LL)

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

#define Minimum(A, B) ((A < B) ? (A) : (B))
#define Maximum(A, B) ((A > B) ? (A) : (B))

#define RGB255TO1(R,G,B) ((float)R)/255.f, ((float)G)/255.f, ((float)B)/255.f
#define RGBHEXTO1(RgbInHex) float((RgbInHex & 0x00FF00000) >> 16)/255.f,\
                            float((RgbInHex & 0x0000FF00) >> 8)/255.f,\
                            float(RgbInHex & 0x000000FF)/255.f

inline u8
SafeTruncateU32(u32 Value)
{
    Assert(Value <= 0xFF);
    u8 Result = (u8)Value;
    return Result;    
}

inline u32
SafeTruncateU64(u64 Value)
{
    // TODO(Kevin): Defines for maximum values UInt32Max
    Assert(Value <= 0xFFFFFFFF);
    u32 Result = (u32)Value;
    return Result;
}

inline u32
SafeCastToU32(s32 Value)
{
    Assert(Value >= 0);
    u32 Result = (u32)Value;
    return Result;
}

inline s32
SafeCastToS32(u32 Value)
{
    Assert(Value < 0x80000000);
    s32 Result = (s32)Value;
    return Result;
}

#endif // SHOOTER_TYPES_H
