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

struct buffer
{
    umm Count;
    uint8 *Data;
};
typedef buffer string;

#define internal static             // static functions are internal to the translation unit
#define global static               // global static variable
#define local_persist static        // local static variable

#define Pi32 3.14159265359f

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

inline uint32
SafeTruncateUInt64(uint64 Value)
{
    // TODO(Kevin): Defines for maximum values UInt32Max
    Assert(Value <= 0xFFFFFFFF);
    uint32 Result = (uint32)Value;
    return Result;
}

#endif // SHOOTER_TYPES_H
