#ifndef SHOOTER_INTRINSICS_H
#define SHOOTER_INTRINSICS_H

//
// TODO(Kevin): Convert all of these to platform-efficient versions
// and remove math.h
//

#include <math.h>
#if COMPILER_MSVC
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
#endif

inline int32
SignOf(int32 Value)
{
    int32 Result = (Value >= 0) ? 1 : -1;
    return Result;
}

inline real32
SquareRoot(real32 Real32)
{
    real32 Result = sqrtf(Real32);
    return Result;
}

inline real32
AbsoluteValue(real32 Real32)
{
    real32 Result = fabsf(Real32);
    return Result;
}

inline uint32
RotateLeft(uint32 Value, int32 Amount)
{
#if COMPILER_MSVC
    uint32 Result = _rotl(Value, Amount);
#else
    // TODO(Kevin): Actually port to other compiler platforms
    Amount &= 31;
    uint32 Result = ((Value << Amount) | (Value >> (32 - Amount)));
#endif
    return Result;
}

inline uint32
RotateRight(uint32 Value, int32 Amount)
{
#if COMPILER_MSVC
    uint32 Result = _rotr(Value, Amount);
#else
    // TODO(Kevin): Actually port to other compiler platforms
    Amount &= 31;
    uint32 Result = ((Value >> Amount) | (Value << (32 - Amount)));
#endif
    return Result;
}

inline real32
RoundReal32(real32 Real32)
{
    real32 Result = roundf(Real32);
    return Result;
}

inline int32
RoundReal32ToInt32(real32 Real32)
{
    int32 Result = (int32)RoundReal32(Real32);
    return Result;
}

inline int32
RoundReal32ToUInt32(real32 Real32)
{
    uint32 Result = (uint32)roundf(Real32);
    return Result;
}

inline int32
FloorReal32ToS32(real32 Real32)
{
    int32 Result = (int32)floorf(Real32);
    return Result;
}

inline uint32
FloorReal32ToU32(real32 Real32)
{
    uint32 Result = (uint32)floorf(Real32);
    return Result;
}

inline int32 
CeilReal32ToS32(real32 Real32)
{
    int32 Result = (int32)ceilf(Real32);
    return Result;
}

inline uint32
CeilReal32ToU32(real32 Real32)
{
    uint32 Result = (uint32)ceilf(Real32);
    return Result;
}

inline int32
TruncateReal32ToInt32(real32 Real32)
{
    int32 Result = (int32)Real32;
    return Result;
}

inline real32
Sin(real32 Angle)
{
    real32 Result = sinf(Angle);
    return Result;
}

inline real32
ASin(real32 X)
{
    real32 Theta = asinf(X);
    return Theta;
}

inline real32
Cos(real32 Angle)
{
    real32 Result = cosf(Angle);
    return Result;
}

inline real32
ACos(real32 X)
{
    real32 Theta = acosf(X);
    return Theta;
}

inline real32
Tan(real32 Angle)
{
    real32 Result = tanf(Angle);
    return Result;
}

inline real32
ATan(real32 X)
{
    real32 Theta = atanf(X);
    return Theta;
}

// Gives the angle between the +X axis and the point (X, Y)
inline real32
ATan2(real32 Y, real32 X)
{
    real32 Result = atan2f(Y, X);
    return Result;
}

struct bit_scan_result
{
    bool32 Found;
    uint32 Index;
};

inline bit_scan_result
FindLeastSignificantSetBit(uint32 Value)
{
    bit_scan_result Result = {};

#if COMPILER_MSVC
    Result.Found = _BitScanForward((unsigned long *)&Result.Index, Value);
#else
    for (uint32 Test = 0; Test < 32; ++Test)
    {
        if (Value & (1 << Test))
        {
            Result.Index = Test;
            Result.Found = true;
            break;
        }
    }
#endif

    return Result;
}

inline bit_scan_result
FindMostSignificantSetBit(uint32 Value)
{
    bit_scan_result Result = {};

#if COMPILER_MSVC
    Result.Found = _BitScanReverse((unsigned long *)&Result.Index, Value);
#else
    for (uint32 Test = 0; Test < 32; ++Test)
    {
        if (Value & (1 << (31 - Test)))
        {
            Result.Index = (31 - Test);
            Result.Found = true;
            break;
        }
    }
#endif

    return Result;
}

inline u32
CeilReal32ToPow2(real32 Value)
{
    u32 V = CeilReal32ToU32(Value);
    bit_scan_result Scan = FindMostSignificantSetBit(V);
    if (Scan.Found)
    {
        if ((u32)(1 << Scan.Index) == V)
        {
            return V;
        }

        u32 Result = (1 << (Scan.Index + 1));
        return Result;
    }
    else
    {
        return 1;
    }
}

#endif // SHOOTER_INTRINSICS_H
