#ifndef SHOOTER_SHARED_H
#define SHOOTER_SHARED_H

#include "shooter_types.h"
#include "shooter_intrinsics.h"

#define CopyArray(Dest, Source, Count) Copy((Dest), (Source), (Count)*sizeof(*(Source)))
internal void *
Copy(void *DestInit, void *SourceInit, memory_index Size)
{
    // TODO(Kevin): make this faster - copy by 8 bytes (or is 4 faster?) at a time then the trailing bytes!
    uint8 *Source = (uint8 *)SourceInit;
    uint8 *Dest = (uint8 *)DestInit;
    while (Size--)
    {
        *Dest++ = *Source++;
    }
    
    return DestInit;
}

#define ZeroStruct(Instance) ZeroSize(sizeof(Instance), &(Instance))
#define ZeroArray(Count, Pointer) ZeroSize((Count)*sizeof((Pointer)[0]), Pointer)
internal void
ZeroSize(memory_index Size, void *Ptr)
{
    uint8 *Byte = (uint8 *)Ptr;
    while (Size--)
    {
        *Byte++ = 0;
    }
}

internal bool32
IsEndOfLine(char C)
{
    bool32 Result = ((C == '\n') ||
                  (C == '\r'));
    
    return Result;
}

internal bool32
IsSpacing(char C)
{
    bool32 Result = ((C == ' ') ||
                  (C == '\t') ||
                  (C == '\v') ||
                  (C == '\f'));
    
    return Result;
}

internal bool32
IsWhitespace(char C)
{
    bool32 Result = (IsSpacing(C) || IsEndOfLine(C));
    
    return Result;
}

internal bool32
IsAlpha(char C)
{
    bool32 Result = (((C >= 'a') && (C <= 'z')) ||
                   ((C >= 'A') && (C <= 'Z')));
    
    return Result;
}

internal bool32
IsNumber(char C)
{
    bool32 Result = ((C >= '0') && (C <= '9'));
    
    return Result;
}

internal bool32
IsHex(char Char)
{
    bool32 Result = (((Char >= '0') && (Char <= '9')) ||
                  ((Char >= 'A') && (Char <= 'F')) ||
                  ((Char >= 'a') && (Char <= 'f')));
    
    return Result;
}

internal uint32
GetHex(char Char)
{
    uint32 Result = 0;
    
    if((Char >= '0') && (Char <= '9'))
    {
        Result = Char - '0';
    }
    else if((Char >= 'A') && (Char <= 'F'))
    {
        Result = 0xA + (Char - 'A');
    }
    else if((Char >= 'a') && (Char <= 'f'))
    {
        Result = 0xA + (Char - 'a');
    }
    
    return Result;
}

internal bool32
MemoryIsEqual(umm Count, void *AInit, void *BInit)
{
    uint8 *A = (uint8 *)AInit;
    uint8 *B = (uint8 *)BInit;
    while(Count--)
    {
        if(*A++ != *B++)
        {
            return(false);
        }
    }
    
    return(true);
}

internal bool32
StringsAreEqual(char *A, char *B)
{
    bool32 Result = (A == B);
    
    if(A && B)
    {
        while(*A && *B && (*A == *B))
        {
            ++A;
            ++B;
        }
        
        Result = ((*A == 0) && (*B == 0));
    }
    
    return Result;
}

internal bool32
StringsAreEqual(memory_index ALength, char *A, memory_index BLength, char *B)
{
    bool32 Result = (ALength == BLength);
    
    if(Result)
    {
        Result = true;
        for(uint32 Index = 0;
            Index < ALength;
            ++Index)
        {
            if(A[Index] != B[Index])
            {
                Result = false;
                break;
            }
        }
    }
    
    return Result;
}

internal bool32
StringsAreEqual(umm ALength, char *A, char *B)
{
    bool32 Result = false;

    if(B)
    {
        char *At = B;
        for(umm Index = 0;
            Index < ALength;
            ++Index, ++At)
        {
            if((*At == 0) ||
               (A[Index] != *At))
            {
                return(false);
            }
        }

        Result = (*At == 0);
    }
    else
    {
        Result = (ALength == 0);
    }

    return(Result);
}

internal char
ToLowercase(char Char)
{
    char Result = Char;
    
    if((Result >= 'A') && (Result <= 'Z'))
    {
        Result += 'a' - 'A';
    }
    
    return(Result);
}

internal bool32
StringsAreEqualLowercase(memory_index ALength, char *A, memory_index BLength, char *B)
{
    bool32 Result = (ALength == BLength);
    
    if(Result)
    {
        Result = true;
        for(uint32 Index = 0;
            Index < ALength;
            ++Index)
        {
            if(ToLowercase(A[Index]) != ToLowercase(B[Index]))
            {
                Result = false;
                break;
            }
        }
    }
    
    return Result;
}

internal void
CatStrings(size_t SourceACount, char *SourceA,
           size_t SourceBCount, char *SourceB,
           size_t DestCount, char *Dest)
{
    for (size_t Index = 0; Index < SourceACount; ++Index)
    {
        *Dest++ = *SourceA++;
    }
    for (size_t Index = 0; Index < SourceBCount; ++Index)
    {
        *Dest++ = *SourceB++;
    }
    *Dest++ = 0;
}

internal int
StringLength(char *String)
{
    int Count = 0;
    while (*String++)
    {
        ++Count;
    }
    return Count;
}

internal real32
HorizontalFOVToVerticalFOV(real32 FOVXInRadians, real32 AspectRatio)
{
    real32 FOVY = 2.0f * ATan(Tan(FOVXInRadians / 2.0f) / AspectRatio);
    return FOVY;
}

internal void
PopulateFreeHandles(u32 *HandlesArray, u32 Count)
{
    s32 SCount = SafeCastToS32(Count);
    for (s32 FreeHandle = 0;
         FreeHandle < SCount;
         ++FreeHandle)
    {
        HandlesArray[SCount - FreeHandle - 1] = SafeCastToU32(FreeHandle);
    }
}

global vec3 DebugColorTable[] =
{
    /* 00 */ {1, 0, 0},
    /* 01 */ {0, 1, 0},
    /* 02 */ {0, 0, 1},
    /* 03 */ {1, 1, 0},
    /* 04 */ {0, 1, 1},
    /* 05 */ {1, 0, 1},
    /* 06 */ {1, 0.5f, 0},
    /* 07 */ {1, 0, 0.5f},
    /* 08 */ {0.5f, 1, 0},
    /* 09 */ {0, 1, 0.5f},
    /* 10 */ {0.5f, 0, 1},
    /* 11 */ {1, 0.75f, 0.5f},
    /* 12 */ {1, 0.5f, 0.75f},
    /* 13 */ {0.75f, 1, 0.5f},
    /* 14 */ {0.5f, 1, 0.75f},
    /* 15 */ {0.5f, 0.75f, 1},
    
    /* 16 */ {1, 0.25f, 0.25f},
    /* 17 */ {0.25f, 1, 0.25f},
    /* 18 */ {0.25f, 0.25f, 1},
    /* 19 */ {1, 1, 0.25f},
    /* 20 */ {0.25f, 1, 1},
    /* 21 */ {1, 0.25f, 1},
    /* 22 */ {1, 0.5f, 0.25f},
    /* 23 */ {1, 0.25f, 0.5f},
    /* 24 */ {0.5f, 1, 0.25f},
    /* 25 */ {0.25f, 1, 0.5f},
    /* 26 */ {0.5f, 0.25f, 1},
    /* 27 */ {1, 0.25f, 0.5f},
    /* 28 */ {1, 0.5f, 0.25f},
    /* 29 */ {0.25f, 1, 0.5f},
    /* 30 */ {0.5f, 1, 0.25f},
    /* 31 */ {0.5f, 0.25f, 1},
};

internal vec3
GetDebugColor3(u32 Value)
{
    vec3 Result = DebugColorTable[Value % ArrayCount(DebugColorTable)];
    return Result;
}

internal vec4
GetDebugColor4(u32 Value, real32 Alpha = 1.0f)
{
    vec4 Result = vec4(GetDebugColor3(Value), Alpha);
    return Result;
}

#endif // SHOOTER_SHARED_H
