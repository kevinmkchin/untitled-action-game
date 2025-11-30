#ifndef SHOOTER_MEMORY_H
#define SHOOTER_MEMORY_H

//
// Linear arena
//

#define PushStruct(Arena, type, ...) (type *)PushSize_(Arena, sizeof(type), ## __VA_ARGS__)
#define PushArray(Arena, Count, type, ...) (type *)PushSize_(Arena, (Count)*sizeof(type), ## __VA_ARGS__)
#define PushSize(Arena, Size, ...) PushSize_(Arena, Size, ## __VA_ARGS__)

struct memory_arena
{
    memory_index Size;
    uint8 *Base;
    memory_index Used;

    int32 TempCount;
};

struct temporary_memory
{
    memory_arena *Arena;
    uint8 *Base;
    memory_index Used;
};

inline void
InitializeArena(memory_arena *Arena, memory_index Size, void *Base)
{
    Arena->Size = Size;
    Arena->Base = (uint8 *)Base;
    Arena->Used = 0;
    Arena->TempCount = 0;
}

inline void
ResetArena(memory_arena *Arena)
{
    Arena->Used = 0;
}

inline uint8 *
CurrentPtr(memory_arena *Arena)
{
    return Arena->Base + Arena->Used;
}

inline temporary_memory
BeginTemporaryMemory(memory_arena *Arena)
{
    temporary_memory Result;

    Result.Arena = Arena;
    Result.Base = CurrentPtr(Arena);
    Result.Used = Arena->Used;

    ++Arena->TempCount;

    return(Result);
}

inline void
EndTemporaryMemory(temporary_memory TempMem)
{
    memory_arena *Arena = TempMem.Arena;
    Assert(Arena->Used >= TempMem.Used);
    Arena->Used = TempMem.Used;
    Assert(Arena->TempCount > 0);
    --Arena->TempCount;
}

inline memory_index
GetAlignmentOffset(memory_arena *Arena, memory_index Alignment)
{
    memory_index AlignmentOffset = 0;
    
    memory_index ResultPointer = (memory_index)Arena->Base + Arena->Used;
    memory_index AlignmentMask = Alignment - 1;
    if(ResultPointer & AlignmentMask)
    {
        AlignmentOffset = Alignment - (ResultPointer & AlignmentMask);
    }

    return(AlignmentOffset);
}

inline memory_index
GetArenaSizeRemaining(memory_arena *Arena, memory_index Alignment = 4)
{
    memory_index Result = Arena->Size - (Arena->Used + GetAlignmentOffset(Arena, Alignment));

    return(Result);
}

inline void *
PushSize_(memory_arena *Arena, memory_index SizeInit, memory_index Alignment = 4)
{
    memory_index Size = SizeInit;
        
    memory_index AlignmentOffset = GetAlignmentOffset(Arena, Alignment);
    Size += AlignmentOffset;
    
    Assert((Arena->Used + Size) <= Arena->Size);
    void *Result = Arena->Base + Arena->Used + AlignmentOffset;
    Arena->Used += Size;

    Assert(Size >= SizeInit);
    
    return(Result);
}

inline char *
PushString(memory_arena *Arena, char *Source)
{
    u32 Size = 1;
    for(char *At = Source;
        *At;
        ++At)
    {
        ++Size;
    }
    
    char *Dest = (char *)PushSize_(Arena, Size);
    for(u32 CharIndex = 0;
        CharIndex < Size;
        ++CharIndex)
    {
        Dest[CharIndex] = Source[CharIndex];
    }

    return(Dest);
}

inline void
CheckArena(memory_arena *Arena)
{
    Assert(Arena->TempCount == 0);
}

inline void
SubArena(memory_arena *Result, memory_arena *Arena, memory_index Size, memory_index Alignment = 16)
{
    Result->Size = Size;
    Result->Base = (uint8 *)PushSize_(Arena, Size, Alignment);
    Result->Used = 0;
    Result->TempCount = 0;
}

// ------------------------------------------------------------------------------------------------

struct memory_blocks
{
    struct header
    {
        memory_index Size;
        // NOTE(Kevin): Used to find the LEFT block header
        memory_index LeftSizeWithHeader;
        header      *Prev;
        header      *Next;
    };

    u8 *Base;
    memory_index Size;
    // TODO(Kevin): Something I could do is sort the free list by size
    header *FreeListHead;
    header *FreeListTail;
};
/*
    Memory blocks

    - My simple fixed-size general purpose memory allocator
    - Uses doubly linked list track free blocks of memory
    - Each block is aware of its own size and the size of the block to its left
    - Every non-left-most block is guaranteed to have a valid block to its left
    - Blocks merge with adjacent blocks upon freeing

    - O(1) allocation if all entries are same size, otherwise requires traversing free list
    - O(1) free always

*/

#define MEMBLOCKALIGNMENT 16
#define MEMBLOCKALIGN(Size) (((Size) + (MEMBLOCKALIGNMENT - 1)) & ~(MEMBLOCKALIGNMENT - 1))
#define MEMBLOCKHEADERSIZE MEMBLOCKALIGN(sizeof(memory_blocks::header))
#define MEMBLOCKMINBLOCKSIZE MEMBLOCKALIGNMENT

#define AllocStruct(MemBlocks, type) (type *)MemBlocksAlloc(MemBlocks, sizeof(type))
#define AllocArray(MemBlocks, Count, type) (type *)MemBlocksAlloc(MemBlocks, sizeof(type)*Count)
#define AllocSize(MemBlocks, Size) MemBlocksAlloc(MemBlocks, Size)
/* Allocation helpers for memory block allocator. */
#define FreeBlock(MemBlocks, PPtr) do { Assert(*PPtr); MemBlocksFree(MemBlocks, *PPtr); *PPtr = 0; } while (0);
/* Takes a pointer to an address allocated with MemBlocksAlloc. */

struct freeable_block
{
    memory_blocks *Allocator;
    void *Block;
};
#define FreeFreeableBlock(FreeableBlock) FreeBlock((FreeableBlock)->Allocator, &(FreeableBlock)->Block);

inline void
InitializeMemoryBlocks(memory_blocks *Blocks, memory_index Size, void *Base)
{
    umm BaseAddr = (umm)Base;
    Blocks->Base = (u8 *)MEMBLOCKALIGN(BaseAddr);
    umm BasePadding = (umm)Blocks->Base - BaseAddr;
    Blocks->Size = MEMBLOCKALIGN(Size - BasePadding) - MEMBLOCKALIGNMENT;
    Blocks->FreeListHead = (memory_blocks::header *)Blocks->Base;
    Blocks->FreeListTail = Blocks->FreeListHead;

    Blocks->FreeListHead->Size = Blocks->Size - MEMBLOCKHEADERSIZE;
    Blocks->FreeListHead->LeftSizeWithHeader = 0;
    Blocks->FreeListHead->Prev = Blocks->FreeListHead;
    Blocks->FreeListHead->Next = Blocks->FreeListHead;
}

inline void
InsertNewBlockAfter(memory_blocks::header *Current, memory_blocks::header *NewBlock)
{
    Assert(Current->Next->Prev == Current);
    // new block next bidirectional relationship
    NewBlock->Next = Current->Next;
    Current->Next->Prev = NewBlock;
    // new block prev bidirectional relationship
    Current->Next = NewBlock;
    NewBlock->Prev = Current;
}

inline void
PopBlock(memory_blocks *Blocks, memory_blocks::header *Current)
{
    Current->Prev->Next = Current->Next;
    Current->Next->Prev = Current->Prev;

    if (Blocks->FreeListHead == Current)
    {
        Blocks->FreeListHead = Current->Next;
    }
    if (Blocks->FreeListTail == Current)
    {
        Blocks->FreeListTail = Current->Prev;
    }
}

inline void
MarkUsed(memory_blocks::header *Header)
{
    Header->Prev = 0;
    Header->Next = 0;
}

inline void
MarkUnused(memory_blocks::header *Header)
{
    Header->Prev = (memory_blocks::header *)0xFF;
    Header->Next = (memory_blocks::header *)0xFF;
}

inline bool32
CheckUnused(memory_blocks::header *Header)
{
    bool32 Result = Header->Prev != 0;
    return Result;
}

inline bool32
IsValidHeader(memory_blocks *Blocks, memory_blocks::header *Header)
{
    bool32 Result = (umm)Header >= (umm)Blocks->Base && 
                    (umm)Header < (umm)((u8 *)Blocks->Base + Blocks->Size - MEMBLOCKMINBLOCKSIZE);
    return Result;
}

inline void *
MemBlocksAlloc(memory_blocks *Blocks, memory_index Size)
{
    void *Result = 0;

    if (CheckUnused(Blocks->FreeListHead))
    {
        // TODO(Kevin): Check for max alignment on this machine
        Size = MEMBLOCKALIGN(Size);
        memory_index SizeWithHeader = Size + MEMBLOCKHEADERSIZE;

        memory_blocks::header *Current = Blocks->FreeListHead;

        while (Current)
        {
            if (Current->Size >= Size)
            {
                if (Current->Size >= Size + MEMBLOCKHEADERSIZE + MEMBLOCKMINBLOCKSIZE)
                {
                    memory_blocks::header *NewBlock = (memory_blocks::header *)((u8 *)Current + SizeWithHeader);
                    InsertNewBlockAfter(Current, NewBlock);

                    NewBlock->Size = Current->Size - Size - MEMBLOCKHEADERSIZE;
                    NewBlock->LeftSizeWithHeader = SizeWithHeader;
                    Current->Size = Current->Size - NewBlock->Size - MEMBLOCKHEADERSIZE;

                    if (Blocks->FreeListTail == Current)
                    {
                        Blocks->FreeListTail = NewBlock;
                    }
                }

                // NOTE(Kevin): Currently, PopBlock doesn't do anything if the block
                //              is the only entry in the list. So, I can check if the
                //              FreeListHead block is in use to know whether the
                //              free list is empty.
                PopBlock(Blocks, Current);

                Result = (u8 *)Current + MEMBLOCKHEADERSIZE;
                MarkUsed(Current);
                break;
            }
            else
            {
                if (Current == Blocks->FreeListTail)
                {
                    break;
                }

                Current = Current->Next;
            }
        }
    }

    return Result;
}

inline bool32
GrowLeftBlock(memory_blocks *Blocks, memory_blocks::header *Header)
{
    if (Header->LeftSizeWithHeader == 0)
    {
        return false;
    }

    memory_blocks::header *Left = (memory_blocks::header *)((u8 *)Header - Header->LeftSizeWithHeader);
    memory_blocks::header *Right = (memory_blocks::header *)((u8 *)Header + Header->Size + MEMBLOCKHEADERSIZE);

    if (IsValidHeader(Blocks, Left) && CheckUnused(Left))
    {
        Left->Size += Header->Size + MEMBLOCKHEADERSIZE;

        if (IsValidHeader(Blocks, Right))
        {
            Right->LeftSizeWithHeader = Left->Size + MEMBLOCKHEADERSIZE;
        }

        return true;
    }
    else
    {
        return false;
    }
}

inline void
MemBlocksFree(memory_blocks *Blocks, void *Pointer)
{
    memory_blocks::header *Header = (memory_blocks::header *)((u8 *)Pointer - MEMBLOCKHEADERSIZE);
    MarkUnused(Header);
    memory_blocks::header *Right = (memory_blocks::header *)((u8 *)Header + Header->Size + MEMBLOCKHEADERSIZE);

    if (CheckUnused(Blocks->FreeListHead))
    {
        bool32 Merged = GrowLeftBlock(Blocks, Header);

        if (!Merged)
        {
            InsertNewBlockAfter(Blocks->FreeListTail, Header);
            Blocks->FreeListHead = Header; // TODO(Kevin): Add to tail or head by default?
        }

        if (IsValidHeader(Blocks, Right) && CheckUnused(Right))
        {
            Merged = GrowLeftBlock(Blocks, Right);

            if (Merged)
            {
                PopBlock(Blocks, Right);
            }
        }
    }
    else
    {
        // The free list is empty; this becomes the first new entry in the free list
        Blocks->FreeListHead = Header;
        Blocks->FreeListTail = Blocks->FreeListHead;
        Blocks->FreeListHead->Prev = Blocks->FreeListHead;
        Blocks->FreeListHead->Next = Blocks->FreeListHead;
    }
}

inline void *
MemBlocksRealloc()
{
    // TODO(Kevin): 
}

#endif // SHOOTER_MEMORY_H
