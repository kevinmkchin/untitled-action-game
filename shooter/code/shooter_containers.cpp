#include "shooter_containers.h"

#include <iostream> // will cause conflict with my global / internal #defines

void manual_heap::Init(size_t Size)
{
    // Allocate 16-byte aligned memory for SIMD compatibility
    Memory = static_cast<uint8_t *>(_aligned_malloc(Size, 16));
    Init(Memory, Size);
}

void manual_heap::Init(void *Base, size_t Size)
{
    Memory = static_cast<uint8_t *>(Base);
    ArenaSize = Size;

    FreeList = reinterpret_cast<block *>(Memory);
    FreeList->Size = Size - sizeof(block);
    FreeList->Free = true;
    FreeList->Next = NULL;    
}

void *manual_heap::Alloc(size_t Size, size_t Align)
{
    std::lock_guard<std::recursive_mutex> lock(HeapMutex);

    if (Align < alignof(std::max_align_t))
    {
        Align = alignof(std::max_align_t);
    }
    Size = AlignUp(Size, Align);

    block *Current = FreeList;

    while (Current)
    {
        if (!Current->Free)
        {
            Current = Current->Next;
            continue;
        }

        uintptr_t DataPtr = reinterpret_cast<uintptr_t>(Current) + sizeof(block);
        uintptr_t AlignedDataPtr = (DataPtr + Align - 1) & ~(Align - 1);
        size_t Padding = AlignedDataPtr - DataPtr;
        size_t TotalSize = Padding + Size;

        if (Current->Size >= TotalSize)
        {
            if (Current->Size >= TotalSize + sizeof(block) + MinBlockSize())
            {
                SplitBlock(Current, TotalSize);
            }

            Current->Free = false;
            return reinterpret_cast<void *>(AlignedDataPtr);
        }

        Current = Current->Next;
    }

    return nullptr;
}

void manual_heap::Free(void* Ptr)
{
    if (!Ptr)
    {
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(HeapMutex);

    block *Block = FindBlock(Ptr);
    if (Block)
    {
        Block->Free = true;
        Coalesce();
    }
}

void* manual_heap::Realloc(void* Ptr, size_t NewSize, size_t Align)
{
    if (!Ptr)
    {
        return Alloc(NewSize, Align);
    }

    std::lock_guard<std::recursive_mutex> lock(HeapMutex);

    if (NewSize == 0)
    {
        free(Ptr);
        return nullptr;
    }

    block *Block = FindBlock(Ptr);
    if (!Block)
    {
        return nullptr;
    }

    size_t OldSize = Block->Size - 
                     (reinterpret_cast<uintptr_t>(Ptr) - (reinterpret_cast<uintptr_t>(Block) + sizeof(block)));

    if (OldSize >= NewSize)
    {
        return Ptr; // Reuse existing block
    }

    void* NewPtr = Alloc(NewSize, Align);
    if (NewPtr)
    {
        memcpy(NewPtr, Ptr, OldSize);
        free(Ptr);
    }

    return NewPtr;
}

size_t manual_heap::AlignUp(size_t Size, size_t Align)
{ 
    return (Size + Align - 1) & ~(Align - 1); 
}

size_t manual_heap::MinBlockSize()
{
    return alignof(std::max_align_t); 
}

void manual_heap::SplitBlock(block *Block, size_t UsedSize)
{
    uint8_t* BlockStart = reinterpret_cast<uint8_t*>(Block);
    block *NewBlock = reinterpret_cast<block *>(BlockStart + sizeof(Block) + UsedSize);

    NewBlock->Size = Block->Size - UsedSize - sizeof(block);
    NewBlock->Free = true;
    NewBlock->Next = Block->Next;

    Block->Size = UsedSize;
    Block->Next = NewBlock;
}

void manual_heap::Coalesce()
{
    block *Current = FreeList;
    while (Current && Current->Next)
    {
        uint8_t* CurrentEnd = reinterpret_cast<uint8_t*>(Current) + sizeof(block) + Current->Size;
        if (Current->Free && Current->Next->Free && reinterpret_cast<uint8_t*>(Current->Next) == CurrentEnd)
        {
            Current->Size += sizeof(block) + Current->Next->Size;
            Current->Next = Current->Next->Next;
        }
        else
        {
            Current = Current->Next;
        }
    }
}

manual_heap::block *manual_heap::FindBlock(void* Ptr)
{
    block *Current = reinterpret_cast<block *>(Memory);
    while (reinterpret_cast<uint8_t*>(Current) < Memory + ArenaSize)
    {
        uintptr_t dataStart = reinterpret_cast<uintptr_t>(Current) + sizeof(block);
        uintptr_t dataEnd = dataStart + Current->Size;
        if (reinterpret_cast<uintptr_t>(Ptr) >= dataStart && reinterpret_cast<uintptr_t>(Ptr) < dataEnd)
        {
            return Current;
        }

        if (!Current->Next)
        {
            break;
        }
        Current = Current->Next;
    }
    return nullptr;
}

void manual_heap::DebugPrint() 
{
    std::lock_guard<std::recursive_mutex> lock(HeapMutex);

    std::cout << "=== ManualHeap Debug ===\n";
    const block *Current = reinterpret_cast<const block *>(Memory);
    size_t Used = 0;
    size_t Free = 0;
    size_t LargestFree = 0;
    int Index = 0;

    while (reinterpret_cast<const uint8_t*>(Current) < Memory + ArenaSize) {
        const void* BlockAddress = static_cast<const void*>(Current);
        const void* DataAddress = reinterpret_cast<const uint8_t*>(Current) + sizeof(block);

        std::cout << "Block " << Index++ << ": "
                  << (Current->Free ? "[FREE ]" : "[Used ]")
                  << " Size: " << Current->Size
                  << " bytes @ " << BlockAddress
                  << ", Data @ " << DataAddress << "\n";

        if (Current->Free)
        {
            Free += Current->Size;
            LargestFree = (LargestFree > Current->Size) ? LargestFree : Current->Size;
        }
        else
        {
            Used += Current->Size;
        }

        if (!Current->Next)
        {
            break;
        }
        Current = Current->Next;
    }

    std::cout << "--------------------------------\n";
    std::cout << "Arena Size       : " << ArenaSize << " bytes\n";
    std::cout << "Used Memory      : " << Used << " bytes\n";
    std::cout << "Free Memory      : " << Free << " bytes\n";
    std::cout << "Largest Free Block: " << LargestFree << " bytes\n";
    std::cout << "================================\n\n";
}
