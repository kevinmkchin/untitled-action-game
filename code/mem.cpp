#include "mem.h"
#define STB_DS_IMPLEMENTATION
#include <stb_ds.h>

// external
linear_arena_t StaticGameMemory;
linear_arena_t StaticLevelMemory;
linear_arena_t FrameMemory;
manualheap_arena_t JoltPhysicsMemory;


// returns pointer aligned forward to given alignment
static u8 *__PointerAlignForward(u8* ptr, size_t align)
{
    ASSERT((align & (align-1)) == 0) // power of 2  
    size_t p = (size_t)ptr;
    size_t modulo = p & (align-1); // cuz power of 2, faster than modulo
    if(modulo != 0)
    {
        return ptr + (align - modulo);
    }
    return ptr;
}

void linear_arena_t::Init(size_t Bytes)
{
    Arena = (u8 *)calloc(Bytes, 1);
    ArenaSize = Bytes;
    ArenaOffset = 0;
}

void *linear_arena_t::Alloc(size_t Bytes, size_t Align)
{
    u8 *current_ptr = Arena + ArenaOffset;
    u8 *aligned_ptr = __PointerAlignForward(current_ptr, Align);
    size_t offset = aligned_ptr - Arena;

    if (offset + Bytes <= ArenaSize)
    {
        void *ptr = Arena + offset;
        ArenaOffset = offset + Bytes;
        return ptr;
    }

    printf("Out of memory in given linear_arena_t");
    return nullptr;
}



void manualheap_arena_t::Init(size_t size)
{
    // Allocate 16-byte aligned memory for SIMD compatibility
    memory = static_cast<uint8_t*>(_aligned_malloc(size, 16));
    arenaSize = size;

    freeList = reinterpret_cast<BlockHeader*>(memory);
    freeList->size = size - sizeof(BlockHeader);
    freeList->free = true;
    freeList->next = nullptr;
}

void* manualheap_arena_t::alloc(size_t size, size_t alignment)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (alignment < alignof(std::max_align_t))
        alignment = alignof(std::max_align_t);
    size = alignUp(size, alignment);

    BlockHeader* current = freeList;

    while (current) {
        if (!current->free) {
            current = current->next;
            continue;
        }

        uintptr_t dataPtr = reinterpret_cast<uintptr_t>(current) + sizeof(BlockHeader);
        uintptr_t alignedDataPtr = (dataPtr + alignment - 1) & ~(alignment - 1);
        size_t padding = alignedDataPtr - dataPtr;
        size_t totalSize = padding + size;

        if (current->size >= totalSize) {
            if (current->size >= totalSize + sizeof(BlockHeader) + minBlockSize()) {
                splitBlock(current, totalSize);
            }

            current->free = false;
            return reinterpret_cast<void*>(alignedDataPtr);
        }

        current = current->next;
    }

    return nullptr; // out of memory
}

void manualheap_arena_t::free(void* ptr)
{
    if (!ptr)
        return;

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    BlockHeader* block = findBlock(ptr);
    if (block) {
        block->free = true;
        coalesce();
    }
}

void* manualheap_arena_t::realloc(void* ptr, size_t new_size, size_t alignment)
{
    if (!ptr)
        return alloc(new_size, alignment);

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (new_size == 0) {
        free(ptr);
        return nullptr;
    }

    BlockHeader* block = findBlock(ptr);
    if (!block)
        return nullptr;

    size_t old_size =
      block->size - (reinterpret_cast<uintptr_t>(ptr) - (reinterpret_cast<uintptr_t>(block) + sizeof(BlockHeader)));

    if (old_size >= new_size) {
        return ptr; // Reuse existing block
    }

    void* new_ptr = alloc(new_size, alignment);
    if (new_ptr) {
        std::memcpy(new_ptr, ptr, old_size);
        free(ptr);
    }

    return new_ptr;
}

void manualheap_arena_t::DebugPrint() 
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    std::cout << "=== SimpleArena Debug ===\n";
    const BlockHeader* current = reinterpret_cast<const BlockHeader*>(memory);
    size_t used = 0;
    size_t free = 0;
    size_t largestFree = 0;
    int index = 0;

    while (reinterpret_cast<const uint8_t*>(current) < memory + arenaSize) {
        const void* blockAddr = static_cast<const void*>(current);
        const void* dataAddr = reinterpret_cast<const uint8_t*>(current) + sizeof(BlockHeader);

        std::cout << "Block " << index++ << ": "
                  << (current->free ? "[FREE ]" : "[USED ]")
                  << " Size: " << current->size
                  << " bytes @ " << blockAddr
                  << ", Data @ " << dataAddr << "\n";

        if (current->free) {
            free += current->size;
            largestFree = std::max(largestFree, current->size);
        } else {
            used += current->size;
        }

        if (!current->next) break;
        current = current->next;
    }

    std::cout << "--------------------------------\n";
    std::cout << "Arena Size       : " << arenaSize << " bytes\n";
    std::cout << "Used Memory      : " << used << " bytes\n";
    std::cout << "Free Memory      : " << free << " bytes\n";
    std::cout << "Largest Free Block: " << largestFree << " bytes\n";
    std::cout << "================================\n\n";
}

size_t manualheap_arena_t::alignUp(size_t size, size_t alignment) const 
{ 
    return (size + alignment - 1) & ~(alignment - 1); 
}

size_t manualheap_arena_t::minBlockSize() const 
{
    return alignof(std::max_align_t); 
}

void manualheap_arena_t::splitBlock(BlockHeader* block, size_t usedSize)
{
    uint8_t* blockStart = reinterpret_cast<uint8_t*>(block);
    BlockHeader* newBlock = reinterpret_cast<BlockHeader*>(blockStart + sizeof(BlockHeader) + usedSize);

    newBlock->size = block->size - usedSize - sizeof(BlockHeader);
    newBlock->free = true;
    newBlock->next = block->next;

    block->size = usedSize;
    block->next = newBlock;
}

void manualheap_arena_t::coalesce()
{
    BlockHeader* current = freeList;
    while (current && current->next) {
        uint8_t* currentEnd = reinterpret_cast<uint8_t*>(current) + sizeof(BlockHeader) + current->size;
        if (current->free && current->next->free && reinterpret_cast<uint8_t*>(current->next) == currentEnd) {
            current->size += sizeof(BlockHeader) + current->next->size;
            current->next = current->next->next;
        } else {
            current = current->next;
        }
    }
}

manualheap_arena_t::BlockHeader* manualheap_arena_t::findBlock(void* ptr)
{
    BlockHeader* current = reinterpret_cast<BlockHeader*>(memory);
    while (reinterpret_cast<uint8_t*>(current) < memory + arenaSize) {
        uintptr_t dataStart = reinterpret_cast<uintptr_t>(current) + sizeof(BlockHeader);
        uintptr_t dataEnd = dataStart + current->size;
        if (reinterpret_cast<uintptr_t>(ptr) >= dataStart && reinterpret_cast<uintptr_t>(ptr) < dataEnd) {
            return current;
        }

        if (!current->next)
            break;
        current = current->next;
    }
    return nullptr;
}



