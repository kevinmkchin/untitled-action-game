#pragma once

#include "common.h"
#define STBDS_REALLOC(c,p,s) ::realloc(p,s)
#define STBDS_FREE(c,p)      ::free(p) // ensure global namespace
#include <stb_ds.h>

enum class MemoryType
{
    Malloc,
    Game,
    Level,
    Frame
};

// === ARENAS ===
struct linear_arena_t
{
    // Linear allocator works best when we don't support freeing memory at the pointer level
    // Carve allocations out of a pre allocated arena
    // There is no per allocation overhead.
    // The arena memory is not modified by the allocator.
    // The allocator is not thread-safe.

    u8 *Arena = nullptr;
    size_t ArenaOffset = 0;
    size_t ArenaSize = 0;

    void Init(size_t Bytes);

    template<typename T>
    void *Alloc();

    void *Alloc(size_t Bytes, size_t Align);
};


struct manualheap_arena_t
{
    // SIMD compatible thread-safe heap allocator
    // Used for Jolt Physics library

    void Init(size_t size);
    void* alloc(size_t size, size_t alignment = 16);
    void free(void* ptr);
    void* realloc(void* ptr, size_t new_size, size_t alignment = 16);

    void DebugPrint();

private:
    struct BlockHeader
    {
        size_t size;
        bool free;
        BlockHeader* next;
    };

    uint8_t* memory = nullptr;
    size_t arenaSize = 0;
    BlockHeader* freeList = nullptr;
    std::recursive_mutex mutex_;

    size_t alignUp(size_t size, size_t alignment) const;
    size_t minBlockSize() const;
    void splitBlock(BlockHeader* block, size_t usedSize);
    void coalesce();
    BlockHeader* findBlock(void* ptr);
};


// MY PRE ALLOCATED MEMORY ARENAS
extern linear_arena_t StaticGameMemory;
extern linear_arena_t StaticLevelMemory;
extern linear_arena_t FrameMemory;
#define new_InGameMemory(type) new (StaticGameMemory.Alloc<type>()) type
#define new_InLevelMemory(type) new (StaticLevelMemory.Alloc<type>()) type
#define new_InFrameMemory(type) new (FrameMemory.Alloc<type>()) type

extern manualheap_arena_t JoltPhysicsMemory;


// === CONTAINERS ===
template<typename T> struct dynamic_array
{
    // stb_ds.h dynamic array wrapper

    // NOTES - DYNAMIC ARRAY
    //
    //   * If you know how long a dynamic array is going to be in advance, you can avoid
    //     extra memory allocations by using arrsetlen to allocate it to that length in
    //     advance and use foo[n] while filling it out, or arrsetcap to allocate the memory
    //     for that length and use arrput/arrpush as normal.
    //
    //   * Unlike some other versions of the dynamic array, this version should
    //     be safe to use with strict-aliasing optimizations.
    //
    //   * Doesn't wrap STL classes well...

    T *data = NULL;

    dynamic_array()
    {
    }

    // Frees the array.
    void free();
    // Changes the length of the array to n. Allocates uninitialized
    // slots at the end if necessary.
    void setlen(int n);
    // Returns the number of elements in the array as an unsigned type.
    size_t lenu() const;
    // Sets the length of allocated storage to at least n. It will not
    // change the length of the array. basically std::vector::reserve
    void setcap(int n);
    // Returns the number of total elements the array can contain without
    // needing to be reallocated.
    size_t cap();
    // Removes the final element of the array and returns it.
    T pop();
    // Appends the item to the end of array. Returns item.
    T put(T item);
    // Inserts the item into the middle of array, into array[p],
    // moving the rest of the array over. Returns item.
    T ins(int p, T item);
    // Inserts n uninitialized items into array starting at array[p],
    // moving the rest of the array over.
    void insn(int p, int n);
    // Appends n uninitialized items onto array at the end.
    // Returns a pointer to the first uninitialized item added.
    T *addnptr(int n);
    // Appends n uninitialized items onto array at the end.
    // Returns the index of the first uninitialized item added.
    size_t addnindex(int n);
    // Deletes the element at a[p], moving the rest of the array over.
    void del(int p);
    // Deletes n elements starting at a[p], moving the rest of the array over.
    void deln(int p, int n);
    // Deletes the element at a[p], replacing it with the element from
    // the end of the array. O(1) performance.
    void delswap(int p);

    T& operator[](int index) { ASSERT(0 <= index && index <= (int)arrlenu(data)); return data[index]; }
    T& operator[](unsigned int index) { ASSERT(index <= (unsigned int)arrlenu(data)); return data[index]; }
    T& operator[](size_t index) { ASSERT(index <= arrlenu(data)); return data[index]; }
    const T& operator[](int index) const { ASSERT(0 <= index && index <= (int)arrlenu(data)); return data[index]; }
    const T& operator[](unsigned int index) const { ASSERT(index <= (unsigned int)arrlenu(data)); return data[index]; }
    const T& operator[](size_t index) const { ASSERT(index <= arrlenu(data)); return data[index]; }
};

template<typename T, int _count> struct c_array
{
    /** Nice wrapper for fixed size C arrays */

    T data[_count] = {};
    int count = 0;
    const int capacity = _count;

    bool contains(T v);
    // Deletes the element at data[index], moving the rest of the array over.
    void del(int index);
    // Call del for the first element that equals v
    void del_first(T v);
    // Call del for every element that equals v
    void del_every(T v);
    // count < capacity
    bool not_at_cap();
    // Appends the item to the end of array
    void put(T elem);
    // Removes the final element of the array and returns it.
    T pop();
    // Retrieve the final element
    T &back();
    // Set count to zero
    void reset_count();
    // Set the bits of the entire buffer to 0
    void memset_zero();

    T& operator[](int index) { ASSERT(0 <= index && index <= (int)count); return data[index]; }
    T& operator[](unsigned int index) { ASSERT(index <= (unsigned int)count); return data[index]; }
    T& operator[](size_t index) { ASSERT(index <= (size_t)count); return data[index]; }
    const T& operator[](int index) const { ASSERT(0 <= index && index <= (int)count); return data[index]; }
    const T& operator[](unsigned int index) const { ASSERT(index <= (unsigned int)count); return data[index]; }
    const T& operator[](size_t index) const { ASSERT(index <= (size_t)count); return data[index]; }
};

template<typename T> struct fixed_array
{
    /** Create fixed array with provided memory type. */

    T *data;
    u32 length;
    u32 capacity;
private:
    MemoryType mem;
public:
    fixed_array() : data(0), length(0), capacity(0), mem(MemoryType::Malloc) {}
    fixed_array(u32 Capacity, MemoryType Mem) 
        : length(0)
        , capacity(Capacity)
        , mem(Mem)
    {
        switch (Mem)
        {
            case MemoryType::Malloc:
                data = (T*)std::malloc(sizeof(T)*Capacity);
                break;
            case MemoryType::Game:
                data = (T*)StaticGameMemory.Alloc(sizeof(T)*Capacity, alignof(T));
                break;
            case MemoryType::Level:
                data = (T*)StaticLevelMemory.Alloc(sizeof(T)*Capacity, alignof(T));
                break;
            case MemoryType::Frame:
                data = (T*)FrameMemory.Alloc(sizeof(T)*Capacity, alignof(T));
                break;
        }
    }

    // Frees the array.
    void free();
    // Changes the length of the array to n with uninitialized slots.
    void setlen(u32 n);
    // Returns the number of elements in the array as an unsigned type.
    u32 lenu() const;
    // Returns the number of total elements the array can contain.
    u32 cap() const;
    // Removes the final element of the array and returns it.
    T pop();
    // Appends the item to the end of array. Returns item.
    T put(T item);
    // Inserts the item into the middle of array, into array[p],
    // moving the rest of the array over. Returns item.
    T ins(u32 p, T item);
    // Inserts n uninitialized items into array starting at array[p],
    // moving the rest of the array over. If exceeds capacity, the items
    // moved past the end are removed.
    void insn(u32 p, u32 n);
    // Appends n uninitialized items onto array at the end.
    // Returns a pointer to the first uninitialized item added.
    T *addnptr(u32 n);
    // Appends n uninitialized items onto array at the end.
    // Returns the index of the first uninitialized item added.
    u32 addnindex(u32 n);
    // Deletes the element at a[p], moving the rest of the array over.
    void del(u32 p);
    // Deletes n elements starting at a[p], moving the rest of the array over.
    void deln(u32 p, u32 n);
    // Deletes the element at a[p], replacing it with the element from
    // the end of the array. O(1) performance.
    void delswap(u32 p);

    T& operator[](int index) { ASSERT(0 <= index && index <= (int)length); return data[index]; }
    T& operator[](unsigned int index) { ASSERT(index <= (unsigned int)length); return data[index]; }
    T& operator[](size_t index) { ASSERT(index <= (size_t)length); return data[index]; }
    const T& operator[](int index) const { ASSERT(0 <= index && index <= (int)length); return data[index]; }
    const T& operator[](unsigned int index) const { ASSERT(index <= (unsigned int)length); return data[index]; }
    const T& operator[](size_t index) const { ASSERT(index <= (size_t)length); return data[index]; }

    T* begin() { return data; }
    T* end() { return data + length; }
    const T* begin() const { return data; }
    const T* end() const { return data + length; }
};

template<typename T> struct mem_indexer
{
    /** Provide own block of memory and count. For when I want to carve 
        an array out of pre-allocated arena and index into it. Smaller
        than fixed_array. */

    T *data = 0;
    int count = 0;

    mem_indexer() : data(0), count(0) {}
    mem_indexer(void *Block, size_t Count) : data((T*)Block), count((int)Count) {}

    T& operator[](int index) { ASSERT(0 <= index && index <= (int)count); return data[index]; }
    T& operator[](unsigned int index) { ASSERT(index <= (unsigned int)count); return data[index]; }
    T& operator[](size_t index) { ASSERT(index <= (size_t)count); return data[index]; }
    const T& operator[](int index) const { ASSERT(0 <= index && index <= (int)count); return data[index]; }
    const T& operator[](unsigned int index) const { ASSERT(index <= (unsigned int)count); return data[index]; }
    const T& operator[](size_t index) const { ASSERT(index <= (size_t)count); return data[index]; }
};


// templated shit
template<typename T>
void *linear_arena_t::Alloc()
{
    return Alloc(sizeof(T), alignof(T));
}

template<typename T> 
void dynamic_array<T>::free()
{
    arrfree(data);
}

template<typename T> 
void dynamic_array<T>::setlen(int n)
{
    arrsetlen(data, n);
}

template<typename T> 
size_t dynamic_array<T>::lenu() const
{
    return arrlenu(data);
}

template<typename T> 
void dynamic_array<T>::setcap(int n)
{
    arrsetcap(data, n);
}

template<typename T> 
size_t dynamic_array<T>::cap()
{
    return arrcap(data);
}

template<typename T> 
T dynamic_array<T>::pop()
{
    return arrpop(data);
}

template<typename T> 
T dynamic_array<T>::put(T item)
{
    return arrput(data, item);
}

template<typename T>
T dynamic_array<T>::ins(int p, T item)
{
    return arrins(data, p, item);
}

template<typename T> 
void dynamic_array<T>::insn(int p, int n)
{
    arrinsn(data, p, n);
}

template<typename T> 
T *dynamic_array<T>::addnptr(int n)
{
    return arraddnptr(data, n);
}

template<typename T> 
size_t dynamic_array<T>::addnindex(int n)
{
    return arraddnindex(data, n);
}

template<typename T> 
void dynamic_array<T>::del(int p)
{
    arrdel(data, p);
}

template<typename T> 
void dynamic_array<T>::deln(int p, int n)
{
    arrdeln(data, p, n);
}

template<typename T> 
void dynamic_array<T>::delswap(int p)
{
    arrdelswap(data, p);
}


template<typename T> 
void fixed_array<T>::free()
{
    switch (mem)
    {
        case MemoryType::Malloc:
            std::free(data);
            break;
        case MemoryType::Game:
        case MemoryType::Level:
            LogWarning("Shouldn't try to free fixed_array allocated in static game memory!");
            break;
    }
    data = 0;
    length = 0;
    capacity = 0;
}

template<typename T> 
void fixed_array<T>::setlen(u32 n)
{
    if (n > capacity)
    {
        LogError("fixed_array cannot set length to greater than capacity");
        ASSERT(0);
    }

    length = n;
}

template<typename T> 
u32 fixed_array<T>::lenu() const
{
    return length;
}

template<typename T> 
u32 fixed_array<T>::cap() const
{
    return capacity;
}

template<typename T> 
T fixed_array<T>::pop()
{
    T copy = *(data + length - 1);
    --length;
    return copy;
}

template<typename T> 
T fixed_array<T>::put(T item)
{
    if (length >= capacity)
    {
        LogError("fixed array is at capacity and cannot put additional items");
        ASSERT(0);
    }

    data[length] = item;
    ++length;
    return data[length-1];
}

template<typename T>
T fixed_array<T>::ins(u32 p, T item)
{
    if (length + 1 > capacity)
    {
        LogError("fixed array is at capacity and cannot insert additional items");
        ASSERT(0);
    }

    memcpy(data+p+1, data+p, length-p);
    length += 1;
    data[p] = item;
    return data[p];
}

template<typename T> 
void fixed_array<T>::insn(u32 p, u32 n)
{
    memcpy(data+p+n, data+p, std::min(length-p, capacity-p-n));
    length = std::min(length + n, capacity);
}

template<typename T> 
T *fixed_array<T>::addnptr(u32 n)
{
    if (length + n > capacity)
    {
        LogError("fixed array is at capacity and cannot insert additional items");
        ASSERT(0);
    }

    T *ptr = data + length;
    length += n;
    return ptr;
}

template<typename T> 
u32 fixed_array<T>::addnindex(u32 n)
{
    if (length + n > capacity)
    {
        LogError("fixed array is at capacity and cannot insert additional items");
        ASSERT(0);
    }

    u32 index = length;
    length += n;
    return index;
}

template<typename T> 
void fixed_array<T>::del(u32 p)
{
    memcpy(data+p, data+p+1, length-(p+1));
    length -= 1;
}

template<typename T> 
void fixed_array<T>::deln(u32 p, u32 n)
{
    memcpy(data+p, data+p+n, length-(p+n));
    length -= n;
}

template<typename T> 
void fixed_array<T>::delswap(u32 p)
{
    if (p >= length)
        return;
    data[p] = data[length-1];
    length -= 1;
}


template<typename T, int C>
bool c_array<T,C>::contains(T v)
{
    for (int i = 0; i < count; ++i)
    {
        if (*(data + i) == v) return true;
    }
    return false;
}

template<typename T, int C>
void c_array<T,C>::del(int index)
{
    if (index < count - 1)
    {
        memmove(data + index, data + index + 1, (count - index - 1) * sizeof(*data));
    }
    --count;
}

template<typename T, int C>
void c_array<T,C>::del_first(T v)
{
    for (int i = 0; i < count; ++i)
    {
        if (*(data + i) == v)
        {
            del(i);
            break;
        }
    }
}

template<typename T, int C>
void c_array<T,C>::del_every(T v)
{
    for (int i = 0; i < count; ++i)
    {
        if (*(data + i) == v)
        {
            del(i);
        }
    }
}

template<typename T, int C>
bool c_array<T,C>::not_at_cap()
{
    return count < capacity;
}

template<typename T, int C>
void c_array<T,C>::put(T elem)
{
    data[count] = elem;
    ++count;
}

template<typename T, int C>
T c_array<T,C>::pop()
{
    T copy = *(data + count - 1);
    --count;
    memset(data + count, 0, sizeof(*data));
    return copy;
}

template<typename T, int C>
T &c_array<T,C>::back()
{
    return *(data + count - 1);
}

template<typename T, int C>
void c_array<T,C>::reset_count()
{
    count = 0;
}

template<typename T, int C>
void c_array<T,C>::memset_zero()
{
    memset(data, 0, capacity * sizeof(*data));
}



