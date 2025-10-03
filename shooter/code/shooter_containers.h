#pragma once

#include <mutex> // shit is slow to compile

template<typename T, int C> struct carray
{
    /** Nice wrapper for fixed size C arrays */

    T Data[C] = {};
    int Count = 0;
    const int Capacity = C;

    bool Contains(T v);
    // Deletes the element at Data[Index], moving the rest of the array over.
    void Delete(int Index);
    // Deletes the element at Data[Index], replacing it with the element from
    // the end of the array. O(1) performance.
    void DeleteSwap(int Index);
    // Call Delete for the first element that equals v
    void DeleteFirst(T v);
    // Call Delete for every element that equals v
    void DeleteEvery(T v);
    // Count < Capacity
    bool NotAtCap();
    // Appends the item to the end of array
    void Put(T v);
    // Removes the final element of the array and returns it.
    T Pop();
    // Retrieve the final element
    T &Back();
    // Set Count to zero
    void ResetCount();
    // Set the bits of the entire buffer to 0
    void MemsetZero();

    T& operator[](int Index) { Assert(0 <= Index && Index <= (int)Count); return Data[Index]; }
    T& operator[](unsigned int Index) { Assert(Index <= (unsigned int)Count); return Data[Index]; }
    T& operator[](size_t Index) { Assert(Index <= (size_t)Count); return Data[Index]; }
    const T& operator[](int Index) const { Assert(0 <= Index && Index <= (int)Count); return Data[Index]; }
    const T& operator[](unsigned int Index) const { Assert(Index <= (unsigned int)Count); return Data[Index]; }
    const T& operator[](size_t Index) const { Assert(Index <= (size_t)Count); return Data[Index]; }
};

template<typename T, int C>
bool carray<T,C>::Contains(T v)
{
    for (int i = 0; i < Count; ++i)
    {
        if (*(Data + i) == v) return true;
    }
    return false;
}

template<typename T, int C>
void carray<T,C>::Delete(int Index)
{
    if (Index < Count - 1)
    {
        memmove(Data + Index, Data + Index + 1, (Count - Index - 1) * sizeof(*Data));
    }
    --Count;
}

template<typename T, int C>
void carray<T,C>::DeleteSwap(int Index)
{
    if (Index < Count - 1)
    {
        Data[Index] = Data[Count - 1];
    }
    --Count;
}

template<typename T, int C>
void carray<T,C>::DeleteFirst(T v)
{
    for (int i = 0; i < Count; ++i)
    {
        if (*(Data + i) == v)
        {
            Delete(i);
            break;
        }
    }
}

template<typename T, int C>
void carray<T,C>::DeleteEvery(T v)
{
    for (int i = 0; i < Count; ++i)
    {
        if (*(Data + i) == v)
        {
            Delete(i);
        }
    }
}

template<typename T, int C>
bool carray<T,C>::NotAtCap()
{
    return Count < Capacity;
}

template<typename T, int C>
void carray<T,C>::Put(T v)
{
    Data[Count] = v;
    ++Count;
}

template<typename T, int C>
T carray<T,C>::Pop()
{
    T copy = *(Data + Count - 1);
    --Count;
    memset(Data + Count, 0, sizeof(*Data));
    return copy;
}

template<typename T, int C>
T &carray<T,C>::Back()
{
    return *(Data + Count - 1);
}

template<typename T, int C>
void carray<T,C>::ResetCount()
{
    Count = 0;
}

template<typename T, int C>
void carray<T,C>::MemsetZero()
{
    memset(Data, 0, Capacity * sizeof(*Data));
}

struct manual_heap
{
    // SIMD compatible thread-safe heap allocator

    void Init(size_t Size);
    void Init(void *Base, size_t Size);
    void *Alloc(size_t Size, size_t Align = 16);
    void Free(void* Ptr);
    void *Realloc(void* Ptr, size_t NewSize, size_t Align = 16);

    void DebugPrint();

private:
    struct block
    {
        size_t Size;
        bool Free;
        block *Next;
    };

    uint8_t *Memory;
    size_t ArenaSize;
    block *FreeList;
    std::recursive_mutex HeapMutex;

    size_t AlignUp(size_t Size, size_t Align);
    size_t MinBlockSize();
    void SplitBlock(block *Block, size_t UsedSize);
    void Coalesce();
    block *FindBlock(void *Ptr);
};
