#include "mem.h"

// external
linear_arena_t StaticGameMemory;
linear_arena_t StaticLevelMemory;



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

template<typename T>
void *linear_arena_t::Alloc()
{
    return Alloc(sizeof(T), alignof(T));
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



