#pragma once

vec3 ScreenPointToWorldRay(ivec2 screenspaceCoords);
vec3 ScreenPointToWorldPoint(ivec2 screenspaceCoords, float z_NDC);
vec3 WorldPointToScreenPoint(vec3 worldPosition);

bool IntersectPlaneAndLine(vec3 pointOnPlane, vec3 normalOfPlane, 
    vec3 pointOnLine, vec3 directionOfLine, vec3 *intersectionPoint);
bool IntersectPlaneAndLineWithDirections(vec3 pointOnPlane, vec3 normalOfPlane, 
    vec3 pointOnLine, vec3 directionOfLine, vec3 *intersectionPoint);

float HorizontalFOVToVerticalFOV_RadianToRadian(float FOVXInRadians, float AspectRatio);

// Blit all of B into A at x y
//    A is destination
//    AW is destination bitmap resolution width (how many pixels across? not bytes)
//    AH is destination bitmap resolution height
//    B is source
//    BW is how many pixels wide source bitmap is 
//    BH is how many pixels tall source bitmap is
//    x and y is pixel X and Y in destination bitmap
//    pixelsz is size of each pixel (for RGBA32F, it would be sizeof(float)*4 assuming float is 32 bits)
//    for RGBA8, it would be sizeof(char)*4
void BlitRect(u8 *A, int AW, int AH, u8 *B, int BW, int BH, int x, int y, size_t pixelsz);

#define RGB255TO1(r,g,b) ((float)r)/255.f, ((float)g)/255.f, ((float)b)/255.f
#define RGBHEXTO1(hex) \
    float((hex & 0x00FF00000) >> 16)/255.f,\
    float((hex & 0x0000FF00) >> 8)/255.f,\
    float(hex & 0x000000FF)/255.f

template<typename T>
inline bool IsOneOfArray(T v, T* array, int count);

/* Pick random integer in range [min, max] inclusive. */
int RandomInt(int min, int max);

// Returns a random number [0..1]
float frand()
{
//  return ((float)(rand() & 0xffff)/(float)0xffff);
    return (float)rand()/(float)RAND_MAX;
}

i32 ModifyASCIIBasedOnModifiers(i32 keycodeASCII, bool shift);

std::string& RemoveCharactersFromEndOfString(std::string& str, char c);

// normalized hsv to rgb
vec3 HSVToRGB(float h, float s, float v);

// normalized rgb to hsv
vec3 RGBToHSV(float r, float g, float b);



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

    // dynamic_array(int size, bool setlength = false)
    // {
    //     if (setlength)
    //         setlen(size);
    //     else
    //         setcap(size);
    // }

    // Frees the array.
    void free();
    // Changes the length of the array to n. Allocates uninitialized
    // slots at the end if necessary.
    void setlen(int n);
    // Returns the number of elements in the array as an unsigned type.
    size_t lenu() const;
    // Sets the length of allocated storage to at least n. It will not
    // change the length of the array.
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


template<typename T, int _count> struct fixed_array
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

#pragma region MEMORY

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

struct ByteBuffer
{
    uint8_t* data = NULL;
    uint32_t position;  // basically the cursor
    uint32_t size;      // currently populated size
    uint32_t capacity;  // currently allocated size
};

#define BYTE_BUFFER_DEFAULT_CAPACITY 1024

// Allocate new byte buffer with default capacity
ByteBuffer ByteBufferNew();

// Generic write function
#define ByteBufferWrite(_buffer, T, _val)\
do {\
    ByteBuffer* _bb = (_buffer);\
    size_t _sz = sizeof(T);\
    T _v = _val;\
    __byteBufferWriteImpl(_bb, (void*)&(_v), _sz);\
} while(0)

// Generic read function
#define ByteBufferRead(_buffer, T, _val_p)\
do {\
    T* _v = (T*)(_val_p);\
    ByteBuffer* _bb = (_buffer);\
    *(_v) = *(T*)(_bb->data + _bb->position);\
    _bb->position += sizeof(T);\
} while(0)

// Useful when using byte buf as a stack
#define ByteBufferPop(_buffer, T, _val_p)\
do {\
    T* _v = (T*)(_val_p);\
    ByteBuffer *_bb = (_buffer);\
    _bb->position -= sizeof(T);\
    *(_v) = *(T*)(_bb->data + _bb->position);\
} while (0)

#define ByteBufferWriteBulk(_buffer, _data, _size)\
do {\
    __byteBufferWriteImpl(_buffer, _data, _size);\
} while(0)

#define ByteBufferReadBulk(_buffer, _dest_p, _size)\
do {\
    ByteBuffer* _bb = (_buffer);\
    memcpy((_dest_p), _bb->data + _bb->position, (_size));\
    _bb->position += (uint32_t)(_size);\
} while(0)

int ByteBufferWriteToFile(ByteBuffer* buffer, const char* filePath);

int ByteBufferReadFromFile(ByteBuffer* buffer, const char* filePath);

void ByteBufferInit(ByteBuffer* buffer);
void ByteBufferFree(ByteBuffer* buffer);
void ByteBufferClear(ByteBuffer* buffer);
void ByteBufferResize(ByteBuffer* buffer, size_t sz);
void ByteBufferSeekToStart(ByteBuffer* buffer);
void ByteBufferSeekToEnd(ByteBuffer* buffer);
void ByteBufferAdvancePosition(ByteBuffer* buffer, size_t sz);
void __byteBufferWriteImpl(ByteBuffer* buffer, void* data, size_t sz);

#pragma endregion // MEMORY

