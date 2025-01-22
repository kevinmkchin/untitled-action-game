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

i32 ModifyASCIIBasedOnModifiers(i32 keycodeASCII, bool shift);

std::string& RemoveCharactersFromEndOfString(std::string& str, char c);

// normalized hsv to rgb
vec3 HSVToRGB(float h, float s, float v);

// normalized rgb to hsv
vec3 RGBToHSV(float r, float g, float b);



template<typename T, int _count> struct NiceArray
{
    /** Nice array wrapper for when you want to keep track of how many active/relevant
        elements are in the array. Essentially a dynamic sized array/vector with a
        maximum defined capacity (s.t. it can be defined on stack or static storage). */

    T data[_count] = {};
    int count = 0;
    const int capacity = _count;

    // todo maybe Insert and Erase?

    bool Contains(T v)
    {
        for (int i = 0; i < count; ++i)
        {
            if (*(data + i) == v) return true;
        }
        return false;
    }

    void EraseAt(int index)
    {
        if (index < count - 1)
        {
            memmove(data + index, data + index + 1, (count - index - 1) * sizeof(*data));
        }
        --count;
    }

    void EraseFirstOf(T v)
    {
        for (int i = 0; i < count; ++i)
        {
            if (*(data + i) == v)
            {
                EraseAt(i);
                break;
            }
        }
    }

    void EraseAllOf(T v)
    {
        for (int i = 0; i < count; ++i)
        {
            if (*(data + i) == v)
            {
                EraseAt(i);
            }
        }
    }

    bool NotAtCapacity()
    {
        return count < capacity;
    }

    void PushBack(T elem)
    {
        data[count] = elem;
        ++count;
    }

    void PopBack()
    {
        --count;
        memset(data + count, 0, sizeof(*data));
    }

    T& At(int index)
    {
        return *(data + index);
    }

    T& At(unsigned int index)
    {
        return At((int) index);
    }

    T& Back()
    {
        return *(data + count - 1);
    }

    void ResetCount()
    {
        count = 0;
    }

    void ResetToZero()
    {
        memset(data, 0, capacity * sizeof(*data));
    }
};


#pragma region MEMORY

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


struct MemoryLinearBuffer
{
    // Linear allocator works best when we don't support freeing memory at the pointer level
    // Carve allocations out of a pre alloced buffer

    // There is no per allocation overhead.
    // The buffer memory is not modified by the allocator.
    // The allocator is not thread-safe.

    u8* buffer = nullptr;
    size_t arenaOffset = 0;
    size_t bufferSize = 0;
};

void MemoryLinearInitialize(MemoryLinearBuffer *buffer, size_t sizeBytes);

#define MEMORY_LINEAR_ALLOCATE(buffer, type)\
    MemoryLinearAllocate(buffer, sizeof(type), alignof(type))

void *MemoryLinearAllocate(MemoryLinearBuffer *buffer, size_t wantedBytes, size_t align);

#pragma endregion

