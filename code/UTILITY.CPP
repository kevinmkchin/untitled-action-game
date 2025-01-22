

vec3 ScreenPointToWorldRay(ivec2 screenspaceCoords);
vec3 ScreenPointToWorldPoint(ivec2 screenspaceCoords, float z_NDC);
vec3 WorldPointToScreenPoint(vec3 worldPosition);

float HorizontalFOVToVerticalFOV_RadianToRadian(float FOVXInRadians, float AspectRatio)
{
    float FOVY = 2.0f * atan(tan(FOVXInRadians / 2.0f) / AspectRatio);
    return FOVY;
}

#define RGB255TO1(r,g,b) ((float)r)/255.f, ((float)g)/255.f, ((float)b)/255.f
#define RGBHEXTO1(hex) \
    float((hex & 0x00FF00000) >> 16)/255.f,\
    float((hex & 0x0000FF00) >> 8)/255.f,\
    float(hex & 0x000000FF)/255.f

template<typename T>
inline bool IsOneOfArray(T v, T* array, int count)
{
    for (int i = 0; i < count; ++i)
        if (v == *(array + i)) return true;
    return false;
}

/* Pick random integer in range [min, max] inclusive. */
int RandomInt(int min, int max)
{
    int retval = min + (rand() % static_cast<int>(max - min + 1));
    return retval;
}

// Blit all of B into A at x y
void BlitRect(u8 *A, int AW, int AH, u8 *B, int BW, int BH, int x, int y, size_t pixelsz)
{
    AW *= (int)pixelsz;
    BW *= (int)pixelsz;
    x *= (int)pixelsz;
    for (int row = 0; row < BH; ++row)
    {
        int desty = y + row;
        if (desty < 0 || desty >= AH)
            continue;
        u8 *dest = A + (desty * AW + x);
        u8 *src  = B + row * BW;
        int cpyw = BW;
        if (x < 0) 
        {
            dest -= x;
            src -= x;
            cpyw += x;
        }
        if (x + BW > AW) 
        {
            cpyw = AW - x;
        }
        memcpy(dest, src, cpyw);
    }
}

i32 ModifyASCIIBasedOnModifiers(i32 keycodeASCII, bool shift)
{
    i32 keycode = keycodeASCII;

    if (shift)
    {
        if (97 <= keycode && keycode <= 122)
        {
            keycode -= 32;
        }
        else if (keycode == 50)
        {
            keycode = 64;
        }
        else if (49 <= keycode && keycode <= 53)
        {
            keycode -= 16;
        }
        else if (91 <= keycode && keycode <= 93)
        {
            keycode += 32;
        }
        else
        {
            switch (keycode)
            {
                case 48: { keycode = 41; } break;
                case 54: { keycode = 94; } break;
                case 55: { keycode = 38; } break;
                case 56: { keycode = 42; } break;
                case 57: { keycode = 40; } break;
                case 45: { keycode = 95; } break;
                case 61: { keycode = 43; } break;
                case 39: { keycode = 34; } break;
                case 59: { keycode = 58; } break;
                case 44: { keycode = 60; } break;
                case 46: { keycode = 62; } break;
                case 47: { keycode = 63; } break;
                case 96: { keycode = 126; } break;
            }
        }
    }

    return keycode;
}

std::string& RemoveCharactersFromEndOfString(std::string& str, char c)
{
    while (str.back() == c)
    {
        str.pop_back();
    }
    return str;
}

// normalized hsv to rgb
vec3 HSVToRGB(float h, float s, float v)
{
    // https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    u16 huedegree = u16(h * 359.f);
    float chroma = v*s;
    float m = v-chroma;
    float normalizedx = chroma * (1.f - abs(fmod((h * 359.f) / 60.f, 2.f) - 1.f));

    vec3 rgb;
    if (huedegree < 60)
        rgb = { chroma, normalizedx, 0.f };
    else if (huedegree < 120)
        rgb = { normalizedx, chroma, 0.f };
    else if (huedegree < 180)
        rgb = { 0.f, chroma, normalizedx };
    else if (huedegree < 240)
        rgb = { 0.f, normalizedx, chroma };
    else if (huedegree < 300)
        rgb = { normalizedx, 0.f, chroma };
    else if (huedegree < 360)
        rgb = { chroma, 0.f, normalizedx };
    rgb += vec3(m,m,m);
    return rgb;
}

// normalized rgb to hsv
vec3 RGBToHSV(float r, float g, float b)
{
    // https://en.wikipedia.org/wiki/Hue
    // https://www.rapidtables.com/convert/color/rgb-to-hsv.html

    float h = atan2(1.7320508f * (g - b), 2 * r - g - b);
    if (h < 0)
        h = GM_TWOPI + h;
    h = abs(h) / GM_TWOPI;

    float cmax = GM_max(r, GM_max(g, b));
    float cmin = GM_min(r, GM_min(g, b));
    float cdelta = cmax - cmin;
    float s = cmax == 0.f ? 0.f : cdelta / cmax;

    vec3 hsv = vec3(h, s, cmax);
    return hsv;
}



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

#pragma region BYTEBUFFERS

struct ByteBuffer
{
    uint8_t* data = NULL;
    uint32_t position;  // basically the cursor
    uint32_t size;      // currently populated size
    uint32_t capacity;  // currently allocated size
};

#define BYTE_BUFFER_DEFAULT_CAPACITY 1024

void ByteBufferInit(ByteBuffer* buffer)
{
    buffer->data = (uint8_t*) malloc(BYTE_BUFFER_DEFAULT_CAPACITY);
    memset(buffer->data, 0, BYTE_BUFFER_DEFAULT_CAPACITY);
    buffer->position = 0;
    buffer->size = 0;
    buffer->capacity = BYTE_BUFFER_DEFAULT_CAPACITY;
}

ByteBuffer ByteBufferNew()
{
    ByteBuffer buffer = {0};
    ByteBufferInit(&buffer);
    return buffer;
}

void ByteBufferFree(ByteBuffer* buffer)
{
    if(buffer && buffer->data) 
    {
        free(buffer->data);
    }
    buffer->size = 0;
    buffer->position = 0;
    buffer->capacity = 0;
}

void ByteBufferClear(ByteBuffer* buffer)
{
    buffer->size = 0;
    buffer->position = 0;   
}

void ByteBufferResize(ByteBuffer* buffer, size_t sz)
{
    uint8_t* data = (uint8_t*)realloc(buffer->data, sz);
    if(data == NULL)
    {
        return;
    }
    buffer->data = data;
    buffer->capacity = (uint32_t)sz;
}

void ByteBufferSeekToStart(ByteBuffer* buffer)
{
    buffer->position = 0;
}

void ByteBufferSeekToEnd(ByteBuffer* buffer)
{
    buffer->position = buffer->size;
}

void ByteBufferAdvancePosition(ByteBuffer* buffer, size_t sz)
{
    buffer->position += (uint32_t)sz; 
}

void __byteBufferWriteImpl(ByteBuffer* buffer, void* data, size_t sz)
{
    size_t totalWriteSize = buffer->position + sz;
    if(totalWriteSize >= buffer->capacity)
    {
        size_t capacity = buffer->capacity ? buffer->capacity * 2 : BYTE_BUFFER_DEFAULT_CAPACITY;
        while(capacity < totalWriteSize)
        {
            capacity *= 2;
        }
        ByteBufferResize(buffer, capacity);
    }
    memcpy(buffer->data + buffer->position, data, sz);
    buffer->position += (uint32_t)sz;
    buffer->size += (uint32_t)sz;
}

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
#define ByteBufferPop(_buffer, T, _val_p) \
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

int ByteBufferWriteToFile(ByteBuffer* buffer, const char* filePath)
{
    FILE* fp;
    fp = fopen(filePath, "wb");

    if(!fp)
    {
        return 0;
    }

    fwrite(buffer->data, 1, buffer->size, fp);

    fclose(fp);

    return 1;
}

int ByteBufferReadFromFile(ByteBuffer* buffer, const char* filePath)
{
    if(!buffer)
    {
        return 0;
    }

    FILE* fp;
    fp = fopen(filePath, "rb");

    if(!fp)
    {
        return 0;
    }

    if(buffer->data)
    {
        ByteBufferFree(buffer);
    }

    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    ByteBuffer bb = ByteBufferNew();
    if(bb.capacity < sz)
    {
        size_t capacity = bb.capacity;
        while(capacity < sz)
        {
            capacity *= 2;
        }
        ByteBufferResize(&bb, capacity);
    }

    fread(bb.data, 1, sz, fp);
    bb.size = (uint32_t)sz;
    *buffer = bb;

    fclose(fp);

    return 1;
}

#pragma endregion

#pragma region MEMORY_ALLOCATORS

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

void MemoryLinearInitialize(MemoryLinearBuffer *buffer, size_t sizeBytes)
{
    buffer->buffer = (u8*) calloc(sizeBytes, 1);
    buffer->bufferSize = sizeBytes;
    buffer->arenaOffset = 0;
}

void *MemoryLinearAllocate(MemoryLinearBuffer *buffer, size_t wantedBytes, size_t align)
{
    u8 *current_ptr = buffer->buffer + buffer->arenaOffset;
    u8 *aligned_ptr = __PointerAlignForward(current_ptr, align);
    size_t offset = aligned_ptr - buffer->buffer;

    if(offset + wantedBytes <= buffer->bufferSize)
    {
        void* ptr = buffer->buffer + offset;
        buffer->arenaOffset = offset + wantedBytes;
        return ptr;
    }
    
    printf("Out of memory in given MemoryLinearBuffer");
    return nullptr;
}

#define MEMORY_LINEAR_ALLOCATE(buffer, type) MemoryLinearAllocate(buffer, sizeof(type), alignof(type))

#pragma endregion







#pragma region MATH_HELPERS

// only checks in direction from
bool IntersectPlaneAndLine(vec3 pointOnPlane, vec3 normalOfPlane, vec3 pointOnLine, vec3 directionOfLine, vec3 *intersectionPoint)
{
    // vec3 pp = pointOnPlane;
    // vec3 pn = normalOfPlane;
    // vec3 lp = pointOnLine;
    // vec3 lv = directionOfLine;
    // line x = lp.x + lv.x * t, line y = lp.y + lv.y * t, line z = lp.z + lv.z * t
    // plane(X, Y, Z) = pn.x * X + pn.y * Y + pn.z * Z - Dot(pn, pp)
    // pn.x * (lp.x + lv.x * t) + pn.y * (lp.y + lv.y * t) + pn.z * (lp.z + lv.z * t) = Dot(pn, pp)
    // pn.x*lp.x + pn.x*lv.x*t + pn.y*lp.y + pn.y*lv.y*t + pn.z*lp.z + pn.z*lv.z*t = Dot(pn, pp)
    // pn.x*lv.x*t + pn.y*lv.y*t + pn.z*lv.z*t = Dot(pn, pp) - pn.x*lp.x - pn.y*lp.y - pn.z*lp.z
    // t (pn.x*lv.x + pn.y*lv.y + pn.z*lv.z) = Dot(pn, pp) - pn.x*lp.x - pn.y*lp.y - pn.z*lp.z
    // t = (Dot(pn, pp) - pn.x*lp.x - pn.y*lp.y - pn.z*lp.z) / (pn.x*lv.x + pn.y*lv.y + pn.z*lv.z)
    // t = (Dot(pn, pp) - (pn.x*lp.x + pn.y*lp.y + pn.z*lp.z)) / (pn.x*lv.x + pn.y*lv.y + pn.z*lv.z)

    float denominator = Dot(normalOfPlane, directionOfLine);
    if (GM_abs(denominator) < 0.000001f)
        return false;

    float t = (Dot(normalOfPlane, pointOnPlane) - Dot(normalOfPlane, pointOnLine)) / denominator;
    *intersectionPoint = pointOnLine + directionOfLine * t;
    return true;
}

bool IntersectPlaneAndLineWithDirections(vec3 pointOnPlane, vec3 normalOfPlane, vec3 pointOnLine, vec3 directionOfLine, vec3 *intersectionPoint)
{
    float denominator = Dot(normalOfPlane, directionOfLine);
    if (denominator > -0.000001f)
        return false;

    float t = (Dot(normalOfPlane, pointOnPlane) - Dot(normalOfPlane, pointOnLine)) / denominator;
    *intersectionPoint = pointOnLine + directionOfLine * t;
    return true;
}

#pragma endregion








