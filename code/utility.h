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


struct rng_t
{
    rng_t()
    {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    rng_t(u32 Seed)
        : rng(Seed)
    {}

    std::mt19937 rng; // much better RNG than rand()

    // Random float [-1, 1]
    float frand() { return NextFloat(-1.f, 1.f); }
    // Random float [0, 1]
    float frand01() { return NextFloat(0.f, 1.f); }
    // Random float [min, max]
    float NextFloat(float min, float max) 
    {
        std::uniform_real_distribution<float> FloatDistrib(min, max);
        return FloatDistrib(rng);
    }
    // Random integer [min, max]
    int NextInt(int min, int max) 
    { 
        std::uniform_int_distribution<int> IntDistrib(min, max);
        return IntDistrib(rng);
    }

    vec3 Direction()
    {
        // Generate random vector until it's not zero-length
        vec3 Dir;
        do 
        {
            Dir.x = frand();
            Dir.y = frand();
            Dir.z = frand();
        } 
        while (Magnitude(Dir) == 0.0f);
        return Normalize(Dir);
    }

    quat Orientation()
    {
        float u1, u2, u3;
        float w, x, y, z;

        // Generate three random numbers in [0, 1]
        u1 = frand01();
        u2 = frand01();
        u3 = frand01();

        // Convert to a uniformly distributed unit quaternion
        float sqrt1_minus_u1 = sqrtf(1.0f - u1);
        float sqrt_u1 = sqrtf(u1);

        w = cosf(2.0f * GM_PI * u2) * sqrt1_minus_u1;
        x = sinf(2.0f * GM_PI * u2) * sqrt1_minus_u1;
        y = cosf(2.0f * GM_PI * u3) * sqrt_u1;
        z = sinf(2.0f * GM_PI * u3) * sqrt_u1;

        return quat(w,x,y,z);
    }
};

extern rng_t RNG;
extern rng_t ENEMYRNG;
extern rng_t SOUNDRNG;

float frand01()
{
    return RNG.frand01();
}


i32 ShiftASCII(i32 keycodeASCII, bool shift);

std::string& RemoveCharactersFromEndOfString(std::string& str, char c);

// normalized hsv to rgb
vec3 HSVToRGB(float h, float s, float v);

// normalized rgb to hsv
vec3 RGBToHSV(float r, float g, float b);



#pragma region BYTEBUFFER

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

#pragma endregion // BYTEBUFFER

