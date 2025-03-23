
vec3 ScreenPointToWorldRay(ivec2 screenspaceCoords)
{
    // https://antongerdelan.net/opengl/raycasting.html
    // assuming 3D world view is taking up the entire window
    // Reversing perspective divide not necessary because this is a vector/direction/ray with no intrinsic depth.
    float x_NDC = ((float)screenspaceCoords.x / (float)BackbufferWidth) * 2.f - 1.f;
    float y_NDC = (float(BackbufferHeight - screenspaceCoords.y) / (float)BackbufferHeight) * 2.f - 1.f;
    vec4 ray_in_clipspace = vec4(x_NDC, y_NDC, -1.f, 1.f);
    vec4 ray_in_viewspace = LevelEditor.ActivePerspectiveMatrix.GetInverse() * ray_in_clipspace;
    ray_in_viewspace = vec4(ray_in_viewspace.x, ray_in_viewspace.y, -1.f, 0.f);
    vec3 ray_in_worldspace = Normalize((LevelEditor.ActiveViewMatrix.GetInverse() * ray_in_viewspace).xyz);
    return ray_in_worldspace;
}

vec3 ScreenPointToWorldPoint(ivec2 screenspaceCoords, float z_NDC)
{
    float x_NDC = ((float)screenspaceCoords.x / (float)BackbufferWidth) * 2.f - 1.f;
    float y_NDC = (float(BackbufferHeight - screenspaceCoords.y) / (float)BackbufferHeight) * 2.f - 1.f;
    vec4 point_in_clipspace = vec4(x_NDC, y_NDC, z_NDC, 1.f);
    // For points, reverse perspective divide after the inverse projection matrix transformation because it's easier that way.
    vec4 point_in_viewspace_before_perspective_divide = LevelEditor.ActivePerspectiveMatrix.GetInverse() * point_in_clipspace;
    vec4 point_in_viewspace = point_in_viewspace_before_perspective_divide / point_in_viewspace_before_perspective_divide.w;
    vec4 point_in_worldspace = LevelEditor.ActiveViewMatrix.GetInverse() * point_in_viewspace;
    return point_in_worldspace.xyz;
}

vec3 WorldPointToScreenPoint(vec3 worldPosition)
{
    vec4 clipspaceCoordinates = LevelEditor.ActivePerspectiveMatrix * LevelEditor.ActiveViewMatrix * vec4(worldPosition, 1.f);
    float screenspaceRatioX = ((clipspaceCoordinates.x / clipspaceCoordinates.w) + 1.f) / 2.f;
    float screenspaceRatioY = 1.f - (((clipspaceCoordinates.y / clipspaceCoordinates.w) + 1.f) / 2.f);
    float internalResolutionWidth = (float)BackbufferWidth;
    float internalResolutionHeight = (float)BackbufferHeight;
    float distanceFromCameraWCS = Dot(worldPosition - LevelEditor.EditorCam.Position, LevelEditor.EditorCam.Direction);
    return vec3(screenspaceRatioX * internalResolutionWidth, screenspaceRatioY * internalResolutionHeight, distanceFromCameraWCS);
}

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

float HorizontalFOVToVerticalFOV_RadianToRadian(float FOVXInRadians, float AspectRatio)
{
    float FOVY = 2.0f * atan(tan(FOVXInRadians / 2.0f) / AspectRatio);
    return FOVY;
}

template<typename T>
inline bool IsOneOfArray(T v, T* array, int count)
{
    for (int i = 0; i < count; ++i)
        if (v == *(array + i)) return true;
    return false;
}

int RandomInt(int min, int max)
{
    int retval = min + (rand() % static_cast<int>(max - min + 1));
    return retval;
}

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
bool fixed_array<T,C>::contains(T v)
{
    for (int i = 0; i < count; ++i)
    {
        if (*(data + i) == v) return true;
    }
    return false;
}

template<typename T, int C>
void fixed_array<T,C>::del(int index)
{
    if (index < count - 1)
    {
        memmove(data + index, data + index + 1, (count - index - 1) * sizeof(*data));
    }
    --count;
}

template<typename T, int C>
void fixed_array<T,C>::del_first(T v)
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
void fixed_array<T,C>::del_every(T v)
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
bool fixed_array<T,C>::not_at_cap()
{
    return count < capacity;
}

template<typename T, int C>
void fixed_array<T,C>::put(T elem)
{
    data[count] = elem;
    ++count;
}

template<typename T, int C>
T fixed_array<T,C>::pop()
{
    T copy = *(data + count - 1);
    --count;
    memset(data + count, 0, sizeof(*data));
    return copy;
}

template<typename T, int C>
T &fixed_array<T,C>::back()
{
    return *(data + count - 1);
}

template<typename T, int C>
void fixed_array<T,C>::reset_count()
{
    count = 0;
}

template<typename T, int C>
void fixed_array<T,C>::memset_zero()
{
    memset(data, 0, capacity * sizeof(*data));
}
