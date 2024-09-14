

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
    uint8_t* data;
    uint32_t position;  // basically the cursor
    uint32_t size;      // currently populated size
    uint32_t capacity;  // currently allocated size
};

#define BYTE_BUFFER_DEFAULT_CAPACITY 1024

void ByteBufferInit(ByteBuffer* buffer)
{
    // TODO(Kevin): If using custom memory alloc, replace the following line
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







#if MESA_WINDOWS
#include <ShObjIdl.h>
#include <codecvt>

std::string OpenFilePrompt(u32 fileTypesCount, COMDLG_FILTERSPEC fileTypes[])
{
    std::string filePathString;
    IFileOpenDialog *pFileOpen;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                          IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (SUCCEEDED(hr))
    {
        hr = pFileOpen->SetFileTypes(fileTypesCount, fileTypes);

        // Show the Open dialog box.
        hr = pFileOpen->Show(NULL);
        // Get the file name from the dialog box.
        if (SUCCEEDED(hr))
        {
            IShellItem *pItem;
            hr = pFileOpen->GetResult(&pItem);
            if (SUCCEEDED(hr))
            {
                PWSTR pszFilePath;
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                if (SUCCEEDED(hr))
                {
                    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converterX;
                    filePathString = converterX.to_bytes(pszFilePath);
                    CoTaskMemFree(pszFilePath);
                }
                pItem->Release();
            }
        }
        pFileOpen->Release();
    }
    CoUninitialize();
    return filePathString;
}

std::vector<std::string> OpenMultipleFilesPrompt(u32 fileTypesCount, COMDLG_FILTERSPEC fileTypes[])
{
    std::vector<std::string> filePathStrings;
    IFileOpenDialog *pFileOpen;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                          IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (SUCCEEDED(hr))
    {
        DWORD dwFlags;
        hr = pFileOpen->GetOptions(&dwFlags);
        if (SUCCEEDED(hr)) {
            hr = pFileOpen->SetOptions(dwFlags | FOS_ALLOWMULTISELECT);
        }
        hr = pFileOpen->SetFileTypes(fileTypesCount, fileTypes);

        // Show the Open dialog box.
        hr = pFileOpen->Show(NULL);
        if (SUCCEEDED(hr)) 
        {
            // Get the results from the dialog
            IShellItemArray* pItems;
            hr = pFileOpen->GetResults(&pItems);

            if (SUCCEEDED(hr)) 
            {
                // Get the number of selected items
                DWORD itemCount;
                hr = pItems->GetCount(&itemCount);

                if (SUCCEEDED(hr)) 
                {
                    for (DWORD i = 0; i < itemCount; i++) 
                    {
                        // Get each selected item
                        IShellItem* pItem;
                        hr = pItems->GetItemAt(i, &pItem);

                        if (SUCCEEDED(hr)) 
                        {
                            // Get the display name of the item (i.e., the file path)
                            PWSTR pszFilePath;
                            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                            if (SUCCEEDED(hr)) 
                            {
                                // Output the file path (for example, print it to console)
                                //wprintf(L"Selected file: %s\n", pszFilePath);

                                std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converterX;
                                filePathStrings.push_back(converterX.to_bytes(pszFilePath));
                                CoTaskMemFree(pszFilePath); // Free the memory allocated for the file path
                            }

                            pItem->Release();  // Release the current IShellItem
                        }
                    }
                }

                pItems->Release();  // Release the IShellItemArray
            }
        }
    }
    CoUninitialize();
    return filePathStrings;
}

std::string SaveFilePrompt(COMDLG_FILTERSPEC fileTypes[], const wchar_t *defaultExtension)
{
    std::string filePathString;
    IFileSaveDialog* pFileSave;
    HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL,
                                  IID_IFileSaveDialog, reinterpret_cast<void**>(&pFileSave));
    if (SUCCEEDED(hr))
    {
        hr = pFileSave->SetFileTypes(1, fileTypes);
        pFileSave->SetDefaultExtension(defaultExtension);

        // Show the Open dialog box.
        hr = pFileSave->Show(NULL);
        // Get the file name from the dialog box.
        if (SUCCEEDED(hr))
        {
            IShellItem *pItem;
            hr = pFileSave->GetResult(&pItem);
            if (SUCCEEDED(hr))
            {
                PWSTR pszFilePath;
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                if (SUCCEEDED(hr))
                {
                    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converterX;
                    filePathString = converterX.to_bytes(pszFilePath);
                    CoTaskMemFree(pszFilePath);
                }
                pItem->Release();
            }
        }
        pFileSave->Release();
    }
    CoUninitialize();
    return filePathString;
}

std::string OpenPNGImageDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"PNG files", L"*.png" } };
    std::string imagepath = OpenFilePrompt(1, fileTypes);
    return imagepath;
}

std::string OpenEditableMapFormatDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"Editable map files", L"*.emf" } };
    std::string path = OpenFilePrompt(1, fileTypes);
    return path;
}

std::string SaveEditableMapFormatDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"Editable map files", L"*.emf" } };
    std::string path = SaveFilePrompt(fileTypes, L"emf");
    return path;
}

// std::string OpenGameMapDialog()
// {
//     COMDLG_FILTERSPEC fileTypes[] = { { L"Game map files", L"*.map" } };
//     std::string path = OpenFilePrompt(1, fileTypes);
//     return path;
// }

std::string SaveGameMapDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"Game map files", L"*.map" } };
    std::string path = SaveFilePrompt(fileTypes, L"map");
    return path;
}

std::vector<std::string> OpenImageFilesDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = {
        { L"JPG, PNG, BMP files", L"*.jpg;*.jpeg;*.png;*.bmp" },
    };
    return OpenMultipleFilesPrompt(1, fileTypes);
}

#endif


#pragma region PRIMITIVES_AND_HANDLES_DRAWING

static u32 PRIM_VERTEX_POS_AND_COLOR_VAO;
static u32 PRIM_VERTEX_POS_AND_COLOR_VBO;

static NiceArray<float, 256000> PRIMITIVE_TRIS_VB;
static GPUShader PRIMITIVES_TRIS_SHADER;
static const char* PRIMITIVES_TRIS_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec4 col;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out vec4 baseColour;\n"
    "void main() {\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "    baseColour = col;\n"
    "}\n";
static const char* PRIMITIVES_TRIS_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in vec4 baseColour;\n"
    "void main() {\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(baseColour.xyz, baseColour.w * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(baseColour.xyz, baseColour.w);\n"
    "    }\n"
    "}\n";

static u32 PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO;
static u32 PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO;

static NiceArray<float, 256000> PRIMITIVE_FATLINES_VB;
static GPUShader FATLINES_SHADER;
static const char* FATLINES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec4 col;\n"
    "layout (location = 2) in float width;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out float gs_distance_from_eye;\n"
    "out vec4 gs_baseColour;\n"
    "out float gs_lineWidth;\n"
    "void main() {\n"
    "    vec4 viewspace_pos = viewMatrix * vec4(pos, 1.0);\n"
    "    gl_Position = projectionMatrix * viewspace_pos;\n"
    "    gs_distance_from_eye = -viewspace_pos.z;\n"
    "    gs_baseColour = col;\n"
    "    gs_lineWidth = width;\n"
    "}\n";
static const char* FATLINES_SHADER_GS = 
    "#version 330 core\n"
    "layout(lines) in;\n"
    "layout(triangle_strip, max_vertices = 4) out;\n"
    "\n"
    "in float gs_distance_from_eye[];\n"
    "in vec4 gs_baseColour[];\n"
    "in float gs_lineWidth[];\n"
    "out float distance_from_eye;\n"
    "out vec4 baseColour;\n"
    "void main() {\n"
    "    vec4 p0 = gl_in[0].gl_Position;"
    "    vec4 p1 = gl_in[1].gl_Position;"
    "\n"
    "    vec2 direction = normalize(p1.xy - p0.xy);\n"
    "    vec2 offset = vec2(-direction.y, direction.x) * gs_lineWidth[0] * 0.5;\n"
    "\n"
    "    vec2 offset0 = offset * p0.w;  // Apply perspective scaling to offset for p0\n"
    "    vec2 offset1 = offset * p1.w;  // Apply perspective scaling to offset for p1\n"
    "    // Emit four vertices for the quad\n"
    "    distance_from_eye = gs_distance_from_eye[0];\n"
    "    baseColour = gs_baseColour[0];\n"
    "    gl_Position = p0 + vec4(offset0, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "    gl_Position = p0 - vec4(offset0, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    distance_from_eye = gs_distance_from_eye[1];\n"
    "    baseColour = gs_baseColour[1];\n"
    "    gl_Position = p1 + vec4(offset1, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "    gl_Position = p1 - vec4(offset1, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    EndPrimitive();\n"
    "}";

static const char* FATLINES_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in float distance_from_eye;\n"
    "in vec4 baseColour;\n"
    "void main() {\n"
    "    float fadeMult = clamp(exp(-pow((distance_from_eye * 0.0001), 5)), 0.0, 1.0);\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult);\n"
    "    }\n"
    "}\n";

static NiceArray<float, 256000> PRIMITIVE_LINES_VB;
static GPUShader LINES_SHADER;
static const char* LINES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec4 col;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out float distance_from_eye;\n"
    "out vec4 baseColour;\n"
    "void main() {\n"
    "    vec4 viewspace_pos = viewMatrix * vec4(pos, 1.0);\n"
    "    gl_Position = projectionMatrix * viewspace_pos;\n"
    "    distance_from_eye = -viewspace_pos.z;\n"
    "    baseColour = col;\n"
    "}\n";
static const char* LINES_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in float distance_from_eye;\n"
    "in vec4 baseColour;\n"
    "void main() {\n"
    "    float fadeMult = clamp(exp(-pow((distance_from_eye * 0.0001), 5)), 0.0, 1.0);\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult);\n"
    "    }\n"
    "}\n";
static bool DrawAxisLines = false;

static GPUFrameBuffer mousePickingRenderTarget;
static u32 HANDLES_VAO = 0;
static u32 HANDLES_VBO = 0;
static GPUShader HANDLES_SHADER;
static const char* HANDLES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec3 idRGB;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out vec3 handlesIdRGB;\n"
    "void main() {\n"
    "    handlesIdRGB = idRGB;\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "}\n";
static const char* HANDLES_SHADER_FS =
    "#version 330 core\n"
    "in vec3 handlesIdRGB;\n"
    "out vec4 colour;\n"
    "void main() {\n"
    "    colour = vec4(handlesIdRGB, 1.0);\n"
    "}\n";
static NiceArray<float, 256 * 128> HANDLES_VB;

static u32 GRID_MESH_VAO;
static u32 GRID_MESH_VBO;
static GPUShader GRID_MESH_SHADER;
static const char* GRID_MESH_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "uniform mat4 transformMatrix;\n"
    "out float distance_from_eye;\n"
    "void main() {\n"
    "    vec4 viewspace_pos = viewMatrix * transformMatrix * vec4(pos, 1.0);\n"
    "    gl_Position = projectionMatrix * viewspace_pos;\n"
    "    distance_from_eye = -viewspace_pos.z;\n"
    "}\n";
static const char* GRID_MESH_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in float distance_from_eye;\n"
    "void main() {\n"
    "    float fadeMult = clamp(exp(-pow((distance_from_eye * 0.0001), 5)), 0.0, 1.0);\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(0.6,0.6,0.6, 0.25 * fadeMult * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(0.6,0.6,0.6, 0.25 * fadeMult);\n"
    "    }\n"
    "}\n";

void InitPrimitivesAndHandlesSystems()
{
    glGenVertexArrays(1, &PRIM_VERTEX_POS_AND_COLOR_VAO);
    glBindVertexArray(PRIM_VERTEX_POS_AND_COLOR_VAO);
    glGenBuffers(1, &PRIM_VERTEX_POS_AND_COLOR_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_AND_COLOR_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*7, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float)*7, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(PRIMITIVES_TRIS_SHADER, PRIMITIVES_TRIS_SHADER_VS, PRIMITIVES_TRIS_SHADER_FS);
    GLCreateShaderProgram(LINES_SHADER, LINES_SHADER_VS, LINES_SHADER_FS);

    glGenVertexArrays(1, &PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO);
    glBindVertexArray(PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO);
    glGenBuffers(1, &PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*8, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(sizeof(float)*7));
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(FATLINES_SHADER, FATLINES_SHADER_VS, FATLINES_SHADER_GS, FATLINES_SHADER_FS);

    glGenVertexArrays(1, &HANDLES_VAO);
    glBindVertexArray(HANDLES_VAO);
    glGenBuffers(1, &HANDLES_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, HANDLES_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(HANDLES_SHADER, HANDLES_SHADER_VS, HANDLES_SHADER_FS);

    vec3 gridmeshdata[4001*4];
    for (int i = -2000; i <= 2000; ++i)
    {
        gridmeshdata[(i+2000)*4+0] = vec3((float)i, 0.f, -2000.f);
        gridmeshdata[(i+2000)*4+1] = vec3((float)i, 0.f, 2000.f);
        gridmeshdata[(i+2000)*4+2] = vec3(-2000.f, 0.f, (float)i);
        gridmeshdata[(i+2000)*4+3] = vec3(2000.f, 0.f, (float)i);
    }
    glGenVertexArrays(1, &GRID_MESH_VAO);
    glBindVertexArray(GRID_MESH_VAO);
    glGenBuffers(1, &GRID_MESH_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, GRID_MESH_VBO);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(4001*4*sizeof(vec3)), (void*)gridmeshdata, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(GRID_MESH_SHADER, GRID_MESH_SHADER_VS, GRID_MESH_SHADER_FS);

    mousePickingRenderTarget.width = 4;
    mousePickingRenderTarget.height = 4;
    CreateGPUFrameBuffer(&mousePickingRenderTarget);
}

void DrawGrid(float scale, vec3 translation, const mat4 *projectionMatrix, const mat4 *viewMatrix, GLuint sceneDepthTextureId, vec2 framebufferSize)
{
    mat4 transformMatrix = TranslationMatrix(translation) * ScaleMatrix(scale, scale, scale);

    UseShader(GRID_MESH_SHADER);
    GLBindMatrix4fv(GRID_MESH_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
    GLBindMatrix4fv(GRID_MESH_SHADER, "viewMatrix", 1, viewMatrix->ptr());
    GLBindMatrix4fv(GRID_MESH_SHADER, "transformMatrix", 1, transformMatrix.ptr());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
    GLBind1i(GRID_MESH_SHADER, "sceneDepthTexture", 0);
    GLBind2f(GRID_MESH_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

    GLBind1f(GRID_MESH_SHADER, "occludedOpacity", 0.0f);
    glBindVertexArray(GRID_MESH_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, GRID_MESH_VBO);
    glDrawArrays(GL_LINES, 0, 4001*4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void PrimitiveDrawAll(const mat4 *projectionMatrix, const mat4 *viewMatrix, GLuint sceneDepthTextureId, vec2 framebufferSize)
{
    if (PRIMITIVE_LINES_VB.count > 0)
    {
        UseShader(LINES_SHADER);
        GLBindMatrix4fv(LINES_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
        GLBindMatrix4fv(LINES_SHADER, "viewMatrix", 1, viewMatrix->ptr());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
        GLBind1i(LINES_SHADER, "sceneDepthTexture", 0);
        GLBind2f(LINES_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

        GLBind1f(LINES_SHADER, "occludedOpacity", 0.0f);
        glBindVertexArray(PRIM_VERTEX_POS_AND_COLOR_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_AND_COLOR_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * PRIMITIVE_LINES_VB.count, PRIMITIVE_LINES_VB.data,
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, PRIMITIVE_LINES_VB.count / 7);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        PRIMITIVE_LINES_VB.ResetCount();
    }

    if (PRIMITIVE_FATLINES_VB.count > 0)
    {
        UseShader(FATLINES_SHADER);
        GLBindMatrix4fv(FATLINES_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
        GLBindMatrix4fv(FATLINES_SHADER, "viewMatrix", 1, viewMatrix->ptr());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
        GLBind1i(FATLINES_SHADER, "sceneDepthTexture", 0);
        GLBind2f(FATLINES_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

        GLBind1f(FATLINES_SHADER, "occludedOpacity", 0.33f);
        glBindVertexArray(PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * PRIMITIVE_FATLINES_VB.count, PRIMITIVE_FATLINES_VB.data,
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, PRIMITIVE_FATLINES_VB.count / 7);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        PRIMITIVE_FATLINES_VB.ResetCount();
    }

    if (PRIMITIVE_TRIS_VB.count > 0)
    {
        UseShader(PRIMITIVES_TRIS_SHADER);
        GLBindMatrix4fv(PRIMITIVES_TRIS_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
        GLBindMatrix4fv(PRIMITIVES_TRIS_SHADER, "viewMatrix", 1, viewMatrix->ptr());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
        GLBind1i(PRIMITIVES_TRIS_SHADER, "sceneDepthTexture", 0);
        GLBind2f(PRIMITIVES_TRIS_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

        GLBind1f(PRIMITIVES_TRIS_SHADER, "occludedOpacity", 0.5f);
        glBindVertexArray(PRIM_VERTEX_POS_AND_COLOR_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_AND_COLOR_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * PRIMITIVE_TRIS_VB.count, PRIMITIVE_TRIS_VB.data,
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, PRIMITIVE_TRIS_VB.count / 7);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        PRIMITIVE_TRIS_VB.ResetCount();
    }
}

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color)
{
    vec3 tangent = Normalize(Cross(normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, normal));
    quat q = quat(GM_DEG2RAD * 30.f, normal);
    for(int i = 0; i < 12; ++i)
    {
        vec3 a = center;
        vec3 b = center + tangent * radius;
        tangent = RotateVector(tangent, q);
        vec3 c = center + tangent * radius;

        PRIMITIVE_TRIS_VB.PushBack(a.x);
        PRIMITIVE_TRIS_VB.PushBack(a.y);
        PRIMITIVE_TRIS_VB.PushBack(a.z);
        PRIMITIVE_TRIS_VB.PushBack(color.x);
        PRIMITIVE_TRIS_VB.PushBack(color.y);
        PRIMITIVE_TRIS_VB.PushBack(color.z);
        PRIMITIVE_TRIS_VB.PushBack(color.w);

        PRIMITIVE_TRIS_VB.PushBack(b.x);
        PRIMITIVE_TRIS_VB.PushBack(b.y);
        PRIMITIVE_TRIS_VB.PushBack(b.z);
        PRIMITIVE_TRIS_VB.PushBack(color.x);
        PRIMITIVE_TRIS_VB.PushBack(color.y);
        PRIMITIVE_TRIS_VB.PushBack(color.z);
        PRIMITIVE_TRIS_VB.PushBack(color.w);

        PRIMITIVE_TRIS_VB.PushBack(c.x);
        PRIMITIVE_TRIS_VB.PushBack(c.y);
        PRIMITIVE_TRIS_VB.PushBack(c.z);
        PRIMITIVE_TRIS_VB.PushBack(color.x);
        PRIMITIVE_TRIS_VB.PushBack(color.y);
        PRIMITIVE_TRIS_VB.PushBack(color.z);
        PRIMITIVE_TRIS_VB.PushBack(color.w);
    }
}

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius)
{
    PrimitiveDrawSolidDisc(center, normal, radius, vec4(RGB255TO1(248, 230, 60), 1.f));
}

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color)
{
    PRIMITIVE_LINES_VB.PushBack(p1.x);
    PRIMITIVE_LINES_VB.PushBack(p1.y);
    PRIMITIVE_LINES_VB.PushBack(p1.z);
    PRIMITIVE_LINES_VB.PushBack(color.x);
    PRIMITIVE_LINES_VB.PushBack(color.y);
    PRIMITIVE_LINES_VB.PushBack(color.z);
    PRIMITIVE_LINES_VB.PushBack(color.w);

    PRIMITIVE_LINES_VB.PushBack(p2.x);
    PRIMITIVE_LINES_VB.PushBack(p2.y);
    PRIMITIVE_LINES_VB.PushBack(p2.z);
    PRIMITIVE_LINES_VB.PushBack(color.x);
    PRIMITIVE_LINES_VB.PushBack(color.y);
    PRIMITIVE_LINES_VB.PushBack(color.z);
    PRIMITIVE_LINES_VB.PushBack(color.w);
}

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color, float thickness)
{
    PRIMITIVE_FATLINES_VB.PushBack(p1.x);
    PRIMITIVE_FATLINES_VB.PushBack(p1.y);
    PRIMITIVE_FATLINES_VB.PushBack(p1.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.x);
    PRIMITIVE_FATLINES_VB.PushBack(color.y);
    PRIMITIVE_FATLINES_VB.PushBack(color.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.w);
    PRIMITIVE_FATLINES_VB.PushBack(thickness*0.0025f);

    PRIMITIVE_FATLINES_VB.PushBack(p2.x);
    PRIMITIVE_FATLINES_VB.PushBack(p2.y);
    PRIMITIVE_FATLINES_VB.PushBack(p2.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.x);
    PRIMITIVE_FATLINES_VB.PushBack(color.y);
    PRIMITIVE_FATLINES_VB.PushBack(color.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.w);
    PRIMITIVE_FATLINES_VB.PushBack(thickness*0.0025f);
}


vec3 HandleIdToRGB(u32 id)
{
    float r = (float) ((id & 0x000000FF) >> 0);
    float g = (float) ((id & 0x0000FF00) >> 8);
    float b = (float) ((id & 0x00FF0000) >> 16);
    return vec3(r,g,b)/255.f;
}

void DoDiscHandle(u32 id, vec3 worldpos, vec3 normal, float radius)
{
    vec3 idrgb = HandleIdToRGB(id);

    vec3 tangent = Normalize(Cross(normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, normal));
    quat q = quat(GM_DEG2RAD * 30.f, normal);
    for(int i = 0; i < 12; ++i)
    {
        vec3 a = worldpos;
        vec3 b = worldpos + tangent * radius;
        tangent = RotateVector(tangent, q);
        vec3 c = worldpos + tangent * radius;

        HANDLES_VB.PushBack(a.x);
        HANDLES_VB.PushBack(a.y);
        HANDLES_VB.PushBack(a.z);
        HANDLES_VB.PushBack(idrgb.x);
        HANDLES_VB.PushBack(idrgb.y);
        HANDLES_VB.PushBack(idrgb.z);

        HANDLES_VB.PushBack(b.x);
        HANDLES_VB.PushBack(b.y);
        HANDLES_VB.PushBack(b.z);
        HANDLES_VB.PushBack(idrgb.x);
        HANDLES_VB.PushBack(idrgb.y);
        HANDLES_VB.PushBack(idrgb.z);

        HANDLES_VB.PushBack(c.x);
        HANDLES_VB.PushBack(c.y);
        HANDLES_VB.PushBack(c.z);
        HANDLES_VB.PushBack(idrgb.x);
        HANDLES_VB.PushBack(idrgb.y);
        HANDLES_VB.PushBack(idrgb.z);
    }
}

// draw all handles, return id of clicked handle
u32 FlushHandles(ivec2 clickat, const GPUFrameBuffer activeSceneTarget,
                 const mat4& activeViewMatrix, const mat4& activeProjectionMatrix, bool orthographic)
{
    if (HANDLES_VB.count == 0) return 0;

    const float sceneResolutionW = (float)activeSceneTarget.width;
    const float sceneResolutionH = (float)activeSceneTarget.height;
    const float mousePickTargetW = (float)mousePickingRenderTarget.width;
    const float mousePickTargetH = (float)mousePickingRenderTarget.width;

    mat4 scaledDownFrustum = ScaleMatrix(sceneResolutionW/mousePickTargetW, sceneResolutionH/mousePickTargetH, 1.f) * activeProjectionMatrix;
    const float offsetX = ((float)clickat.x - sceneResolutionW*0.5f) / (mousePickTargetW * 0.5f);
    const float offsetY = ((float)clickat.y - sceneResolutionH*0.5f) / (mousePickTargetH * 0.5f);
    if (orthographic)
    {
        // NOTE(Kevin): untested! haven't tried orthographic picking yet! prob needs some debugging
        scaledDownFrustum[3][0] += -offsetX;
        scaledDownFrustum[3][1] -= -offsetY;
    }
    else
    {
        // then the offset moves it around like light coming through a moving hole on the near plane
        scaledDownFrustum[2][0] += offsetX;
        scaledDownFrustum[2][1] -= offsetY;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, mousePickingRenderTarget.fbo);
    glViewport(0,0,(int)mousePickTargetW,(int)mousePickTargetH);
    glClearColor(0.f,0.f,0.f,1.f); // clear with rgb(id 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    UseShader(HANDLES_SHADER);

    GLBindMatrix4fv(HANDLES_SHADER, "projectionMatrix", 1, scaledDownFrustum.ptr());
    GLBindMatrix4fv(HANDLES_SHADER, "viewMatrix", 1, activeViewMatrix.ptr());

    glBindVertexArray(HANDLES_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, HANDLES_VBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float)*HANDLES_VB.count, HANDLES_VB.data, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, HANDLES_VB.count / 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    HANDLES_VB.ResetCount();

    glFlush();
    glFinish();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    u8 pickedRGBData[4];
    glReadPixels((int)mousePickTargetW/2, (int)mousePickTargetH/2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pickedRGBData);
    u32 pickedHandleId = pickedRGBData[0] + pickedRGBData[1] * 256 + pickedRGBData[2] * 256 * 256;
    return pickedHandleId;
}

#pragma endregion



#pragma region MATH_HELPERS

// only check in direction from
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








