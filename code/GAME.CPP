/*

TODO:
- Edge & Face select & move 
    - SELECT THE SELECTABLE VERTICES SINCE THAT EXISTS probably make SELECTED_SELECTABLE_VERTICES_INDICES a list of vertex ptrs
    - Face select and choose texture
    - CREATE GRID MESH STATICALLY AT THE START!!! DONT NEED TO MAKE IT EVERY FRAME (just one grid mesh, scale it and translate it)

basic layout and structures of maps can be created, they are textured
    - handles (have render pick ID, reference a Vert, or Edge, or Face?)
Save/load maps into fps game
can navigate map with collisions
enemy billboards two behaviour:
    - walk towards player
    - stop and shoot after split second to shoot projectile at player, projectile hurts player health
player holds 3D gun model, shoot hitscan at enemies, hitscan hurts enemy health
swap to a second 3D gun model, shoot shotgun hitscan at enemies
swap to a third 3D gun model, shoot hitscan with different stats at enemies


Finish implementing Editable Volumes and Atomics
(cube draw) Super simple create volume tool where you just drag mouse once? and then click once and raise mouse for height?
(poly draw) Draw on plane then raise to create volume tool
Move volume tool (maybe this is select all vertices and move?)
Multiselect vertices and move them
Multiselect edges and move them
Multiselect faces and move them
Multiselect volumes and edit them all at the same time


What else can be stripped out reused?
    Console front and backend

    use flat faces for collision? instead of meshes? GJK EPA code accepts 2D meshes


Priorty #1 is building the game, not the engine/tech

*/



#include <cstdint>
#include <cassert>
#include <fstream>
#include <string>
#include <chrono>
#include <iterator>

#include "BUILDINFO.H"

#if MESA_WINDOWS
#include <windows.h>
#include <dwmapi.h>
#endif

#if MESA_WINDOWS
    #define GL3W_IMPLEMENTATION
    #include <gl3w.h>
    #define MESA_USING_GL3W
#elif MESA_MACOSX
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
#endif

#include <SDL.h>
#include <SDL_mixer.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_SPRINTF_IMPLEMENTATION
#include <stb_sprintf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#define VERTEXT_IMPLEMENTATION
#include <vertext.h>
#include <gmath.h>

#include <unordered_map>
#include <array>
#include <stack>




typedef uint8_t       u8;
typedef uint16_t      u16;
typedef uint32_t      u32;
typedef uint64_t      u64;
typedef int8_t        i8;
typedef int16_t       i16;
typedef int32_t       i32;
typedef int64_t       i64;
typedef uint_fast8_t  u8f;
typedef uint_fast16_t u16f;
typedef uint_fast32_t u32f;
typedef int_fast8_t   i8f;
typedef int_fast16_t  i16f;
typedef int_fast32_t  i32f;
typedef i16           bool16;
typedef i32           bool32;

#if (defined _MSC_VER)
#define ASSERT(predicate) if(!(predicate)) { __debugbreak(); }
#else
#define ASSERT(predicate) if(!(predicate)) { __builtin_trap(); }
#endif

inline std::string wd_path() { return std::string(PROJECT_WORKING_DIR); }
inline std::string wd_path(const std::string& name) { return wd_path() + std::string(name); }
inline std::string data_path() { return wd_path() + "data/"; }
inline std::string data_path(const std::string& name) { return wd_path() + "data/" + name; }

#define ARRAY_COUNT(a) (sizeof(a) / (sizeof(a[0])))

// fflush this shit for CLion
#define LogMessage(...)                        \
    do {                                       \
        fprintf(stdout, __VA_ARGS__);          \
        fprintf(stdout, "\n");                 \
        fflush(stdout);                        \
    } while (false)
#define LogWarning(...)                        \
    do {                                       \
        fprintf(stderr, __VA_ARGS__);          \
        fprintf(stderr, "\n");                 \
    } while (false)
#define LogError(...)                          \
    do {                                       \
        fprintf(stderr, __VA_ARGS__);          \
        fprintf(stderr, "\n");                 \
    } while (false)



#pragma region SUPER_GLOBAL_VARIABLES

// let 1 unit = 1 inch, this approximates 32 units to 0.82 metres
#define STANDARD_LENGTH_IN_GAME_UNITS 32
// sqrt(32^2 + 32^2) = 45.254833996 ~= 45
#define STANDARD_LENGTH_DIAGONAL 45
#define THIRTYTWO STANDARD_LENGTH_IN_GAME_UNITS

SDL_Window *g_MainSDLWindow;
SDL_GLContext g_SDLGLContext;
bool g_ProgramShouldShutdown = false;
float g_DeltaTime = 0.f;
float g_UnscaledDeltaTime = 0.f;
float g_GameTimeScale = 1.f;
float g_CurrentTime = 0.f;
float g_TimeSinceStart = 0.f;
u32 mousecurrent;
u32 mousepressed;
u32 mousereleased;
vec2 mousedelta;
ivec2 mousepos;
u8 keyscurrent[256] = {0};
u8 keyspressed[256] = {0};
u8 keysreleased[256] = {0};

i32 BACKBUFFER_WIDTH = -1;
i32 BACKBUFFER_HEIGHT = -1;


#pragma endregion

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



struct BinaryFileHandle
{
    /** Handle for a file in memory */
    u32     size   = 0;        // size of file in memory
    void*   memory = nullptr;  // pointer to file in memory
};

struct BitmapHandle : BinaryFileHandle
{
    /** Handle for an UNSIGNED BYTE bitmap in memory */
    u32 width    = 0;   // image width
    u32 height   = 0;   // image height
    u8  bitDepth = 0;   // bit depth of bitmap in bytes (e.g. bit depth = 3 means there are 3 bytes in the bitmap per pixel)
};

void FreeFileBinary(BinaryFileHandle& binary_file_to_free)
{
    if (binary_file_to_free.memory)
    {
        free(binary_file_to_free.memory);
        binary_file_to_free.memory = nullptr;
        binary_file_to_free.size = 0;
    }
}

/** Allocates memory, stores the binary file data in memory, makes binary_file_handle_t.memory
    point to it. Pass along a binary_file_handle_t to receive the pointer to the file data in
    memory and the size in bytes. */
void ReadFileBinary(BinaryFileHandle& mem_to_read_to, const char* file_path)
{
    if(mem_to_read_to.memory)
    {
        printf("WARNING: Binary File Handle already points to allocated memory. Freeing memory first...\n");
        FreeFileBinary(mem_to_read_to);
    }

    SDL_RWops* binary_file_rw = SDL_RWFromFile(file_path, "rb");
    if(binary_file_rw)
    {
        mem_to_read_to.size = (u32) SDL_RWsize(binary_file_rw); // total size in bytes
        mem_to_read_to.memory = malloc((size_t) mem_to_read_to.size);
        SDL_RWread(binary_file_rw, mem_to_read_to.memory, (size_t) mem_to_read_to.size, 1);
        SDL_RWclose(binary_file_rw);
    }
    else
    {
        printf("Failed to read %s! File doesn't exist.\n", file_path);
        return;
    }
}

bool WriteFileBinary(const BinaryFileHandle& bin, const char* file_path)
{
    if (bin.memory == NULL)
    {
        printf("WARNING: Binary File Handle does not point to any memory. Cancelled write to file operation.\n");
        return false;
    }

    SDL_RWops* bin_w = SDL_RWFromFile(file_path, "wb");
    if(bin_w)
    {
        SDL_RWwrite(bin_w, bin.memory, bin.size, 1);
        SDL_RWclose(bin_w);
        return true;
    }

    return false;
}

/** Returns the string content of a file as an std::string */
std::string ReadFileString(const char* file_path)
{
    std::string string_content;

    std::ifstream file_stream(file_path, std::ios::in);
    if (file_stream.is_open() == false)
    {
        printf("Failed to read %s! File doesn't exist.\n", file_path);
    }

    std::string line = "";
    while (file_stream.eof() == false)
    {
        std::getline(file_stream, line);
        string_content.append(line + "\n");
    }

    file_stream.close();

    return string_content;
}

void FreeImage(BitmapHandle& image_handle)
{
    FreeFileBinary(image_handle);
    image_handle.width = 0;
    image_handle.height = 0;
    image_handle.bitDepth = 0;
}

/** Allocates memory, loads an image file as an UNSIGNED BYTE bitmap, makes bitmap_handle_t.memory
    point to it. Pass along a bitmap_handle_t to receive the pointer to the bitmap in memory and
    bitmap information. */
void ReadImage(BitmapHandle& image_handle, const char* image_file_path)
{
    if(image_handle.memory)
    {
        printf("WARNING: Binary File Handle already points to allocated memory. Freeing memory first...\n");
        FreeImage(image_handle);
    }

    stbi_set_flip_vertically_on_load(1);
    image_handle.memory = stbi_load(image_file_path, (int*)&image_handle.width, (int*)&image_handle.height, (int*)&image_handle.bitDepth, 0);
    if(image_handle.memory)
    {
        image_handle.size = image_handle.width * image_handle.height * image_handle.bitDepth;
    }
    else
    {
        printf("Failed to find image file at: %s\n", image_file_path);
        image_handle.width = 0;
        image_handle.height = 0;
        image_handle.bitDepth = 0;
        return;
    }
}



struct GPUFrameBuffer
{
    u32 fbo;
    u32 colorTexId;
    u32 depthTexId;
    i32 width;
    i32 height;
};

struct GPUMesh
{
    u32 idVAO = 0;
    u32 idVBO = 0;
    u32 idIBO = 0;
    u32 indicesCount = 0;
};

struct GPUTexture
{
    GLuint id = 0; // ID for the texture in GPU memory
    i32 width = 0;
    i32 height = 0;
    GLenum  format = GL_NONE; // format / bitdepth of texture (GL_RGB would be 3 byte bit depth)
};

bool GLHasErrors();

static void CreateGPUFrameBuffer(GPUFrameBuffer *buffer)
{
    buffer->fbo = 0;

    glGenTextures(1, &buffer->colorTexId);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, buffer->width, buffer->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenTextures(1, &buffer->depthTexId);
    glBindTexture(GL_TEXTURE_2D, buffer->depthTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, buffer->width, buffer->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &buffer->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, buffer->fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer->colorTexId, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, buffer->depthTexId, 0);

    ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void UpdateGPUFrameBufferSize(GPUFrameBuffer *buffer, i32 w, i32 h)
{
    if (buffer->width == w && buffer->height == h) return;

    buffer->width = w;
    buffer->height = h;
    glBindFramebuffer(GL_FRAMEBUFFER, buffer->fbo);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, buffer->width, buffer->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, buffer->depthTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, buffer->width, buffer->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        fprintf(stderr, "Failed to change size of Internal FrameBuffer Object.");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void RenderGPUMesh(GPUMesh mesh, GLenum rendermode = GL_TRIANGLES)
{
    if (mesh.indicesCount == 0) // Early out if index_count == 0, nothing to draw
    {
        printf("WARNING: Attempting to Render a mesh with 0 index count!\n");
        return;
    }

    // Bind VAO, bind VBO, draw elements(indexed draw)
    glBindVertexArray(mesh.idVAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.idIBO);
    glDrawElements(rendermode, mesh.indicesCount, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

static void RebindGPUMeshData(GPUMesh *mesh, 
                              float *vertices, 
                              u32 *indices, u32 
                              verticesArrayCount, 
                              u32 indicesArrayCount, 
                              GLenum drawUsage = GL_DYNAMIC_DRAW)
{
    if (mesh->idVBO == 0 || mesh->idIBO == 0)
        return;

    mesh->indicesCount = indicesArrayCount;
    glBindVertexArray(mesh->idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->idVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) 4 * verticesArrayCount, vertices, drawUsage);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->idIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) 4 * indicesArrayCount, indices, drawUsage);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

static void CreateGPUMesh(GPUMesh *mesh, 
                          float *vertices, 
                          u32 *indices, 
                          u32 verticesArrayCount, 
                          u32 indicesArrayCount, 
                          u8 positionAttribSize = 3, 
                          u8 textureAttribSize = 2, 
                          u8 normalAttribSize = 3, 
                          GLenum drawUsage = GL_DYNAMIC_DRAW)
{
    ASSERT(mesh->idVAO == 0);

    u8 stride = 0;
    if (textureAttribSize)
    {
        stride += positionAttribSize + textureAttribSize;
        if (normalAttribSize)
        {
            stride += normalAttribSize;
        }
    }

    mesh->indicesCount = indicesArrayCount;

    glGenVertexArrays(1, &mesh->idVAO);
    glBindVertexArray(mesh->idVAO);
    glGenBuffers(1, &mesh->idVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->idVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) 4 /*bytes cuz float*/ * verticesArrayCount, vertices, drawUsage);
    glVertexAttribPointer(0, positionAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride, nullptr);
    glEnableVertexAttribArray(0);
    if (textureAttribSize > 0)
    {
        glVertexAttribPointer(1, textureAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride,
                              (void*)(sizeof(float) * positionAttribSize));
        glEnableVertexAttribArray(1);
        if (normalAttribSize > 0)
        {
            glVertexAttribPointer(2, normalAttribSize, GL_FLOAT, GL_FALSE, sizeof(float) * stride,
                                  (void*)(sizeof(float) * ((GLsizeiptr) positionAttribSize + textureAttribSize)));
            glEnableVertexAttribArray(2);
        }
    }

    glGenBuffers(1, &mesh->idIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->idIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) 4 /*bytes cuz uint32*/ * indicesArrayCount, indices, drawUsage);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0); // Unbind the VAO;
}

static void DeleteGPUMesh(GPUMesh *mesh)
{
    if (mesh->idIBO != 0)
    {
        glDeleteBuffers(1, &mesh->idIBO);
        mesh->idIBO = 0;
    }
    if (mesh->idVBO != 0)
    {
        glDeleteBuffers(1, &mesh->idVBO);
        mesh->idVBO = 0;
    }
    if (mesh->idVAO != 0)
    {
        glDeleteVertexArrays(1, &mesh->idVAO);
        mesh->idVAO = 0;
    }

    mesh->indicesCount = 0;
}

static void CreateGPUTextureFromBitmap(GPUTexture        *texture,
                                       unsigned char     *bitmap,
                                       u32               bitmap_width,
                                       u32               bitmap_height,
                                       GLenum            target_format,
                                       GLenum            source_format,
                                       GLenum            min_filter_mode = GL_NEAREST,
                                       GLenum            mag_filter_mode = GL_NEAREST)
{
    if (bitmap == NULL)
    {
        LogError("CreateGPUTextureFromBitmap error: provided bitmap is null.");
        return;
    }

    ASSERT(texture->id == 0);

    texture->width = bitmap_width;
    texture->height = bitmap_height;
    texture->format = source_format;

    glGenTextures(1, &texture->id);   // generate texture and grab texture id
    glBindTexture(GL_TEXTURE_2D, texture->id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter_mode); // filtering (e.g. GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter_mode);
    glTexImage2D(
            GL_TEXTURE_2D,            // texture target type
            0,                        // level-of-detail number n = n-th mipmap reduction image
            target_format,            // format of data to store (target): num of color components
            bitmap_width,             // texture width
            bitmap_height,            // texture height
            0,                        // must be 0 (legacy)
            source_format,            // format of data being loaded (source)
            GL_UNSIGNED_BYTE,         // data type of the texture data
            bitmap);                  // data
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void CreateGPUTextureFromDisk(GPUTexture *texture, const char* filePath, GLenum targetFormat = GL_RGBA)
{
    BitmapHandle textureBitmapHandle;
    ReadImage(textureBitmapHandle, filePath);
    if (textureBitmapHandle.memory == nullptr)
        return;

    CreateGPUTextureFromBitmap(texture,
                               (unsigned char*) textureBitmapHandle.memory,
                               textureBitmapHandle.width,textureBitmapHandle.height,
                               targetFormat,
                               (textureBitmapHandle.bitDepth == 3 ? GL_RGB : GL_RGBA));
    GLHasErrors();
    FreeImage(textureBitmapHandle); // texture data has been copied to GPU memory, so we can free image from memory
}

static void UpdateGPUTextureFromBitmap(GPUTexture *texture, unsigned char *bitmap, i32 w, i32 h)
{
    ASSERT(texture->id != 0);

    texture->width = w;
    texture->height = h;

    glBindTexture(GL_TEXTURE_2D, texture->id);
    glTexImage2D(
        GL_TEXTURE_2D,            // texture target type
        0,                        // level-of-detail number n = n-th mipmap reduction image
        texture->format,          // format of data to store (target): num of color components
        w,                        // texture width
        h,                        // texture height
        0,                        // must be 0 (legacy)
        texture->format,          // format of data being loaded (source)
        GL_UNSIGNED_BYTE,         // data type of the texture data
        bitmap);                  // data
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void DeleteGPUTexture(GPUTexture *texture)
{
    if (texture->id == 0)
        return;

    glDeleteTextures(1, &texture->id);

    texture->id = 0;
    texture->width = 0;
    texture->height = 0;
    texture->format = GL_NONE;
}

static void GLActiveTextureUnit0(GLuint textureId)
{
    // if (textureId == 0)
    // {
    //     glActiveTexture(GL_TEXTURE0);
    //     glBindTexture(GL_TEXTURE_2D, Resources.defaultMissingTexture.textureId);
    // }
    // else
    // {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureId);
    // }
}



struct MeshGroup
{
    std::vector<GPUMesh>     meshes;
    std::vector<GPUTexture>  textures;
    std::vector<u16>         meshToTextureMap; // stores index of textures element corresponding to mesh
    std::string              filePath = "";
};

static void MeshGroupRender(MeshGroup *meshgroup)
{
    for (size_t i = 0; i < meshgroup->meshes.size(); ++i)
    {
        u16 materialIndex = meshgroup->meshToTextureMap[i];
        if (materialIndex < meshgroup->textures.size())
        {
            // GLActiveTextureUnit0(meshgroup->textures[materialIndex].id);
        }
        RenderGPUMesh(meshgroup->meshes[i]);
    }
}

static GPUMesh __assimpLoadMeshHelper(aiMesh* meshNode)
{
    const u8 vb_entries_per_vertex = 8;
    std::vector<float> vb(meshNode->mNumVertices * vb_entries_per_vertex);
    std::vector<u32> ib(meshNode->mNumFaces * meshNode->mFaces[0].mNumIndices);
    if (meshNode->mTextureCoords[0])
    {
        for (size_t i = 0; i < meshNode->mNumVertices; ++i)
        {
            // mNormals and mVertices are both mNumVertices in size
            size_t v_start_index = i * vb_entries_per_vertex;
            vb[v_start_index] = meshNode->mVertices[i].x;
            vb[v_start_index + 1] = meshNode->mVertices[i].y;
            vb[v_start_index + 2] = meshNode->mVertices[i].z;
            vb[v_start_index + 3] = meshNode->mTextureCoords[0][i].x;
            vb[v_start_index + 4] = meshNode->mTextureCoords[0][i].y;
            vb[v_start_index + 5] = meshNode->mNormals[i].x;
            vb[v_start_index + 6] = meshNode->mNormals[i].y;
            vb[v_start_index + 7] = meshNode->mNormals[i].z;
        }
    }
    else
    {
        for (size_t i = 0; i < meshNode->mNumVertices; ++i)
        {
            size_t v_start_index = i * vb_entries_per_vertex;
            vb[v_start_index] = meshNode->mVertices[i].x;
            vb[v_start_index + 1] = meshNode->mVertices[i].y;
            vb[v_start_index + 2] = meshNode->mVertices[i].z;
            vb[v_start_index + 3] = 0.f;
            vb[v_start_index + 4] = 0.f;
            vb[v_start_index + 5] = meshNode->mNormals[i].x;
            vb[v_start_index + 6] = meshNode->mNormals[i].y;
            vb[v_start_index + 7] = meshNode->mNormals[i].z;
        }
    }

    for (size_t i = 0; i < meshNode->mNumFaces; ++i)
    {
        aiFace face = meshNode->mFaces[i];
        for (size_t j = 0; j < face.mNumIndices; ++j)
        {
            ib[i * face.mNumIndices + j] = face.mIndices[j]; // prob sometimes not correct to index ib this way
        }
    }

    GPUMesh mesh;
    CreateGPUMesh(&mesh, &vb[0], &ib[0], (u32)vb.size(), (u32)ib.size());
    return mesh;
}

static MeshGroup MeshGroupLoad(const char* fileName)
{
    Assimp::Importer importer;
    /*  NOTE: To create smooth normals respecting edges sharper than a given angle,
        use importer.SetPropertyFloat("PP_GSN_MAX_SMOOTHING_ANGLE", 90) along with
        aiProcess_GenSmoothNormals flag. https://github.com/assimp/assimp/issues/1713

        aiProcess_GenSmoothNormals
        This flag may not be specified together with #aiProcess_GenNormals. There's
        a importer property, #AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE which allows you to
        specify an angle maximum for the normal smoothing algorithm. Normals exceeding
        this limit are not smoothed, resulting in a 'hard' seam between two faces.
        Using a decent angle here (e.g. 80 degrees) results in very good visual
        appearance. To create smooth normals respecting edges sharper than a given angle,
        use importer.SetPropertyFloat("PP_GSN_MAX_SMOOTHING_ANGLE", 90) along with
        aiProcess_GenSmoothNormals flag. https://github.com/assimp/assimp/issues/1713

        aiProcess_JoinIdenticalVertices
        is compulsory for indexed drawing. This still works with flat shaded geometry
        because it only joins vertices that are identical in both position and normal.
        e.g. a flat shaded cube will have 24 vertices after joining because each side
        of the cube will have 4 unique vertices and the vertices at the corners will
        not be shared by multiple faces of the cube because they will have different
        normals even though their positions are the same.
    */
    MeshGroup meshGroupToLoadTo;

    const aiScene* scene = importer.ReadFile(fileName,
        aiProcess_Triangulate
        | aiProcess_GenNormals
        | aiProcess_FlipUVs // textures were fucked cuz this line was missing
        | aiProcess_JoinIdenticalVertices
    );
    if (!scene)
    {
        printf("Model '%s' failed to load: %s\n", fileName, importer.GetErrorString());
        return meshGroupToLoadTo;
    }


    meshGroupToLoadTo.meshes = std::vector<GPUMesh>(scene->mNumMeshes);
    meshGroupToLoadTo.textures = std::vector<GPUTexture>(scene->mNumMaterials);
    meshGroupToLoadTo.meshToTextureMap = std::vector<u16>(scene->mNumMeshes);

    // Unpack meshes
    for (size_t i = 0; i < scene->mNumMeshes; ++i)
    {
        aiMesh* mesh_node = scene->mMeshes[i];
        meshGroupToLoadTo.meshes[i] = __assimpLoadMeshHelper(mesh_node);
        meshGroupToLoadTo.meshToTextureMap[i] = mesh_node->mMaterialIndex;
    }

    // Load diffuse textures
    for (size_t i = 0; i < scene->mNumMaterials; ++i)
    {
        aiMaterial* mat = scene->mMaterials[i];
        if (mat->GetTextureCount(aiTextureType_DIFFUSE))
        {
            aiString path;
            if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
            {
                int idx = (int)std::string(path.data).find_last_of("\\");
                std::string texture_file_name = std::string(path.data).substr(idx + 1);
                std::string model_file_directory = std::string(fileName);
                idx = GM_max((int)model_file_directory.find_last_of("/"), (int)model_file_directory.find_last_of("\\"));
                model_file_directory = model_file_directory.substr(0, idx + 1);

                std::string tex_path = model_file_directory + texture_file_name;
                CreateGPUTextureFromDisk(&meshGroupToLoadTo.textures[i], tex_path.c_str());
            }
        }
    }

    return meshGroupToLoadTo;
}

static void MeshGroupDelete(MeshGroup *modelToDelete)
{
    for (auto& mesh : modelToDelete->meshes)
    {
        DeleteGPUMesh(&mesh);
    }
    modelToDelete->meshes.clear();
    for (auto& texture : modelToDelete->textures)
    {
        DeleteGPUTexture(&texture);
    }
    modelToDelete->textures.clear();
}


struct FaceBatch
{
    /** Set of triangles that all use the same texture. The position values 
        of each triangle vertex are defined in world space. 

        data layout:
            position x y z, texture u v, normal i j k, ...
    */

    GPUTexture sharedTexture;
    u32 idVAO = 0;
    u32 idVBO = 0;
    u32 verticesCount = 0;
};

static void CreateFaceBatchMeshOnGPU(u32 *idVAO, u32 *idVBO)
{
    glGenVertexArrays(1, idVAO);
    glBindVertexArray(*idVAO);
    glGenBuffers(1, idVBO);
    glBindBuffer(GL_ARRAY_BUFFER, *idVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * (3 + 2)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

static void DeleteFaceBatchMeshOnGPU(u32 idVAO, u32 idVBO)
{
    glDeleteBuffers(1, &idVBO);
    glDeleteVertexArrays(1, &idVAO);
}

static void RebindFaceBatchBufferObject(FaceBatch *faceBatch, u32 sizeInBytes, float *data)
{
    glBindVertexArray(faceBatch->idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, faceBatch->idVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeInBytes, data, GL_DYNAMIC_DRAW);
    faceBatch->verticesCount = sizeInBytes / (sizeof(float) * 8);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    // NOTE(Kevin): What even is the point of unbinding...I'm going to be careful about always binding before doing anything with VAOs anyways...
}

static void RenderFaceBatch(FaceBatch faceBatch)
{
    glBindVertexArray(faceBatch.idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, faceBatch.idVBO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, faceBatch.sharedTexture.id);
    glDrawArrays(GL_TRIANGLES, 0, faceBatch.verticesCount);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

#include "SHADERS.CPP"
#include "UTILITY.CPP"

#pragma region MORE_GLOBAL_VARIABLES
GPUShader sceneShader;
GPUShader editorShader_FaceSelected;
GPUShader finalPassShader;
GPUFrameBuffer renderTargetGame;
GPUFrameBuffer renderTargetGUI;
GPUMesh __final_render_output_quad;
float GAMEPROJECTION_NEARCLIP = 4.f; // even 2 works fine to remove z fighting
float GAMEPROJECTION_FARCLIP = 32000.f;
bool primitivesDepthTest = false;
#pragma endregion

#include "GUI.CPP"


///////////////////////////////
// GJK + EPA
///////////////////////////////


struct Collider
{
    /* Returns the furthest point on the collider along a given direction */
    virtual vec3 FindFurthestPoint(vec3 direction) const = 0;
};

struct MeshCollider : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
        vec3 maxVertexSoFar;
        float maxDistanceSoFar = -FLT_MAX;
        for (auto pointWCS : pointCloud)
        {
            float distance = Dot(pointWCS, direction);
            if (distance > maxDistanceSoFar)
            {
                maxVertexSoFar = pointWCS;
                maxDistanceSoFar = distance;
            }
        }
        return maxVertexSoFar;
    }

    std::vector<vec3> pointCloud;
};

struct SphereCollider : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
        vec3 offset = Normalize(direction) * radius;
        return position + offset;
    }

    vec3 position;
    float radius;
};

struct LineCollider : Collider
{
    vec3 FindFurthestPoint(vec3 direction) const override
    {
        float distanceA = Dot(a, direction);
        float distanceB = Dot(b, direction);
        return (distanceA > distanceB ? a : b);
    }

    vec3 a;
    vec3 b;
};


struct GJKSimplex
{
private:
    std::array<vec3, 4> points = {};
    u8 count = 0;

public:
    GJKSimplex& operator=(std::initializer_list<vec3> list)
    {
        count = 0;
        for (auto v : list)
        {
            points[count] = v;
            ++count;
        }
        return *this;
    }

    vec3& operator[](int i)
    {
        return points[i];
    }

    void pushFront(vec3 point)
    {
        points = { point, points[0], points[1], points[2] };
        count = GM_min(count + 1, 4u);
    }

    u8 size() const
    {
        return count;
    }
};

static vec3 GJKSupport(const Collider *colA, const Collider *colB, const vec3& direction)
{
    // https://en.wikipedia.org/wiki/Support_function
    return colA->FindFurthestPoint(direction) - colB->FindFurthestPoint(-direction);
}

#define GJKSameDirection(dir, ao) (Dot(dir, ao) > 0.f)

static bool GJKLine(GJKSimplex& capture, vec3& direction)
{
    vec3 a = capture[0];
    vec3 b = capture[1];
    vec3 ab = b - a;
    // Note that vec3 ao (a to origin) is just -a

    if (GJKSameDirection(ab, -a))
    {
        // NOTE(Kevin): If initial search direction aligns with the collision vector between two spheres, then it doesn't work.
        direction = Cross(Cross(ab, -a), ab);
    }
    else
    {
        capture = { a };
        direction = -a;
        // THIS PATH WILL SIGNAL THAT THERE IS NO COLLISION, but it might waste a support call
        // Is there a way to stop the algorithm here?
        // if origin is past point A, then it's impossible to be included in the simplex.
        // I need to figure out how to stop the algorithm here and return that they are not intersecting.
    }

    return false;
}

static bool GJKTriangle(GJKSimplex& capture, vec3& direction)
{
    vec3 a = capture[0];
    vec3 b = capture[1];
    vec3 c = capture[2];
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 triangleNormal = Cross(ab, ac);

    if (GJKSameDirection(Cross(triangleNormal, ac), -a))
    {
        if (GJKSameDirection(ac, -a))
        {
            capture = { a, c };
            direction = Cross(Cross(ac, -a), ac);
        }
        else
        {
            capture = { a };
            direction = -a;
        }
    }
    else
    {
        if (GJKSameDirection(Cross(ab, triangleNormal), -a))
        {
            if (GJKSameDirection(ab, -a))
            {
                capture = { a, b };
                direction = Cross(Cross(ab, -a), ab);
            }
            else
            {
                capture = { a };
                direction = -a;
            }
        }
        else
        {
            // The origin must be above or below this triangle
            if (GJKSameDirection(triangleNormal, -a))
            {
                capture = { a, b, c };
                direction = triangleNormal;
            }
            else
            {
                capture = { a, c, b };
                direction = -triangleNormal;
            }
        }
    }

    return false;
}

static bool GJKTetrahedron(GJKSimplex& capture, vec3& direction)
{
    vec3 a = capture[0];
    vec3 b = capture[1];
    vec3 c = capture[2];
    vec3 d = capture[3];
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ad = d - a;
    vec3 abcNormal = Cross(ab, ac);
    vec3 acdNormal = Cross(ac, ad);
    vec3 adbNormal = Cross(ad, ab);

    /*
     * ORDER NEEDS TO BE ABC first, ACD second, ADB third OR IT MIGHT INFINITELY
     * LOOP THE TETRAHEDRON AND TRIANGLE CASES BUT I DON'T UNDERSTAND WHY
     * */

    if (GJKSameDirection(abcNormal, -a))
    {
        capture = { a, b, c };
        return GJKTriangle(capture, direction);
    }
    if (GJKSameDirection(acdNormal, -a))
    {
        capture = { a, c, d };
        return GJKTriangle(capture, direction);
    }
    if (GJKSameDirection(adbNormal, -a))
    {
        capture = { a, d, b };
        return GJKTriangle(capture, direction);
    }

    // IN THIS CASE: The origin is not in the direction of the normals of the three triangles formed with a.
    // AND we know that the origin is not "below" the triangle (if bcd forms the base) because we just added
    // the point a i.e. the origin is in direction of direction given to this function.
    // THEREFORE, the origin MUST be inside the tetrahedron
    return true;
}

static bool GJKNextSimplex(GJKSimplex& capture, vec3& direction)
{
    /* We need a series of different checks for each shape of simplex
     * to see what the new simplex should be and what direction we'll
     * search in next */
    switch (capture.size()) {
    case 2: return GJKLine(capture, direction);
    case 3: return GJKTriangle(capture, direction);
    case 4: return GJKTetrahedron(capture, direction);
    }

    ASSERT(false) // should never happen
        return false;
}

struct GJKEPACollisionInfo
{
    vec3 normal;
    float penetrationDepth = 0.f;
    bool hasCollision = false;
};

static std::pair<std::vector<vec4>, int> EPAGetFaceNormals(
    const std::vector<vec3>& polytope,
    const std::vector<int>& faces)
{
    /* Returns the normal of every face in the polytope, their distances
     * from the origin, and the index of the face closest to the origin. */

    std::vector<vec4> normals;
    int minTriangle = 0;
    float minDistance = FLT_MAX;

    for (int i = 0; i < faces.size(); i += 3)
    {
        vec3 a = polytope[faces[i]];
        vec3 b = polytope[faces[(size_t) i + 1]];
        vec3 c = polytope[faces[(size_t) i + 2]];
        vec3 normal = Normalize(Cross((b - a), (c - a)));
        float distance = Dot(normal, a); // the distance is the component of vector a projected onto the normal

        if (distance < 0.f)
        {
            normal *= -1.f;
            distance *= -1.f;
        }

        normals.emplace_back(normal, distance);

        if (distance < minDistance)
        {
            minDistance = distance;
            minTriangle = i / 3;
        }
    }

    return { normals, minTriangle };
}

static void EPAAddIfUniqueEdge(
    std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& faces,
    int a,
    int b)
{
    auto reverse = std::find(edges.begin(), edges.end(),
        std::make_pair(faces[b], faces[a]));
    if (reverse != edges.end())
    {
        edges.erase(reverse);
    }
    else
    {
        edges.emplace_back(faces[a], faces[b]);
    }
}

#ifndef EPA_MAX_ITERATIONS
#define EPA_MAX_ITERATIONS 10
#endif
static GJKEPACollisionInfo EPA(GJKSimplex& simplex, const Collider* colA, const Collider* colB)
{
    std::vector<vec3> polytope;
    ASSERT(simplex.size() == 4)
        for (int i = 0; i < simplex.size(); ++i)
        {
            polytope.push_back(simplex[i]);
        }
    std::vector<int> faces = {
            0, 1, 2,
            0, 3, 1,
            0, 2, 3,
            1, 3, 2
    };

    auto normalsResult = EPAGetFaceNormals(polytope, faces);
    auto normals = normalsResult.first;
    auto minFace = normalsResult.second;

    vec3 minNormal;
    float minDistance = FLT_MAX;

    u8 epaIterations = 0; // the EPA loop can be infinite if we have curved edges - just return whatever it has after X iterations.
    while (minDistance == FLT_MAX)
    {
        minNormal = normals[minFace].xyz;
        minDistance = normals[minFace].w;

        if (epaIterations++ > EPA_MAX_ITERATIONS)
        {
            break;
        }

        vec3 support = GJKSupport(colA, colB, minNormal); // i.e. is there a point to be added in this direction?
        float sDistance = Dot(minNormal, support);

        if (GM_abs(sDistance - minDistance) > 0.001f)
        {
            // then we must expand the polytope
            std::vector<std::pair<int, int>> uniqueEdges;
            for (int i = 0; i < normals.size(); ++i)
            {
                if (GJKSameDirection(normals[i].xyz, support))
                {
                    // for each face, if the face is in the same direction as support, remove
                    int f = i * 3;
                    EPAAddIfUniqueEdge(uniqueEdges, faces, f, f + 1);
                    EPAAddIfUniqueEdge(uniqueEdges, faces, f + 1, f + 2);
                    EPAAddIfUniqueEdge(uniqueEdges, faces, f + 2, f);
                    // reconstruction
                    faces[(size_t) f + 2] = faces.back();
                    faces.pop_back();
                    faces[(size_t) f + 1] = faces.back();
                    faces.pop_back();
                    faces[f] = faces.back();
                    faces.pop_back();
                    normals[i] = normals.back();
                    normals.pop_back();
                    --i;
                }
            }

            if (uniqueEdges.size() == 0)
            {
                minNormal = vec3(0.f, 0.f, 0.f);
                minDistance = 0.f;
                break;
            }

            std::vector<int> newFaces;
            for (auto edge : uniqueEdges)
            {
                newFaces.push_back(edge.first);
                newFaces.push_back(edge.second);
                newFaces.push_back((int)polytope.size());
            }
            polytope.push_back(support);

            normalsResult = EPAGetFaceNormals(polytope, newFaces);
            auto newNormals = normalsResult.first;
            auto newMinFace = normalsResult.second;

            float oldMinDistance = FLT_MAX;
            for (int i = 0; i < normals.size(); ++i)
            {
                if (normals[i].w < oldMinDistance)
                {
                    oldMinDistance = normals[i].w;
                    minFace = i;
                }
            }

            if (newNormals[newMinFace].w < oldMinDistance)
            {
                minFace = newMinFace + (int)normals.size();
            }

            faces.insert(faces.end(), newFaces.begin(), newFaces.end());
            normals.insert(normals.end(), newNormals.begin(), newNormals.end());

            minDistance = FLT_MAX;
        }
    }

    GJKEPACollisionInfo points;
    points.normal = minNormal;
    points.penetrationDepth = minDistance + 0.001f;
    points.hasCollision = true;
    return points;
}

// static GJKEPACollisionInfo CheckSphereCollision(SphereCollider* a, SphereCollider* b)
// {
//     vec3 aPos = a->GetPosition();
//     vec3 bPos = b->GetPosition();
//     float aRad = a->GetScaledRadius();
//     float bRad = b->GetScaledRadius();
//     float baLen = Magnitude(bPos - aPos);
//     if (baLen < aRad + bRad)
//     {
//         return { Normalize(bPos - aPos), aRad + bRad - baLen, true };
//     }
//     else
//     {
//         return { vec3(), 0.f, false };
//     }
// }

static GJKEPACollisionInfo GJK(Collider* colA, Collider* colB)
{
    // colA->TransformCollider();
    // colB->TransformCollider();

    // if (colA->colliderType == ColliderType::Sphere && colB->colliderType == ColliderType::Sphere)
    // {
    //     return CheckSphereCollision((SphereCollider*)colA, (SphereCollider*)colB);
    // }

    //NOTE(Kevin): start direction could be origin - (colA owning object position - colB owning object position)
    vec3 support = GJKSupport(colA, colB, { 1.f,1.f,1.f });
    GJKSimplex capture;
    capture.pushFront(support);
    vec3 direction = -support;

    for (;;)
    {
        support = GJKSupport(colA, colB, direction);
        if (Dot(support, direction) <= 0)
        {
            // previous support point was already the furthest along direction
            GJKEPACollisionInfo collisionInfo;
            collisionInfo.hasCollision = false;
            return collisionInfo;
        }

        capture.pushFront(support);
        if (GJKNextSimplex(capture, direction))
        {
            GJKEPACollisionInfo collisionInfo = EPA(capture, colA, colB);
            collisionInfo.hasCollision = true;
            return collisionInfo;
        }
    }
}

// MIXER
Mix_Chunk *Mixer_LoadChunk(const char *filepath)
{
    Mix_Chunk *chunk = Mix_LoadWAV(filepath);
    if (chunk == NULL)
        printf("Failed to load sound effect! SDL_mixer error: %s\n", Mix_GetError());
    return chunk;
}


GPUTexture tex_Default;
Mix_Chunk *sfx_Jump;

vec3 playerControllerRoot = vec3(100,10,100);
vec3 cameraRotation = vec3(0,130,-30);
vec3 cameraDirection;
vec3 cameraRight;
vec3 cameraUp;
mat4 activePerspectiveMatrix;
mat4 activeViewMatrix;
vec3 playerWalkDirectionForward;
vec3 playerWalkDirectionRight;
vec3 playerColliderPoints[] = {
    vec3(8,0,8), vec3(8,0,-8), vec3(-8,0,8), vec3(-8,0,-8),
    vec3(8,48,8), vec3(8,48,-8), vec3(-8,48,8), vec3(-8,48,-8)
};
MeshCollider playerCollider;
MeshCollider *PlayerColliderAtPos(vec3 playerWorldPosition)
{
    playerCollider.pointCloud.resize(8);
    for (int i = 0; i < 8; ++i)
        playerCollider.pointCloud[i] = playerWorldPosition + playerColliderPoints[i];
    return &playerCollider;
}
float playeryvel = 0.f;

vec3 bodyguardEnemyColliderPoints[] = {
    vec3(8,0,8), vec3(8,0,-8), vec3(-8,0,8), vec3(-8,0,-8),
    vec3(8,48,8), vec3(8,48,-8), vec3(-8,48,8), vec3(-8,48,-8)
};
MeshCollider bodyguardEnemyCollider;
MeshCollider *BodyguardEnemyColliderAtPos(vec3 worldPos)
{
    bodyguardEnemyCollider.pointCloud.resize(8);
    for (int i = 0; i < 8; ++i)
        bodyguardEnemyCollider.pointCloud[i] = worldPos + bodyguardEnemyColliderPoints[i];
    return &bodyguardEnemyCollider;
}

MeshGroup mg_Enemy_Bodyguard;

struct BodyguardEnemy
{
    vec3 root;
    vec3 facing;
};
BodyguardEnemy enemy0;


#include "EDITOR.CPP"


static void Stuff()
{
    SDL_SetRelativeMouseMode(SDL_TRUE);

    CreateGPUTextureFromDisk(&tex_Default, wd_path("default.png").c_str());
    mg_Enemy_Bodyguard = MeshGroupLoad(wd_path("enemy_bodyguard.obj").c_str());
    enemy0.root = vec3(-320, 0, -320);

    sfx_Jump = Mixer_LoadChunk(wd_path("gunshot-37055.ogg").c_str());

//    FaceBatch fb0;
//    CreateFaceBatchMeshOnGPU(&fb0.idVAO, &fb0.idVBO);
//    std::vector<float> fb0data {
//        0, 0, 0, 0, 0, 1, 0, 0,
//        0, 32, 32, 1, 1, 1, 0, 0,
//        0, 0, 32, 0, 1, 1, 0, 0,
//        0, 0, 0, 0, 0, 1, 0, 0,
//        0, 32, 0, 1, 0, 1, 0, 0,
//        0, 32, 32, 1, 1, 1, 0, 0,
//
//        -320, 0, -320, 0, 0, 0, 1, 0,
//        -320, 0, 320, 0, 20, 0, 1, 0,
//        320, 0, 320, 20, 20, 0, 1, 0,
//        -320, 0, -320, 0, 0, 0, 1, 0,
//        320, 0, 320, 20, 20, 0, 1, 0,
//        320, 0, -320, 20, 0, 0, 1, 0,
//    };
//    RebindFaceBatchBufferObject(&fb0, (u32)fb0data.size()*sizeof(float), fb0data.data());
//    fb0.sharedTexture = tex_Default;
//    activeFaceBatches.push_back(fb0);

    // EditableVolumes::Volume vol;
    // EditableVolumes::MakeRectangularVolume(&vol);
    // for (EditableVolumes::Face *f : vol.faces)
    // {
    //     TriangulateFace_QuickDumb(*f, &MY_VERTEX_BUFFER);
    // }

    // RebindFaceBatchBufferObject(&fb0, (u32)MY_VERTEX_BUFFER.size()*sizeof(float), MY_VERTEX_BUFFER.data());
}

static void TickGame()
{
    // ENEMY 0
    vec3 toPlayer = playerControllerRoot - enemy0.root;
    enemy0.facing = Normalize(vec3(toPlayer.x, 0.f, toPlayer.z));
    enemy0.root += enemy0.facing * 32.f * g_DeltaTime;

    // CALCULATE PLAYER FACING DIRECTION
    float camYawDelta = mousedelta.x*0.1f;
    float camPitchDelta = mousedelta.y*0.1f;
    cameraRotation.y -= camYawDelta;
    cameraRotation.z -= camPitchDelta;
    if (cameraRotation.z > 89.f)
        cameraRotation.z = 89.f;
    if (cameraRotation.z < -89.f)
        cameraRotation.z = -89.f;
    cameraDirection = Normalize(OrientationToDirection(EulerToQuat(cameraRotation * GM_DEG2RAD)));
    cameraRight = Normalize(Cross(cameraDirection, GM_UP_VECTOR));
    cameraUp = Normalize(Cross(cameraRight, cameraDirection));
    playerWalkDirectionRight = cameraRight;
    playerWalkDirectionForward = Normalize(Cross(GM_UP_VECTOR, playerWalkDirectionRight));

    // PLAYER MOVE
    float moveSpeed = 250.f;
    vec3 playerPositionDelta;
    if (keyscurrent[SDL_SCANCODE_W])
        playerPositionDelta += playerWalkDirectionForward * moveSpeed * g_DeltaTime;
    if (keyscurrent[SDL_SCANCODE_A])
        playerPositionDelta += -playerWalkDirectionRight * moveSpeed * g_DeltaTime;
    if (keyscurrent[SDL_SCANCODE_S])
        playerPositionDelta += -playerWalkDirectionForward * moveSpeed * g_DeltaTime;
    if (keyscurrent[SDL_SCANCODE_D])
        playerPositionDelta += playerWalkDirectionRight * moveSpeed * g_DeltaTime;
    playerControllerRoot += playerPositionDelta;
    
    static int channelrotationtesting = 0;
    if (keyspressed[SDL_SCANCODE_SPACE])
    {
        playeryvel = 160.f;
        Mix_VolumeChunk(sfx_Jump, 48);
//        Mix_PlayChannel(channelrotationtesting++%3, sfx_Jump, 0);
    }
    playeryvel -= 314.f * g_DeltaTime; // 9.81 m/s * 32 units/m ~= 314 units/s
    playerControllerRoot.y += playeryvel * g_DeltaTime;

    // PLAYER RESOLVE COLLISIONS
    MeshCollider floor;
    floor.pointCloud = {
        vec3(-320,0,-320), vec3(-320,0,320), vec3(320,0,-320), vec3(320,0,320),
        vec3(-320,-100,-320), vec3(-320,-100,320), vec3(320,-100,-320), vec3(320,-100,320),
    };
    MeshCollider *pcol = PlayerColliderAtPos(playerControllerRoot);
    GJKEPACollisionInfo colinfo = GJK(&floor, pcol);

    if (colinfo.hasCollision)
    {
        vec3 moveby = colinfo.normal * colinfo.penetrationDepth;
        playerControllerRoot += moveby;
        if (Dot(colinfo.normal, GM_UP_VECTOR) > 0.1f)
            playeryvel = 0.f;
    }

    // PLAYER CAMERA 
    vec3 cameraPosOffsetFromRoot = vec3(0,40,0);
    vec3 cameraPosition = playerControllerRoot + cameraPosOffsetFromRoot;

    static float camLean = 0.f;
    static float desiredCamLean = 0.f;
    const float camLeanSpeed = 15;
    const float maxCamLean = 0.07f;
    desiredCamLean = 0.f;
    if(keyscurrent[SDL_SCANCODE_D])
        desiredCamLean += maxCamLean;
    if(keyscurrent[SDL_SCANCODE_A])
        desiredCamLean += -maxCamLean;
    camLean = Lerp(camLean, desiredCamLean, g_DeltaTime*camLeanSpeed);

    quat fromto = RotationFromTo(cameraUp, cameraRight);
    quat sle = Slerp(quat(), fromto, camLean);
    vec3 cameraUpWithSway = RotateVector(cameraUp, sle);
    float dot = Dot(Normalize(Cross(cameraUpWithSway, cameraRight)), cameraDirection);
    if (dot < 0.99f)
        printf("bad cam up %f\n", dot);
    activeViewMatrix = ViewMatrixLookAt(cameraPosition, cameraPosition + cameraDirection, cameraUpWithSway);
}

static void RenderGameLayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, renderTargetGame.fbo);
    glViewport(0, 0, renderTargetGame.width, renderTargetGame.height);
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //RGBHEXTO1(0x6495ed), 1.f);//(RGB255TO1(211, 203, 190), 1.f);//(0.674f, 0.847f, 1.0f, 1.f); //RGB255TO1(46, 88, 120)
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    activePerspectiveMatrix = ProjectionMatrixPerspective((90.f/2.f)*GM_DEG2RAD,
                                                          float(BACKBUFFER_WIDTH) / float(BACKBUFFER_HEIGHT), GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = activePerspectiveMatrix;
    mat4 viewMatrix = activeViewMatrix;

    UseShader(sceneShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(sceneShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(sceneShader, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());

    GLActiveTextureUnit0(tex_Default.id);

//    for (size_t i = 0; i < activeFaceBatches.size(); ++i)
//    {
//        FaceBatch fb = activeFaceBatches.at(i);
//        RenderFaceBatch(fb);
//    }

    modelMatrix = TranslationMatrix(enemy0.root) * RotationMatrix(DirectionToOrientation(enemy0.facing));
    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());
    MeshGroupRender(&mg_Enemy_Bodyguard);

    // PRIMITIVES
    if (keyspressed[SDL_SCANCODE_X])
        primitivesDepthTest = !primitivesDepthTest;
    if (primitivesDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, renderTargetGame.depthTexId, vec2((float)renderTargetGame.width,(float)renderTargetGame.height));
}

static void RenderMapEditorLayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, renderTargetGame.fbo);
    glViewport(0, 0, renderTargetGame.width, renderTargetGame.height);
    glClearColor(0.15f, 0.15f, 0.15f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    activePerspectiveMatrix = ProjectionMatrixPerspective((90.f/2.f)*GM_DEG2RAD, float(BACKBUFFER_WIDTH) / float(BACKBUFFER_HEIGHT),
                                                          GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = activePerspectiveMatrix;
    mat4 viewMatrix = activeViewMatrix;

    UseShader(sceneShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(sceneShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(sceneShader, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());


    FaceBatch hoveredFaceMesh;
    for (int i = 0; i < MapEdit::EDITOR_FACES.count; ++i)
    {
        MapEdit::Face *editorVolumeFace = MapEdit::EDITOR_FACES.At(i);
        if (editorActiveTool == MapEditorTools::FaceManip && editorVolumeFace->hovered)
        {
            hoveredFaceMesh = editorVolumeFace->facemesh;
            editorVolumeFace->hovered = false;
            continue;
        }
        const FaceBatch fb = editorVolumeFace->facemesh;
        RenderFaceBatch(fb);
    }

    if (editorActiveTool == MapEditorTools::FaceManip && hoveredFaceMesh.idVAO > 0)
    {
        UseShader(editorShader_FaceSelected);
        GLBindMatrix4fv(editorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
        GLBindMatrix4fv(editorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
        modelMatrix = mat4();
        GLBindMatrix4fv(editorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
        RenderFaceBatch(hoveredFaceMesh);
    }


    // PRIMITIVES
    if (editorActiveTool == MapEditorTools::VertexManip)
    {
        for (int i = 0; i < SELECTABLE_VERTICES.size(); ++i)
        {
            MapEdit::Vert *v = SELECTABLE_VERTICES[i];
            vec4 discHandleColor = vec4(RGBHEXTO1(0xFF8000), 1.f);
            if (std::find(SELECTED_SELECTABLE_VERTICES_INDICES.begin(), SELECTED_SELECTABLE_VERTICES_INDICES.end(), i+1)
                != SELECTED_SELECTABLE_VERTICES_INDICES.end())
                discHandleColor = vec4(RGB255TO1(254,8,8),1.f);
            PrimitiveDrawSolidDisc(v->pos, editorCameraPosition - v->pos, GetEditorHandleSize(v->pos, DISC_HANDLE_RADIUS),
                                   discHandleColor);
        }
    }
    if (keyspressed[SDL_SCANCODE_X])
        primitivesDepthTest = !primitivesDepthTest;
    if (primitivesDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, renderTargetGame.depthTexId, vec2((float)renderTargetGame.width,(float)renderTargetGame.height));
}

static void RenderGUILayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, renderTargetGUI.fbo);
    glViewport(0, 0, renderTargetGUI.width, renderTargetGUI.height);
//    glDepthRange(0.00001f, 10.f); I should just be doing painters algorithm
    glClearColor(RGB255TO1(244, 194, 194), 0.0f);
//    glClearDepth(10.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST); // I forgot why the fuck I'm disabling depth test when using glDepthRange

    Gui::Draw();
}

static void FinalRenderToBackBuffer()
{
    UseShader(finalPassShader);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, BACKBUFFER_WIDTH, BACKBUFFER_HEIGHT);
//    glDepthRange(0, 10);
    glClearColor(RGB255TO1(0, 0, 0), 1.f);
//    glClearDepth(1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    // Draw game frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderTargetGame.colorTexId);
    RenderGPUMesh(__final_render_output_quad);

    // Draw GUI frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderTargetGUI.colorTexId);
    RenderGPUMesh(__final_render_output_quad);

    //    // Draw Debug UI frame
    //    glActiveTexture(GL_TEXTURE0);
    //    glBindTexture(GL_TEXTURE_2D, debugUILayer.colorTexId);
    //    RenderMesh(screenSizeQuad);

    GLHasErrors();
}

static void InitGameRenderer()
{
    // alpha blending func: (srcRGB) * srcA + (dstRGB) * (1 - srcA)  = final color output
    // alpha blending func: (srcA) * a + (dstA) * 1 = final alpha output
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);


    SDL_GL_GetDrawableSize(g_MainSDLWindow, &BACKBUFFER_WIDTH, &BACKBUFFER_HEIGHT);
    renderTargetGame.width = BACKBUFFER_WIDTH;
    renderTargetGame.height = BACKBUFFER_HEIGHT;
    CreateGPUFrameBuffer(&renderTargetGame);
    renderTargetGUI.width = BACKBUFFER_WIDTH/2;
    renderTargetGUI.height = BACKBUFFER_HEIGHT/2;
    CreateGPUFrameBuffer(&renderTargetGUI);


    GLCreateShaderProgram(sceneShader, __scene_shader_vs, __scene_shader_fs);
    GLCreateShaderProgram(editorShader_FaceSelected, __editor_shader_face_selected_vs, __editor_shader_face_selected_fs);
    GLCreateShaderProgram(finalPassShader, __finalpass_shader_vs, __finalpass_shader_fs);


    float refQuadVertices[16] = {
        //  x   y    u    v
        -1.f, -1.f, 0.f, 0.f,
        1.f, -1.f, 1.f, 0.f,
        -1.f, 1.f, 0.f, 1.f,
        1.f, 1.f, 1.f, 1.f
    };
    u32 refQuadIndices[6] = {
        0, 1, 3,
        0, 3, 2
    };
    CreateGPUMesh(&__final_render_output_quad, refQuadVertices, refQuadIndices, 16, 6, 2, 2, 0, GL_STATIC_DRAW);

    InitPrimitivesAndHandlesSystems();
}

static void DrawGame()
{
    RenderGameLayer();
    RenderGUILayer();
    FinalRenderToBackBuffer();
}

static void DrawMapEditor()
{
    RenderMapEditorLayer();
    RenderGUILayer();
    FinalRenderToBackBuffer();
}

static void TickTime()
{
    static std::chrono::high_resolution_clock::time_point timeAtLastUpdate = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float elapsedMs = (float)(std::chrono::duration_cast<std::chrono::microseconds>(now - timeAtLastUpdate)).count() * 0.001f;
    timeAtLastUpdate = now;
    float deltaTimeInSeconds = elapsedMs * 0.001f; // elapsed time in SECONDS
    float currentTimeInSeconds = (float)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() * 0.001f);
    g_CurrentTime = currentTimeInSeconds;
    g_UnscaledDeltaTime = deltaTimeInSeconds;
    g_TimeSinceStart += g_UnscaledDeltaTime;
    g_DeltaTime = g_UnscaledDeltaTime * g_GameTimeScale;
}

static bool InitializeEverything()
{
    g_ProgramShouldShutdown = false;

    SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "system"); // https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/LearnWin32/dpi-and-device-independent-pixels.md#dwm-scaling
    SDL_SetHint(SDL_HINT_WINDOWS_DPI_SCALING, "0"); // https://github.com/libsdl-org/SDL/commit/ab81a559f43abc0858c96788f8e00bbb352287e8

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) return false;

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    g_MainSDLWindow = SDL_CreateWindow("game",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       2560,
                                       1440,
                                       SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    g_SDLGLContext = SDL_GL_CreateContext(g_MainSDLWindow);

    if (g_MainSDLWindow == nullptr || g_SDLGLContext == nullptr) return false;

#ifdef MESA_USING_GL3W
    if (gl3w_init())
    {
        fprintf(stderr, "Failed to initialize OpenGL\n");
        return false;
    }
#endif

    SDL_SetWindowMinimumSize(g_MainSDLWindow, 200, 100);
    SDL_GL_SetSwapInterval(0);
    // SDL_SetWindowMouseGrab(g_MainSDLWindow, SDL_TRUE);

    if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0 || Mix_Init(MIX_INIT_OGG) < 0)
        return false;

    Gui::Init();

    return true;
}

static void ProcessSDLEvents()
{
    mousedelta = vec2(0.f,0.f);
    ivec2 throwaway;
    u32 mousestate = SDL_GetMouseState(&throwaway.x, &throwaway.y);
    u32 mousechanged = mousecurrent ^ mousestate;
    mousepressed = mousechanged & mousestate;
    mousereleased = mousechanged & mousecurrent;
    mousecurrent = mousestate;

    const u8 *keystate = SDL_GetKeyboardState(NULL);
    u8 keyschanged[256];
    for (int i=0;i<256;++i)
    {
        keyschanged[i] = keyscurrent[i] ^ keystate[i];
        keyspressed[i] = keyschanged[i] & keystate[i];
        keysreleased[i] = keyschanged[i] & keyscurrent[i];
        keyscurrent[i] = keystate[i];
    }

    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_WINDOWEVENT:
            {
                switch (event.window.event) 
                {
                    case SDL_WINDOWEVENT_SIZE_CHANGED:
                    case SDL_WINDOWEVENT_RESIZED:
                    {
                        SDL_GL_GetDrawableSize(g_MainSDLWindow, &BACKBUFFER_WIDTH, &BACKBUFFER_HEIGHT);
                        UpdateGPUFrameBufferSize(&renderTargetGame, BACKBUFFER_WIDTH, BACKBUFFER_HEIGHT);
                        UpdateGPUFrameBufferSize(&renderTargetGUI, BACKBUFFER_WIDTH/2, BACKBUFFER_HEIGHT/2);
                        break;
                    }
                }
                break;
            }

            case SDL_QUIT:
            {
                g_ProgramShouldShutdown = true;
                break;
            }

            case SDL_KEYDOWN:
            {
                SDL_Keycode sdlkey = event.key.keysym.sym;

                if (sdlkey == SDLK_RETURN && SDL_GetModState() & KMOD_LALT)
                {
                    if (SDL_GetWindowFlags(g_MainSDLWindow) & SDL_WINDOW_FULLSCREEN_DESKTOP)
                        SDL_SetWindowFullscreen(g_MainSDLWindow, 0);
                    else
                        SDL_SetWindowFullscreen(g_MainSDLWindow, SDL_WINDOW_FULLSCREEN_DESKTOP); 
                    event.type = 0;
                }
                break;
            }

            case SDL_MOUSEMOTION: 
            {
                mousedelta = vec2((float)event.motion.xrel, (float)event.motion.yrel);
                mousepos = ivec2(event.motion.x, event.motion.y);
                break;
            }
        }

        Gui::ProcessSDLEvent(event);
    }
}

int main(int argc, char* argv[])
{
    if (!InitializeEverything()) return -1;

    InitGameRenderer();

    Stuff();

    while (!g_ProgramShouldShutdown)
    {
        TickTime();
        if (g_DeltaTime > 0.1f) continue; // if delta time is too large, will cause glitches

        Gui::NewFrame();
        ProcessSDLEvents();

        // TickGame();
        // DrawGame();
        TickMapEditor();
        DrawMapEditor();

        SDL_GL_SwapWindow(g_MainSDLWindow);

        // printf("LAST FRAME TIME: %.2fms   FPS: %dhz\n", 1000.f * g_UnscaledDeltaTime, (i16)(1.f / g_UnscaledDeltaTime));
    }

    SDL_DestroyWindow(g_MainSDLWindow);
    SDL_GL_DeleteContext(g_SDLGLContext);
    SDL_Quit();
    return 0;
}
