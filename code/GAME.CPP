/*

TODO:
- Make simple textured playground level
- Skeletal animations
- Enemy moves and shoots at player
- Gun model?
- Player shoots at enemy

- serialize/deserialize asset database
- better collision surfaces
- move volume mode/tool (don't need whole tool...maybe make brush tool part of this mode? like trenchbroom)
- multi select vertices need to be fixed
- edge select

EPICS:
- Lighting (build/load level LIGHT MAP info)
- Enemy billboards two behaviour:
    - walk towards player
    - stop and shoot after split second to shoot projectile at player, projectile hurts player health
- Player holds 3D gun model, shoot hitscan at enemies, hitscan hurts enemy health
- Swap to a second 3D gun model, shoot shotgun hitscan at enemies
- Swap to a third 3D gun model, shoot hitscan with different stats at enemies

OTHER:
Finish implementing Editable Volumes and Atomics
(poly draw) Draw poly on plane then raise to create volume tool

DONT FORGET:
Port over Console front and backend


use flat faces for collision? instead of meshes? GJK EPA code accepts 2D meshes


Priorty #1 is building the game, not the engine/tech
Engine work is almost done. 
~~Octree collision acceleration~~
Skeletal animations
Navmesh/pathfinding
Lightmap baking (potentially radiosity) 
Those are the hard things left.


I don't really want to support non flat faces

The game level static colliders being flat polygons makes collisions weird. They are double sided.
What if I optionally enforce resolution normal for some colliders? that would make resolution only happen
in that direction. But that doesn't solve the problem of colliding at all with those double sided colliders.
Maybe...for player I can check if that optional normal (basically face normal) is same direction (cos theta)
as player movement direction and if so ignore that collider?

cuz even if I was using 3D volumes for collisions, the issue persists...If I have three cubes sitting flush
with each other, and I was running diagonally into them, I might stop moving because I'm colliding at one of
the seams.

if I'm moving in a direction, I can only collide with surfaces whose normals are in the opposite direction of
that my movement direction. This is a fact.

I think I just need to be more intentional with assigning colliders? and each surface collider should have
a resolution normal defined.



Notes:

    64x64 pixel texture for each 32x32 unit in game looks decent
    
    try for every 16 units in the game, 1 lightmap sample


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
#define STB_DS_IMPLEMENTATION
#include <stb_ds.h>

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <stack>
#include <memory>
#include <utility>




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
#define ALWAYSASSERT(predicate) if(!(predicate)) { __debugbreak(); }
#else
#define ALWAYSASSERT(predicate) if(!(predicate)) { __builtin_trap(); }
#endif

#if INTERNAL_BUILD
    #define ASSERT(predicate) ALWAYSASSERT(predicate)
#else
    #define ASSERT(predicate)
#endif

inline std::string wd_path() { return std::string(PROJECT_WORKING_DIR); }
inline std::string wd_path(const std::string& name) { return wd_path() + std::string(name); }
inline std::string model_path() { return wd_path() + "models/"; }
inline std::string model_path(const std::string& name) { return wd_path() + "models/" + name; }
inline std::string texture_path() { return wd_path() + "textures/"; }
inline std::string texture_path(const std::string& name) { return wd_path() + "textures/" + name; }
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

bool IsEditorActive = false;


#pragma endregion



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
    if (min_filter_mode == GL_NEAREST_MIPMAP_LINEAR || 
        min_filter_mode == GL_NEAREST_MIPMAP_NEAREST ||
        min_filter_mode == GL_LINEAR_MIPMAP_NEAREST ||
        min_filter_mode == GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter_mode); // filtering (e.g. GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter_mode);
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



struct ModelGLTF
{
    GPUMesh    *meshes   = NULL;
    GPUTexture *color    = NULL;
    // animations and bones and shit
};

void FreeModelGLTF(ModelGLTF model)
{
    ASSERT(arrlen(model.meshes) == arrlen(model.color));

    size_t meshcount = arrlenu(model.meshes);
    for (size_t i = 0; i < meshcount; ++i)
    {
        DeleteGPUMesh(&model.meshes[i]);
        DeleteGPUTexture(&model.color[i]);
    }

    arrfree(model.meshes);
    arrfree(model.color);
}

void RenderModelGLTF(ModelGLTF model)
{
    size_t meshcount = arrlenu(model.meshes);
    for (size_t i = 0; i < meshcount; ++i)
    {
        GPUMesh    m = model.meshes[i];
        GPUTexture t = model.color[i];

        // TODO(Kevin): if t.id is 0 then bind MissingTexture 
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, t.id);

        RenderGPUMesh(m);
    }
}

GPUMesh ASSIMPMeshToGPUMesh(aiMesh* meshNode)
{
    const u32 vertexStride = 8;

    std::vector<float> vb(meshNode->mNumVertices * vertexStride);
    std::vector<u32> ib(meshNode->mNumFaces * meshNode->mFaces[0].mNumIndices);
    if (meshNode->mTextureCoords[0])
    {
        for (size_t i = 0; i < meshNode->mNumVertices; ++i)
        {
            // mNormals and mVertices are both mNumVertices in size
            size_t v_start_index = i * vertexStride;
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
            size_t v_start_index = i * vertexStride;
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
            ib[i * face.mNumIndices + j] = face.mIndices[j];
        }
    }

    GPUMesh mesh;
    CreateGPUMesh(&mesh, &vb[0], &ib[0], (u32)vb.size(), (u32)ib.size());
    return mesh;
}

bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath)
{
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(filepath,
        aiProcess_Triangulate);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        LogError("ASSIMP failed to load file at '%s'.\nErr msg: %s", filepath, importer.GetErrorString());
        return false;
    }

    // == Load and create all the textures ==
    // For each material, I'm only going to load one texture per texture type.
    // I'm only going to support embedded textures (GLB).
    // Assume 8-bit per channel RGBA format for texture input format

    GPUTexture *matEmissiveTextures = NULL;

    for (u32 matIndex = 0; matIndex < scene->mNumMaterials; ++matIndex)
    {
        aiMaterial *mat = scene->mMaterials[matIndex];

        GPUTexture gputexEmissive;
        if (mat->GetTextureCount(aiTextureType_EMISSIVE))
        {
            aiString path;
            if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &path) == AI_SUCCESS)
            {
                ALWAYSASSERT(path.C_Str()[0] == '*'); // Assert texture is embedded into the binary

                int textureIndex = std::atoi(path.C_Str()+1); // Skip the '*' character
                
                ALWAYSASSERT(textureIndex >= 0 && textureIndex < (int)scene->mNumTextures);

                aiTexture *texture = scene->mTextures[textureIndex];

                bool compressed = texture->mHeight == 0;
                u8 *rawPixelData = (u8*)texture->pcData;
                i32 width = texture->mWidth; // Width is stored in mWidth for uncompressed
                i32 height = texture->mHeight; // Height is stored in mHeight for uncompressed

                // Uncompress if compressed format (e.g. PNG / JPG)
                if (compressed)
                {
                    u8 *compressedImageData = (u8*)texture->pcData;
                    i32 channelsInFile;
                    rawPixelData = stbi_load_from_memory(compressedImageData, texture->mWidth, &width, &height, &channelsInFile, 0);
                    ALWAYSASSERT(channelsInFile == 4);
                }

                ALWAYSASSERT(rawPixelData);

                CreateGPUTextureFromBitmap(&gputexEmissive, rawPixelData, width, height, 
                    GL_RGBA, GL_RGBA, GL_NEAREST, GL_NEAREST);

                if (compressed)
                {
                    stbi_image_free(rawPixelData);
                }
            }
        }
        arrput(matEmissiveTextures, gputexEmissive);
    }

    ASSERT(scene->mNumMeshes > 0);
    ASSERT(model->meshes == NULL);

    arrsetcap(model->meshes, scene->mNumMeshes);
    arrsetcap(model->color, scene->mNumMeshes);

    for (u32 meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
    {
        aiMesh* meshNode = scene->mMeshes[meshIndex];
        GPUMesh gpumesh = ASSIMPMeshToGPUMesh(meshNode);
        u32 matIndex = meshNode->mMaterialIndex;
        GPUTexture colorTex = matEmissiveTextures[matIndex];

        arrput(model->meshes, gpumesh);
        arrput(model->color, colorTex);
    }

    arrfree(matEmissiveTextures);

    return true;
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
#include "COLLISION.CPP"

#pragma region MORE_GLOBAL_VARIABLES
GPUShader sceneShader;
GPUShader editorShader_Wireframe;
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






// MIXER
Mix_Chunk *Mixer_LoadChunk(const char *filepath)
{
    Mix_Chunk *chunk = Mix_LoadWAV(filepath);
    if (chunk == NULL)
        printf("Failed to load sound effect! SDL_mixer error: %s\n", Mix_GetError());
    return chunk;
}


GPUTexture tex_Missing;
Mix_Chunk *sfx_Jump;

std::vector<vec3> GameLevelColliderPoints;
std::vector<StaticMeshCollider> GameLevelColliders;
Octree GameLevelCollisionTree;
std::vector<FaceBatch> GameLevelFaceBatches;

vec3 playerControllerRoot = vec3(215.f,0.01f,-38.7f);
vec3 cameraRotation = vec3(0,192.3f,7.56f);// vec3(0,130,-30);
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


struct BodyguardEnemy
{
    vec3 root;
    vec3 facing;
};
BodyguardEnemy enemy0;

ModelGLTF model_Knight;


#include "EDITOR.CPP"


static void Stuff()
{
    SDL_SetRelativeMouseMode(SDL_TRUE);

    CreateGPUTextureFromDisk(&tex_Missing, wd_path("missing_texture.png").c_str());
    enemy0.root = vec3(-200, 0, -200);

    sfx_Jump = Mixer_LoadChunk(wd_path("gunshot-37055.ogg").c_str());

    LoadModelGLTF2Bin(&model_Knight, wd_path("models/knight.glb").c_str());


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
    // enemy0.root += enemy0.facing * 32.f * g_DeltaTime;

    // CALCULATE PLAYER FACING DIRECTION
    float camYawDelta = mousedelta.x*0.055f;
    float camPitchDelta = mousedelta.y*0.055f;
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

    // temp crosshair
    ivec2 guiwh = ivec2(renderTargetGUI.width, renderTargetGUI.height);
    Gui::PrimitivePanel(Gui::UIRect(guiwh.x/2 - 3, guiwh.y/2 - 3, 6, 6), vec4(0,0,0,1));
    Gui::PrimitivePanel(Gui::UIRect(guiwh.x/2 - 2, guiwh.y/2 - 2, 4, 4), vec4(1,1,1,1));


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
    MeshCollider *pcol = PlayerColliderAtPos(playerControllerRoot);
    Bounds playerBounds = BoundsFromMeshCollider(*pcol);
    std::unordered_set<Collider*> pcollisionquery = GameLevelCollisionTree.Query(playerBounds);
    // NOTE(Kevin): If I wanted, I could do something like sort these potential colliders by walls before floors or something
    for (Collider *levelCollider : pcollisionquery)
    {
        ColResult result = GJK(levelCollider, pcol);
        if (result.hasCollision)
        {
            vec3 moveby = result.normal * result.penetrationDepth;
            playerControllerRoot += moveby;
            if (Dot(result.normal, GM_UP_VECTOR) > 0.1f)
                playeryvel = 0.f;
        }
    }

    // PLAYER CAMERA 
    vec3 cameraPosOffsetFromRoot = vec3(0,40,0);
    vec3 cameraPosition = playerControllerRoot + cameraPosOffsetFromRoot;
    // LogMessage("pos %f, %f, %f", playerControllerRoot.x, playerControllerRoot.y, playerControllerRoot.z);
    // LogMessage("dir y %f, z %f\n", cameraRotation.y, cameraRotation.z);

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

    float aspectratio = float(BACKBUFFER_WIDTH) / float(BACKBUFFER_HEIGHT);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio); // NOTE(Kevin): have FOV slider in game up to insane value like 170
    activePerspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = activePerspectiveMatrix;
    mat4 viewMatrix = activeViewMatrix;

    UseShader(sceneShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(sceneShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(sceneShader, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());


    for (size_t i = 0; i < GameLevelFaceBatches.size(); ++i)
    {
        FaceBatch fb = GameLevelFaceBatches.at(i);
        RenderFaceBatch(fb);
    }

    modelMatrix = TranslationMatrix(enemy0.root) * RotationMatrix(DirectionToOrientation(enemy0.facing));
    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());
    // RenderModelGLTF(model_Knight);

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

    float aspectratio = float(BACKBUFFER_WIDTH) / float(BACKBUFFER_HEIGHT);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    activePerspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = activePerspectiveMatrix;
    mat4 viewMatrix = activeViewMatrix;

    UseShader(sceneShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(sceneShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(sceneShader, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());

    FaceBatch selectedFaceMesh;
    FaceBatch hoveredFaceMesh;
    for (int i = 0; i < MapEdit::EDITOR_FACES.count; ++i)
    {
        MapEdit::Face *editorVolumeFace = MapEdit::EDITOR_FACES.At(i);
        if (editorActiveTool == MapEditorTools::FaceManip)
        {
            if (Session.SelectedFace == editorVolumeFace)
            {
                selectedFaceMesh = editorVolumeFace->facemesh;
                continue;
            }
            else if (editorVolumeFace->hovered)
            {
                // DONT CARE ABOUT HOVERED hoveredFaceMesh = editorVolumeFace->facemesh;
                editorVolumeFace->hovered = false;
                //continue;
            }
        }

        const FaceBatch fb = editorVolumeFace->facemesh;
        RenderFaceBatch(fb);
    }

    if (editorActiveTool == MapEditorTools::FaceManip)
    {
        if (selectedFaceMesh.idVAO > 0)
        {
            UseShader(editorShader_FaceSelected);
            float sf = (sinf(g_TimeSinceStart * 2.7f) + 1.f) / 2.f;
            sf *= 0.1f;
            if (primitivesDepthTest)
                sf = 0.0f;
            GLBind3f(editorShader_FaceSelected, "tint", 1.0f, 1.0f-sf, 1.0f-sf);
            GLBindMatrix4fv(editorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
            GLBindMatrix4fv(editorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
            modelMatrix = mat4();
            GLBindMatrix4fv(editorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
            RenderFaceBatch(selectedFaceMesh);
        }
        else if (hoveredFaceMesh.idVAO > 0)
        {
            UseShader(editorShader_FaceSelected);
            GLBind3f(editorShader_FaceSelected, "tint", 0.9f, 0.9f, 0.0f);
            GLBindMatrix4fv(editorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
            GLBindMatrix4fv(editorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
            modelMatrix = mat4();
            GLBindMatrix4fv(editorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
            RenderFaceBatch(hoveredFaceMesh);
        }
    }

    // PRIMITIVES
    // Draw outline of selected faces
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        const MapEdit::Volume& volume = EDITOR_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
        for (MapEdit::Face *selVolFace : volume.faces)
        {
            std::vector<MapEdit::Edge*> faceEdges = selVolFace->GetEdges();
            for (MapEdit::Edge* e : faceEdges)
            {
                PrimitiveDrawLine(e->a->pos, e->b->pos, vec4(1,0,0,0.5f), 2.f);
            }
        }
    }
    // Draw vertex handles
    if (editorActiveTool == MapEditorTools::VertexManip)
    {
        for (int i = 0; i < SELECTABLE_VERTICES.size(); ++i)
        {
            MapEdit::Vert *v = SELECTABLE_VERTICES[i];
            vec4 discHandleColor = vec4(RGBHEXTO1(0xFF8000), 1.f);
            if (std::find(SELECTED_VERTICES.begin(), SELECTED_VERTICES.end(), v) != SELECTED_VERTICES.end())
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
    vec3 gridTranslation = vec3(SnapToGrid(editorCameraPosition.x), 0.f, SnapToGrid(editorCameraPosition.z));
    mat3 gridRotation = mat3();
    if (GRID_ORIGIN != vec3())
    {
        gridTranslation = GRID_ORIGIN;
        gridRotation = GetGridRotationMatrix();
    }
    DrawGrid(GRID_INCREMENT, gridRotation, gridTranslation, &perspectiveMatrix, &viewMatrix, renderTargetGame.depthTexId, vec2((float)renderTargetGame.width,(float)renderTargetGame.height));
    PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, renderTargetGame.depthTexId, vec2((float)renderTargetGame.width,(float)renderTargetGame.height));

    // // UseShader(editorShader_Wireframe);
    // // glEnable(GL_CULL_FACE);
    // // GLBindMatrix4fv(editorShader_Wireframe, "projMatrix", 1, perspectiveMatrix.ptr());
    // // GLBindMatrix4fv(editorShader_Wireframe, "viewMatrix", 1, viewMatrix.ptr());
    // // GLBindMatrix4fv(editorShader_Wireframe, "modelMatrix", 1, modelMatrix.ptr());
    // // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // glEnable(GL_DEPTH_TEST);
    // for (int i = 0; i < MapEdit::EDITOR_FACES.count; ++i)
    // {
    //     MapEdit::Face *editorVolumeFace = MapEdit::EDITOR_FACES.At(i);
    //     std::vector<MapEdit::Edge*> faceEdges = editorVolumeFace->GetEdges();
    //     for (MapEdit::Edge* e : faceEdges)
    //     {
    //         PrimitiveDrawLine(e->a->pos, e->b->pos, vec4(1,1,1,1), 1.2f);
    //     }
    //     // const FaceBatch fb = editorVolumeFace->facemesh;
    //     // RenderFaceBatch(fb);
    // }
    // PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, renderTargetGame.depthTexId, vec2((float)renderTargetGame.width,(float)renderTargetGame.height));
    // if (primitivesDepthTest)
    //     glEnable(GL_DEPTH_TEST);
    // else
    //     glDisable(GL_DEPTH_TEST);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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
    GLCreateShaderProgram(editorShader_Wireframe, __editor_scene_wireframe_shader_vs, __editor_scene_wireframe_shader_fs);
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

    if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0 || Mix_Init(MIX_INIT_OGG) < 0)
        return false;

    Gui::Init();

    return true;
}

static void ProcessSDLEvents()
{
    // MOUSE
    ivec2 prevmp = mousepos;
    u32 mousestate = SDL_GetMouseState(&mousepos.x, &mousepos.y); // mousepos only valid when mouse not relative mode
    u32 mousechanged = mousecurrent ^ mousestate;
    mousepressed = mousechanged & mousestate;
    mousereleased = mousechanged & mousecurrent;
    mousecurrent = mousestate;
    ivec2 md;
    SDL_GetRelativeMouseState(&md.x, &md.y);
    mousedelta.x = (float)md.x;
    mousedelta.y = (float)md.y;

    // KEYBOARD
    const u8 *keystate = SDL_GetKeyboardState(NULL);
    u8 keyschanged[256];
    for (int i=0;i<256;++i)
    {
        keyschanged[i] = keyscurrent[i] ^ keystate[i];
        keyspressed[i] = keyschanged[i] & keystate[i];
        keysreleased[i] = keyschanged[i] & keyscurrent[i];
        keyscurrent[i] = keystate[i];
    }

    // EVENT HANDLING
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
        }

        Gui::ProcessSDLEvent(event);
    }
}

int main(int argc, char* argv[])
{
    if (!InitializeEverything()) return -1;

    InitGameRenderer();

    Stuff();

    LoadMapEditor();

    if (LoadGameMap(wd_path("LightTest.map").c_str()) == false)
        LogError("failed to load game map");

    while (!g_ProgramShouldShutdown)
    {
        TickTime();
        Sleep(6);
        if (g_DeltaTime > 0.1f) continue; // if delta time is too large, will cause glitches

        Gui::NewFrame();
        ProcessSDLEvents();

        TickGame();
        DrawGame();
        //TickMapEditor();
        //DrawMapEditor();

        SDL_GL_SwapWindow(g_MainSDLWindow);

        // printf("LAST FRAME TIME: %.2fms   FPS: %dhz\n", 1000.f * g_UnscaledDeltaTime, (i16)(1.f / g_UnscaledDeltaTime));
    }

    CloseMapEditor();

    SDL_DestroyWindow(g_MainSDLWindow);
    SDL_GL_DeleteContext(g_SDLGLContext);
    SDL_Quit();
    return 0;
}
