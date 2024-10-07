/*

Something fucking cool.

Priorty #1 is building the game, not the engine/tech

Just be specific and code exactly that tailored solution e.g. LevelPolygonOctree rather than Octree.

Handcrafted with love.


- Irradiance Caching!!!
    My Irradiance Caching implementation might be marginally slower than drawing hemicube
    at every single texel at a big patch size like 8x8 or 4x4. BUT! The point of it is that
    I can scale down the patch size (i.e. way more texels per surface) and irradiance caching
    should perform better than drawing millions of hemicubes. Only one way to find out.

    Reduce hemicube sample count (adaptive sampling)
    Able to extrapolate irradiance values across surface
    Bad hemicube samples are corrected for free via extrapolation. Just need to detect and discard bad samples.

- Revisit lightmap data transfers and buffers
    Larger Hemicube atlas for download. Also I have so many intermediate buffers for light.


- Apply light map to FaceBatch as uv2 and texture as uv1
- I should have option (by default) to just create faces rather than cuboid volumes in my editor. 
    Volume wasteful bcause lots of useless collider faces and light mapped faces.
    I think Source 2 mainly uses face creation rather than volume.
      Best workflow is extrude faces and extrude edges.


TODO:
- Multiple light sources
- Skeletal animations
- Better collision surfaces
- Enemy moves and shoots at player
- Gun model?
- Player shoots at enemy
- Replace all the dogshit GJK stuff with SPECIFIC collider checks (GJK for finegrained narrow-phase test on complex meshes; do quicker broad-phase tests; simpler test if just polygon x line)
    Bullets can check a Line from position last frame to position this frame against Polygon (face). 
    "Ball" projectiles could also do that? Then narrow-phase sphere against polygon. Most projectiles even flying enemies will be spheres (at least against level geometry).
- serialize/deserialize asset database
- move volume mode/tool (don't need whole tool...maybe make brush tool part of this mode? like trenchbroom)
- multi select vertices need to be fixed
- Edge select
- Edge loop
- Toggle element translation along axis
- Gamma correction - set GL_SRGB for textures being loaded into game so that they are not gamma corrected twice
- do GL_LINEAR_MIPMAP_LINEAR for minification filter for game textures
- Anisotropic filtering https://www.khronos.org/opengl/wiki/Sampler_Object#Anisotropic_filtering

Engine work is almost done. 
~~- Octree collision acceleration~~
~~- Lightmap baking (potentially radiosity)~~
- Skeletal animations
- Navmesh/pathfinding
- More reliable collisions and colliders
Those are the hard things left.

EPICS:
- Lighting (build/load level LIGHT MAP info)
- Enemy billboards two behaviour:
    - walk towards player
    - stop and shoot after split second to shoot projectile at player, projectile hurts player health
- Player holds 3D gun model, shoot hitscan at enemies, hitscan hurts enemy health
- Swap to a second 3D gun model, shoot shotgun hitscan at enemies
- Swap to a third 3D gun model, shoot hitscan with different stats at enemies

OTHER:
Try upgrading to SDL3 first...or to latest SDL2...ugh might still just have to rewrite platforms code
Rewrite platforms code in Win32 and remove SDL - FPS mouselook feels janky (lag + stutter when low framerate), yaw and pitch calculation as euler might be dumb. Just write directly to a rotation matrix?
Finish implementing Editable Volumes and Atomics
(poly draw) Draw poly on plane then raise to create volume tool

BUGS:
- (2024-09-24 T480) periodic frame rate drops down to ~59/60fps then back up when in fullscreen mode (both SDL_WINDOW_FULLSCREEN and SDL_WINDOW_FULLSCREEN_DESKTOP)
                    I don't think it has to do with TickTime - Changing to use SDL_GetTicks64 no effect.
                    Look into https://wiki.libsdl.org/SDL2/SDL_GetWindowDisplayMode
                    Also, setting the window size to the screen size when creating window makes SDL enter fullscreen?

NOTES:

    64x64 pixel texture for each 32x32 unit in game looks decent

    Embrace the fact that lighting will be crude and not perfect. The visual artifacts is part of the charm of my game and engine. Something that differentiates it from the perfect crispy lighting of engines like Godot or Unity.

    Ultimately, the game will very much have my identity. From some UI looking crusty, or enemy animations being janky, but that personal touch is part of the charm of an indie game like this.

    port over dropdown console?

    Fun ideas:
        FOV slider in game up to insane value like 170
        Allow turning bilinear filtering for light maps off



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


*/
#define SUNLIGHT_TEST 1

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
#include <direct.h>
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
#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#define STB_DS_IMPLEMENTATION
#include <stb_ds.h>

#define VERTEXT_IMPLEMENTATION
#include <vertext.h>
#include <gmath.h>
#if INTERNAL_BUILD
#include <renderdoc_app.h>
#endif

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <stack>
#include <memory>
#include <utility>
#include <thread>




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

SDL_Window *SDLMainWindow;
SDL_GLContext SDLGLContext;
bool ProgramShutdownRequested = false;
float DeltaTime = 0.f;
float UnscaledDeltaTime = 0.f;
float GameTimeScale = 1.f;
float CurrentTime = 0.f;
float TimeSinceStart = 0.f;
u32 MouseCurrent;
u32 MousePressed;
u32 MouseReleased;
vec2 MouseDelta;
ivec2 MousePos;
u8 KeysCurrent[256] = {0};
u8 KeysPressed[256] = {0};
u8 KeysReleased[256] = {0};

i32 BackbufferWidth = -1;
i32 BackbufferHeight = -1;

char CurrentWorkingDirectory[128];

#if INTERNAL_BUILD
RENDERDOC_API_1_6_0 *RDOCAPI = NULL;
#endif

bool IsEditorActive = false;

std::vector<float> MyFuckingMesh;


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
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
    GLenum TargetFormat = GL_NONE; // internalformat
    GLenum SourceFormat = GL_NONE; // pixel data format
    GLenum PixelDataType = GL_NONE; // pixel data type
};

bool GLHasErrors();

static void CreateGPUFrameBuffer(GPUFrameBuffer *buffer, 
    GLenum TargetFormat = GL_RGBA, GLenum SourceFormat = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE)
{
    buffer->fbo = 0;

    glGenTextures(1, &buffer->colorTexId);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, TargetFormat, buffer->width, buffer->height, 0, SourceFormat, type, nullptr);
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

static void DeleteGPUFrameBuffer(GPUFrameBuffer *buffer)
{
    glDeleteTextures(1, &buffer->colorTexId);
    glDeleteTextures(1, &buffer->depthTexId);
    glDeleteFramebuffers(1, &buffer->fbo);
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
                                       void              *bitmap,
                                       u32               bitmap_width,
                                       u32               bitmap_height,
                                       GLenum            target_format,
                                       GLenum            source_format,
                                       GLenum            min_filter_mode = GL_NEAREST,
                                       GLenum            mag_filter_mode = GL_NEAREST,
                                       GLenum            pixel_data_type = GL_UNSIGNED_BYTE)
{
    if (bitmap == NULL)
    {
        LogError("CreateGPUTextureFromBitmap error: provided bitmap is null.");
        return;
    }

    ASSERT(texture->id == 0);

    texture->width = bitmap_width;
    texture->height = bitmap_height;
    texture->TargetFormat = target_format;
    texture->SourceFormat = source_format;
    texture->PixelDataType = pixel_data_type;

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
            pixel_data_type,          // data type of the texture data
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
                               (void*) textureBitmapHandle.memory,
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
        texture->TargetFormat,    // format of data to store (target): num of color components
        w,                        // texture width
        h,                        // texture height
        0,                        // must be 0 (legacy)
        texture->SourceFormat,    // format of data being loaded (source)
        texture->PixelDataType,   // data type of the texture data
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
    texture->TargetFormat = GL_NONE;
    texture->SourceFormat = GL_NONE;
    texture->PixelDataType = GL_NONE;
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
                void *rawPixelData = (void*)texture->pcData;
                i32 width = texture->mWidth; // Width is stored in mWidth for uncompressed
                i32 height = texture->mHeight; // Height is stored in mHeight for uncompressed

                // Uncompress if compressed format (e.g. PNG / JPG)
                if (compressed)
                {
                    u8 *compressedImageData = (u8*)texture->pcData;
                    i32 channelsInFile;
                    rawPixelData = (void*)stbi_load_from_memory(compressedImageData, texture->mWidth, &width, &height, &channelsInFile, 0);
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
GPUShader GameLevelShader;
GPUShader PatchesIDShader;
GPUShader EditorShader_Scene;
GPUShader EditorShader_Wireframe;
GPUShader EditorShader_FaceSelected;
GPUShader FinalPassShader;
GPUFrameBuffer RenderTargetGame;
GPUFrameBuffer RenderTargetGUI;
GPUMesh FinalRenderOutputQuad;
float GAMEPROJECTION_NEARCLIP = 4.f; // even 2 works fine to remove z fighting
float GAMEPROJECTION_FARCLIP = 32000.f;
bool DoPrimitivesDepthTest = false;
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

GPUTexture Temporary_LightMapVisualizeTex;
GPUTexture Temporary_TestTex0;
GPUTexture Temporary_TestTex1;

GPUTexture tex_Missing;
Mix_Chunk *sfx_Jump;

std::vector<vec3> GameLevelColliderPoints;
std::vector<FlatPolygonCollider> GameLevelColliders;
Octree GameLevelCollisionTree;
std::vector<FaceBatch> GameLevelFaceBatches;

#if SUNLIGHT_TEST
vec3 PlayerControllerRoot = vec3(0.f, 0.01f, 0.f);
#else
vec3 PlayerControllerRoot = vec3(215.f, 0.01f, -38.7f);
#endif
vec3 CameraRotation = vec3(0, 192.3f, 7.56f);// vec3(0,130,-30);
vec3 CameraDirection;
vec3 CameraRight;
vec3 CameraUp;
mat4 ActivePerspectiveMatrix;
mat4 ActiveViewMatrix;
vec3 PlayerWalkDirectionForward;
vec3 PlayerWalkDirectionRight;
vec3 PlayerColliderPoints[] = {
    vec3(8,0,8), vec3(8,0,-8), vec3(-8,0,8), vec3(-8,0,-8),
    vec3(8,48,8), vec3(8,48,-8), vec3(-8,48,8), vec3(-8,48,-8)
};
MeshCollider PlayerCollider;
MeshCollider *PlayerColliderAtPos(vec3 playerWorldPosition)
{
    PlayerCollider.pointCloud.resize(8);
    for (int i = 0; i < 8; ++i)
        PlayerCollider.pointCloud[i] = playerWorldPosition + PlayerColliderPoints[i];
    return &PlayerCollider;
}
float PlayerYVel = 0.f;

vec3 BodyguardEnemyColliderPoints[] = {
    vec3(8,0,8), vec3(8,0,-8), vec3(-8,0,8), vec3(-8,0,-8),
    vec3(8,48,8), vec3(8,48,-8), vec3(-8,48,8), vec3(-8,48,-8)
};
MeshCollider BodyguardEnemyCollider;
MeshCollider *BodyguardEnemyColliderAtPos(vec3 worldPos)
{
    BodyguardEnemyCollider.pointCloud.resize(8);
    for (int i = 0; i < 8; ++i)
        BodyguardEnemyCollider.pointCloud[i] = worldPos + BodyguardEnemyColliderPoints[i];
    return &BodyguardEnemyCollider;
}


struct BodyguardEnemy
{
    vec3 root;
    vec3 facing;
};
BodyguardEnemy Enemy0;

ModelGLTF Model_Knight;


#include "EDITOR.CPP"


static void Stuff()
{
    SDL_SetRelativeMouseMode(SDL_TRUE);

    CreateGPUTextureFromDisk(&tex_Missing, wd_path("missing_texture.png").c_str());
    Enemy0.root = vec3(-200, 0, -200);

    sfx_Jump = Mixer_LoadChunk(wd_path("gunshot-37055.ogg").c_str());

    LoadModelGLTF2Bin(&Model_Knight, wd_path("models/knight.glb").c_str());


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
#if !SUNLIGHT_TEST
    PrimitiveDrawSolidDisc(TestLightSource, -CameraDirection, 3.f);
    // PrimitiveDrawSolidDisc(TestLightSource2, -CameraDirection, 3.f);
#endif
    // Gui::PrimitivePanel(Gui::UIRect(0,0, 400,400), Temporary_TestTex0.id);
    // Gui::PrimitivePanel(Gui::UIRect(0,400, 400,200), Temporary_TestTex1.id);
    if (KeysPressed[SDL_SCANCODE_F1])
    {
        for (FaceBatch& batch : GameLevelFaceBatches)
            batch.sharedTexture = Temporary_LightMapVisualizeTex;
    }
    if (KeysPressed[SDL_SCANCODE_F2])
    {
        for (FaceBatch& batch : GameLevelFaceBatches)
            batch.sharedTexture = Temporary_TestTex0;
    }

    // ENEMY 0
    vec3 toPlayer = PlayerControllerRoot - Enemy0.root;
    Enemy0.facing = Normalize(vec3(toPlayer.x, 0.f, toPlayer.z));
    // enemy0.root += enemy0.facing * 32.f * g_DeltaTime;

    // CALCULATE PLAYER FACING DIRECTION
    float camYawDelta = MouseDelta.x * 0.055f;
    float camPitchDelta = MouseDelta.y * 0.055f;
    CameraRotation.y -= camYawDelta;
    CameraRotation.z -= camPitchDelta;
    if (CameraRotation.z > 89.f)
        CameraRotation.z = 89.f;
    if (CameraRotation.z < -89.f)
        CameraRotation.z = -89.f;
    CameraDirection = Normalize(OrientationToDirection(EulerToQuat(CameraRotation * GM_DEG2RAD)));
    CameraRight = Normalize(Cross(CameraDirection, GM_UP_VECTOR));
    CameraUp = Normalize(Cross(CameraRight, CameraDirection));
    PlayerWalkDirectionRight = CameraRight;
    PlayerWalkDirectionForward = Normalize(Cross(GM_UP_VECTOR, PlayerWalkDirectionRight));

    // temp crosshair
    ivec2 guiwh = ivec2(RenderTargetGUI.width, RenderTargetGUI.height);
    Gui::PrimitivePanel(Gui::UIRect(guiwh.x/2 - 3, guiwh.y/2 - 3, 6, 6), vec4(0,0,0,1));
    Gui::PrimitivePanel(Gui::UIRect(guiwh.x/2 - 2, guiwh.y/2 - 2, 4, 4), vec4(1,1,1,1));


    // PLAYER MOVE
    float moveSpeed = 250.f;
    vec3 playerPositionDelta;
    if (KeysCurrent[SDL_SCANCODE_W])
        playerPositionDelta += PlayerWalkDirectionForward * moveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_A])
        playerPositionDelta += -PlayerWalkDirectionRight * moveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_S])
        playerPositionDelta += -PlayerWalkDirectionForward * moveSpeed * DeltaTime;
    if (KeysCurrent[SDL_SCANCODE_D])
        playerPositionDelta += PlayerWalkDirectionRight * moveSpeed * DeltaTime;
    PlayerControllerRoot += playerPositionDelta;
    
    static int channelrotationtesting = 0;
    if (KeysPressed[SDL_SCANCODE_SPACE])
    {
        PlayerYVel = 160.f;
        Mix_VolumeChunk(sfx_Jump, 48);
//        Mix_PlayChannel(channelrotationtesting++%3, sfx_Jump, 0);
    }
    PlayerYVel -= 314.f * DeltaTime; // 9.81 m/s * 32 units/m ~= 314 units/s
    PlayerControllerRoot.y += PlayerYVel * DeltaTime;

    // PLAYER RESOLVE COLLISIONS
    MeshCollider *pcol = PlayerColliderAtPos(PlayerControllerRoot);
    Bounds playerBounds = BoundsFromMeshCollider(*pcol);
    std::unordered_set<Collider*> pcollisionquery = GameLevelCollisionTree.Query(playerBounds);
    // NOTE(Kevin): If I wanted, I could do something like sort these potential colliders by walls before floors or something
    for (Collider *levelCollider : pcollisionquery)
    {
        CollisionResult result = GJK(levelCollider, pcol);
        if (result.hasCollision)
        {
            vec3 moveby = result.normal * result.penetrationDepth;
            PlayerControllerRoot += moveby;
            if (Dot(result.normal, GM_UP_VECTOR) > 0.1f)
                PlayerYVel = 0.f;
        }
    }

    // PLAYER CAMERA 
    vec3 cameraPosOffsetFromRoot = vec3(0,40,0);
    vec3 cameraPosition = PlayerControllerRoot + cameraPosOffsetFromRoot;
    // LogMessage("pos %f, %f, %f", playerControllerRoot.x, playerControllerRoot.y, playerControllerRoot.z);
    // LogMessage("dir y %f, z %f\n", cameraRotation.y, cameraRotation.z);

    static float camLean = 0.f;
    static float desiredCamLean = 0.f;
    const float camLeanSpeed = 15;
    const float maxCamLean = 0.07f;
    desiredCamLean = 0.f;
    if(KeysCurrent[SDL_SCANCODE_D])
        desiredCamLean += maxCamLean;
    if(KeysCurrent[SDL_SCANCODE_A])
        desiredCamLean += -maxCamLean;
    camLean = Lerp(camLean, desiredCamLean, DeltaTime * camLeanSpeed);

    quat fromto = RotationFromTo(CameraUp, CameraRight);
    quat sle = Slerp(quat(), fromto, camLean);
    vec3 cameraUpWithSway = RotateVector(CameraUp, sle);
    float dot = Dot(Normalize(Cross(cameraUpWithSway, CameraRight)), CameraDirection);
    if (dot < 0.99f)
        printf("bad cam up %f\n", dot);
    ActiveViewMatrix = ViewMatrixLookAt(cameraPosition, cameraPosition + CameraDirection, cameraUpWithSway);
}

static void RenderGameLayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGame.fbo);
    glViewport(0, 0, RenderTargetGame.width, RenderTargetGame.height);
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //RGBHEXTO1(0x6495ed), 1.f);//(RGB255TO1(211, 203, 190), 1.f);//(0.674f, 0.847f, 1.0f, 1.f); //RGB255TO1(46, 88, 120)
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    float aspectratio = float(BackbufferWidth) / float(BackbufferHeight);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    ActivePerspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = ActivePerspectiveMatrix;
    mat4 viewMatrix = ActiveViewMatrix;

    UseShader(GameLevelShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(GameLevelShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(GameLevelShader, "viewMatrix", 1, viewMatrix.ptr());

    for (size_t i = 0; i < GameLevelFaceBatches.size(); ++i)
    {
        FaceBatch fb = GameLevelFaceBatches.at(i);
        RenderFaceBatch(fb);
    }

    // modelMatrix = TranslationMatrix(enemy0.root) * RotationMatrix(DirectionToOrientation(enemy0.facing));
    // GLBindMatrix4fv(gameLevelShader, "modelMatrix", 1, modelMatrix.ptr());
    // RenderModelGLTF(model_Knight);

    // PRIMITIVES
    if (KeysPressed[SDL_SCANCODE_X])
        DoPrimitivesDepthTest = !DoPrimitivesDepthTest;
    if (DoPrimitivesDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

    // TODO
    //DrawHandlesVertexArray_GL(MyFuckingMesh.data(), MyFuckingMesh.size(), RenderTargetGame.fbo, 
    //    RenderTargetGame.width, RenderTargetGame.height, perspectiveMatrix.ptr(), viewMatrix.ptr());
}

static void RenderMapEditorLayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGame.fbo);
    glViewport(0, 0, RenderTargetGame.width, RenderTargetGame.height);
    glClearColor(0.15f, 0.15f, 0.15f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    float aspectratio = float(BackbufferWidth) / float(BackbufferHeight);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    ActivePerspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = ActivePerspectiveMatrix;
    mat4 viewMatrix = ActiveViewMatrix;

    UseShader(EditorShader_Scene);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(EditorShader_Scene, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(EditorShader_Scene, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, modelMatrix.ptr());

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
            UseShader(EditorShader_FaceSelected);
            float sf = (sinf(TimeSinceStart * 2.7f) + 1.f) / 2.f;
            sf *= 0.1f;
            if (DoPrimitivesDepthTest)
                sf = 0.0f;
            GLBind3f(EditorShader_FaceSelected, "tint", 1.0f, 1.0f - sf, 1.0f - sf);
            GLBindMatrix4fv(EditorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
            GLBindMatrix4fv(EditorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
            modelMatrix = mat4();
            GLBindMatrix4fv(EditorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
            RenderFaceBatch(selectedFaceMesh);
        }
        else if (hoveredFaceMesh.idVAO > 0)
        {
            UseShader(EditorShader_FaceSelected);
            GLBind3f(EditorShader_FaceSelected, "tint", 0.9f, 0.9f, 0.0f);
            GLBindMatrix4fv(EditorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
            GLBindMatrix4fv(EditorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
            modelMatrix = mat4();
            GLBindMatrix4fv(EditorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
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
    if (KeysPressed[SDL_SCANCODE_X])
        DoPrimitivesDepthTest = !DoPrimitivesDepthTest;
    if (DoPrimitivesDepthTest)
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
    DrawGrid(GRID_INCREMENT, gridRotation, gridTranslation, &perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));
    PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

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
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGUI.fbo);
    glViewport(0, 0, RenderTargetGUI.width, RenderTargetGUI.height);
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
    UseShader(FinalPassShader);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, BackbufferWidth, BackbufferHeight);
//    glDepthRange(0, 10);
    glClearColor(RGB255TO1(0, 0, 0), 1.f);
//    glClearDepth(1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    // Draw game frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderTargetGame.colorTexId);
    RenderGPUMesh(FinalRenderOutputQuad);

    // Draw GUI frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderTargetGUI.colorTexId);
    RenderGPUMesh(FinalRenderOutputQuad);

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
    glFrontFace(GL_CCW); // OpenGL default is GL_CCW


    SDL_GL_GetDrawableSize(SDLMainWindow, &BackbufferWidth, &BackbufferHeight);
    RenderTargetGame.width = BackbufferWidth;
    RenderTargetGame.height = BackbufferHeight;
    CreateGPUFrameBuffer(&RenderTargetGame);
    RenderTargetGUI.width = BackbufferWidth / 2;
    RenderTargetGUI.height = BackbufferHeight / 2;
    CreateGPUFrameBuffer(&RenderTargetGUI);


    GLLoadShaderProgramFromFile(GameLevelShader, wd_path("__game_level.vert").c_str(), wd_path("__game_level.frag").c_str());
    GLLoadShaderProgramFromFile(PatchesIDShader, wd_path("__patches_id.vert").c_str(), wd_path("__patches_id.frag").c_str());
    GLCreateShaderProgram(EditorShader_Scene, __editor_scene_shader_vs, __editor_scene_shader_fs);
    GLCreateShaderProgram(EditorShader_Wireframe, __editor_scene_wireframe_shader_vs, __editor_scene_wireframe_shader_fs);
    GLCreateShaderProgram(EditorShader_FaceSelected, __editor_shader_face_selected_vs, __editor_shader_face_selected_fs);
    GLCreateShaderProgram(FinalPassShader, __finalpass_shader_vs, __finalpass_shader_fs);


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
    CreateGPUMesh(&FinalRenderOutputQuad, refQuadVertices, refQuadIndices, 16, 6, 2, 2, 0, GL_STATIC_DRAW);

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
    CurrentTime = currentTimeInSeconds;
    UnscaledDeltaTime = deltaTimeInSeconds;
    TimeSinceStart += UnscaledDeltaTime;
    DeltaTime = UnscaledDeltaTime * GameTimeScale;
}

static bool InitializeEverything()
{
    _getcwd(CurrentWorkingDirectory, 128);

#if INTERNAL_BUILD
    // === RENDER DOC API ===
    LoadLibrary("renderdoc.dll");
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&RDOCAPI);
        ASSERT(ret == 1);
        // rdoc_api->SetCaptureFilePathTemplate(...);
        LogMessage("Loaded renderdoc.dll");
    }
    else
    {
        LogWarning("Failed to load renderdoc.dll");
    }
#endif

    ProgramShutdownRequested = false;

    SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "system"); // https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/LearnWin32/dpi-and-device-independent-pixels.md#dwm-scaling
    SDL_SetHint(SDL_HINT_WINDOWS_DPI_SCALING, "0"); // https://github.com/libsdl-org/SDL/commit/ab81a559f43abc0858c96788f8e00bbb352287e8

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) return false;

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDLMainWindow = SDL_CreateWindow("game",
                                     SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                     2560,
                                     1440,
                                       SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    SDLGLContext = SDL_GL_CreateContext(SDLMainWindow);

    if (SDLMainWindow == nullptr || SDLGLContext == nullptr) return false;

#ifdef MESA_USING_GL3W
    if (gl3w_init())
    {
        fprintf(stderr, "Failed to initialize OpenGL\n");
        return false;
    }
#endif

    SDL_SetWindowMinimumSize(SDLMainWindow, 200, 100);
    SDL_GL_SetSwapInterval(0);

    if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0 || Mix_Init(MIX_INIT_OGG) < 0)
        return false;

    Gui::Init();

    return true;
}

static void ProcessSDLEvents()
{
    // MOUSE
    ivec2 prevmp = MousePos;
    u32 mousestate = SDL_GetMouseState(&MousePos.x, &MousePos.y); // mousepos only valid when mouse not relative mode
    u32 mousechanged = MouseCurrent ^ mousestate;
    MousePressed = mousechanged & mousestate;
    MouseReleased = mousechanged & MouseCurrent;
    MouseCurrent = mousestate;
    ivec2 md;
    SDL_GetRelativeMouseState(&md.x, &md.y);
    MouseDelta.x = (float)md.x;
    MouseDelta.y = (float)md.y;

    // KEYBOARD
    const u8 *keystate = SDL_GetKeyboardState(NULL);
    u8 keyschanged[256];
    for (int i=0;i<256;++i)
    {
        keyschanged[i] = KeysCurrent[i] ^ keystate[i];
        KeysPressed[i] = keyschanged[i] & keystate[i];
        KeysReleased[i] = keyschanged[i] & KeysCurrent[i];
        KeysCurrent[i] = keystate[i];
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
                        SDL_GL_GetDrawableSize(SDLMainWindow, &BackbufferWidth, &BackbufferHeight);
                        UpdateGPUFrameBufferSize(&RenderTargetGame, BackbufferWidth, BackbufferHeight);
                        UpdateGPUFrameBufferSize(&RenderTargetGUI, BackbufferWidth / 2, BackbufferHeight / 2);
                        break;
                    }
                }
                break;
            }

            case SDL_QUIT:
            {
                ProgramShutdownRequested = true;
                break;
            }

            case SDL_KEYDOWN:
            {
                SDL_Keycode sdlkey = event.key.keysym.sym;

                if (sdlkey == SDLK_RETURN && SDL_GetModState() & KMOD_LALT)
                {
                    if (SDL_GetWindowFlags(SDLMainWindow) & SDL_WINDOW_FULLSCREEN_DESKTOP)
                        SDL_SetWindowFullscreen(SDLMainWindow, 0);
                    else
                        SDL_SetWindowFullscreen(SDLMainWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);
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

    // RDOCAPI->LaunchReplayUI(1, "");

#if SUNLIGHT_TEST
    EditorDeserializeMap(wd_path("House.emf").c_str());
    // EditorDeserializeMap(wd_path("IrradianceCachingTest.emf").c_str());
#else
    EditorDeserializeMap(wd_path("LightTest.emf").c_str());
#endif
    BuildGameMap(wd_path("buildtest.map").c_str());

    if (LoadGameMap(wd_path("buildtest.map").c_str()) == false)
        LogError("failed to load game map");

    while (!ProgramShutdownRequested)
    {
        TickTime();
        Sleep(6);
        if (DeltaTime > 0.1f) continue; // if delta time is too large, will cause glitches

        Gui::NewFrame();
        ProcessSDLEvents();

        if (RDOCAPI && KeysPressed[SDL_SCANCODE_HOME])
            if (RDOCAPI->ShowReplayUI() == 0)
                RDOCAPI->LaunchReplayUI(1, "");

        TickGame();
        DrawGame();
        // TickMapEditor();
        // DrawMapEditor();

        SDL_GL_SwapWindow(SDLMainWindow);
    }

    CloseMapEditor();

    SDL_DestroyWindow(SDLMainWindow);
    SDL_GL_DeleteContext(SDLGLContext);
    SDL_Quit();
    return 0;
}
