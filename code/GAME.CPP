/*

2024-06-23
Just fucking around stripping out core engine stuff from Mesa/Pip. 
What else can be stripped out reused?
    GUI
    MemoryAllocator
    Input??? my existing version is way too complex for making a basic PC game tbh...    
    Console front and backend

And then what? Make a FPS? Fuck level editor for now just use Blender to make basic playgrounds.

    Port MeshGroup rendering and Assimp loading from Q/PS1-Renderer projects
    Port GJK collision code from Q 
    Port 3D camera code from Q

*/



#include <cstdint>
#include <cassert>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_map>

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

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


#include <gmath.h>





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


SDL_Window *g_MainSDLWindow;
SDL_GLContext g_SDLGLContext;
bool g_ProgramShouldShutdown = false;
float g_DeltaTime = 0.f;
float g_UnscaledDeltaTime = 0.f;
float g_GameTimeScale = 1.f;
float g_CurrentTime = 0.f;
float g_TimeSinceStart = 0.f;


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






#include "UTILITY.CPP"
#include "SHADERS.CPP"





struct GPUFrameBuffer
{
    u32 fbo;
    u32 colorTexId;
    u32 depthRBO;
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

static void CreateGPUFrameBuffer(GPUFrameBuffer *buffer)
{
    buffer->fbo = 0;
    glGenFramebuffers(1, &buffer->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, buffer->fbo);

    glGenTextures(1, &buffer->colorTexId);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, buffer->width, buffer->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenRenderbuffers(1, &buffer->depthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer->depthRBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer->colorTexId, 0);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, buffer->width, buffer->height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer->depthRBO);

    ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

static void UpdateGPUFrameBufferSize(GPUFrameBuffer *buffer, i32 w, i32 h)
{
    if (buffer->width == w && buffer->height == h) return;

    buffer->width = w;
    buffer->height = h;
    glBindFramebuffer(GL_FRAMEBUFFER, buffer->fbo);
    glBindTexture(GL_TEXTURE_2D, buffer->colorTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, buffer->width, buffer->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer->depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, buffer->width, buffer->height);
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
                                       GLenum            filter_mode = GL_NEAREST)
{
    ASSERT(texture->id == 0);

    texture->width = bitmap_width;
    texture->height = bitmap_height;
    texture->format = source_format;

    glGenTextures(1, &texture->id);   // generate texture and grab texture id
    glBindTexture(GL_TEXTURE_2D, texture->id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode); // filtering (e.g. GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode);
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

    bool bUseNearest = true; // TODO get from somewhere else
    CreateGPUTextureFromBitmap(texture,
                               (unsigned char*) textureBitmapHandle.memory,
                               textureBitmapHandle.width,textureBitmapHandle.height,
                               targetFormat,
                               (textureBitmapHandle.bitDepth == 3 ? GL_RGB : GL_RGBA),
                               (bUseNearest ? GL_NEAREST : GL_LINEAR_MIPMAP_LINEAR));
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

MeshGroup MeshGroupLoad(const char* fileName)
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

void MeshGroupDelete(MeshGroup *modelToDelete)
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



MeshGroup mg_Cube;
GPUTexture tex_Default;


GPUShader sceneShader;
GPUShader finalPassShader;
GPUFrameBuffer renderTargetGame;
GPUFrameBuffer renderTargetGUI;
GPUMesh __final_render_output_quad;
i32 backBufferWidth = -1;
i32 backBufferHeight = -1;

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

    UseShader(sceneShader);

    glEnable(GL_CULL_FACE);

    mat4 perspectiveMatrix = ProjectionMatrixPerspective((90.f/2.f)*GM_DEG2RAD, 
        float(backBufferWidth)/float(backBufferHeight), 0.1f, 256000.f);
    mat4 viewMatrix = ViewMatrixLookAt(vec3(4000,500,4000), vec3(0,500,0), vec3(0,1,0));

    GLBindMatrix4fv(sceneShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(sceneShader, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(sceneShader, "modelMatrix", 1, modelMatrix.ptr());
    GLActiveTextureUnit0(tex_Default.id);
    MeshGroupRender(&mg_Cube);
}

static void RenderGUILayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, renderTargetGUI.fbo);
    glViewport(0, 0, renderTargetGUI.width, renderTargetGUI.height);
    glDepthRange(0.00001f, 10.f);
    glClearColor(RGB255TO1(244, 194, 194), 0.0f);
    glClearDepth(10.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    // Gui::Draw();
}

static void FinalRenderToBackBuffer()
{
    UseShader(finalPassShader);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, backBufferWidth, backBufferHeight);
    glDepthRange(0, 10);
    glClearColor(RGB255TO1(0, 0, 0), 1.f);
    glClearDepth(1.f);
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


    SDL_GL_GetDrawableSize(g_MainSDLWindow, &backBufferWidth, &backBufferHeight);
    renderTargetGame.width = backBufferWidth;
    renderTargetGame.height = backBufferHeight;
    CreateGPUFrameBuffer(&renderTargetGame);
    renderTargetGUI.width = backBufferWidth;
    renderTargetGUI.height = backBufferHeight;
    CreateGPUFrameBuffer(&renderTargetGUI);


    GLCreateShaderProgram(sceneShader, __scene_shader_vs, __scene_shader_fs);
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
}

static void DrawGameRenderer()
{
    RenderGameLayer();
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
                                       1920,
                                       1080,
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
    //SDL_SetWindowFullscreen(g_MainSDLWindow, SDL_WINDOW_FULLSCREEN);

    return true;
}

static void ProcessSDLEvents()
{
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
                        SDL_GL_GetDrawableSize(g_MainSDLWindow, &backBufferWidth, &backBufferHeight);
                        UpdateGPUFrameBufferSize(&renderTargetGame, backBufferWidth, backBufferHeight);
                        UpdateGPUFrameBufferSize(&renderTargetGUI, backBufferWidth, backBufferHeight);
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
                if (event.key.keysym.sym == SDLK_RETURN && SDL_GetModState() & KMOD_LALT)
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

        // Send SDL events to other systems
        // Input.ProcessSDLEvent(event);
        // Gui::ProcessSDLEvent(event);
    }
}

int main(int argc, char* argv[])
{
    if (!InitializeEverything()) return -1;

    InitGameRenderer();

    CreateGPUTextureFromDisk(&tex_Default, wd_path("default.png").c_str());
    mg_Cube = MeshGroupLoad(wd_path("testscene.obj").c_str());

    while (!g_ProgramShouldShutdown)
    {
        TickTime();
        if (g_DeltaTime > 0.1f) continue; // if delta time is too large, will cause glitches

        // Gui::NewFrame();
        ProcessSDLEvents();

        DrawGameRenderer();

        SDL_GL_SwapWindow(g_MainSDLWindow);

        // Input.ResetInputStatesAtEndOfFrame();
    }

    SDL_DestroyWindow(g_MainSDLWindow);
    SDL_GL_DeleteContext(g_SDLGLContext);
    SDL_Quit();
    return 0;
}
