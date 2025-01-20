/*

Refactor will never be perfect. I just need to get things to a point where I find it clean enough and easy enough to come back to in the future. Organized enough.  

2025-01-14
Okay back to it. Just use Jolt for physics/collisions. Refactor/reorganize so code isn't a mess.
Jolt should handle all collision resolution. We don't need to do any of that bullshit.

todo:
- rn gpumesh to gpumeshindexed; add gpumesh type which is not instanced
- rm facebatch (editor) and rename gamefacebatch to facebatch; for editor, just use gpumesh







World - I want a crispy fucking visually nice to look at world being created and loaded in game

    Refine Hemicube GI
    - larger hemicube atlas for download
    - interpolate (w/o irrad caching) for patches that clip backface to reduce artifacts
    - Texels too far from face polygon should be marked as ignore for hemicube. worst case distance: magnitude(1.5*texelsize, 0.5*texelsize)

    Refine FaceBatch rendering and lighting code 

    Face creation and manipulation by default over cuboid voluesm in level editor
    - volume is wasteful; source 2 defaults to faces; best workflow is extrude faces and extrude edges

Physics - I want a crispy bug-free physics simulation with the world

    Hook player controller into Jolt. Move around world simulated via Jolt.

THEN, we have a fucking engine. A fucking starting point for the game.

View-space projected blood decals

Get some things to shoot on the screen

Skeletal animations



Something fucking cool.
Priorty #1 is building the game, not the engine/tech
Handcrafted with love.



TODO:
- Physics
- Skeletal animations
- Enemy moves and shoots at player
- Gun model?
- Player shoots at enemy
- serialize/deserialize asset database
- move volume mode/tool (don't need whole tool...maybe make brush tool part of this mode? like trenchbroom)
- multi select vertices need to be fixed
- Edge select
- Edge loop
- Toggle element translation along axis
- Anisotropic filtering https://www.khronos.org/opengl/wiki/Sampler_Object#Anisotropic_filtering
- Use a triangulation library for face triangulation https://github.com/artem-ogre/CDT

Engine work is almost done. 
~~- Lightmap baking (potentially radiosity)~~
- Skeletal animations
- Navmesh/pathfinding
- Physics

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



*/
#define SUNLIGHT_TEST 0

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


// let 1 unit = 1 inch, this approximates 32 units to 0.82 metres
#define STANDARD_LENGTH_IN_GAME_UNITS 32
// sqrt(32^2 + 32^2) = 45.254833996 ~= 45
#define STANDARD_LENGTH_DIAGONAL 45
#define THIRTYTWO STANDARD_LENGTH_IN_GAME_UNITS



#include "resources.h"
#include "physics.h"
#include "shaders.h"



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



// Editor code
// - geometry modelling
// - geometry rendering
// - editor UI
// - editor picking / utility functions
// - serialize / deserialize map file
// - build / lightmap baking
// - Asset DB



// enum batch_mesh_attribs : u16
// {
//     BATCHMESH_NO_ATTRIBS = 0 << 0,
//     BATCHMESH_USE_UV2 = 1 << 0
//     // BATCHMESH_USE_UV3,
// };

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

    // batch_mesh_attribs MeshAttribs;
};

struct GameFaceBatch
{
    // world pos x y z, normal i j k, uv1 u v, uv2 u v
    GPUTexture ColorTexture;
    GPUTexture LightMapTexture;
    u32 VAO = 0;
    u32 VBO = 0;
    u32 VerticesCount = 0;
};

void CreateFaceBatchMeshOnGPU(u32 *idVAO, u32 *idVBO)
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

void CreateGameFaceBatchMeshOnGPU(u32 *VAO, u32 *VBO)
{
    glGenVertexArrays(1, VAO);
    glBindVertexArray(*VAO);
    glGenBuffers(1, VBO);
    glBindBuffer(GL_ARRAY_BUFFER, *VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    // pos x y z
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 10, nullptr);
    glEnableVertexAttribArray(0);
    // norm i j k
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 10, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);
    // uv1 u v
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 10, (void*)(sizeof(float) * (3 + 3)));
    glEnableVertexAttribArray(2);
    // uv2 u v
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 10, (void*)(sizeof(float) * (3 + 3 + 2)));
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);
}

void DeleteFaceBatchMeshOnGPU(u32 idVAO, u32 idVBO)
{
    glDeleteBuffers(1, &idVBO);
    glDeleteVertexArrays(1, &idVAO);
}

void RebindFaceBatchBufferObject(FaceBatch *faceBatch, u32 sizeInBytes, float *data)
{
    glBindVertexArray(faceBatch->idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, faceBatch->idVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeInBytes, data, GL_DYNAMIC_DRAW);
    faceBatch->verticesCount = sizeInBytes / (sizeof(float) * 8);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void RebindGameFaceBatchBufferObject(GameFaceBatch *faceBatch, u32 sizeInBytes, float *data)
{
    glBindVertexArray(faceBatch->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, faceBatch->VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeInBytes, data, GL_DYNAMIC_DRAW);
    faceBatch->VerticesCount = sizeInBytes / (sizeof(float) * 10);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // GLHasErrors();
}

void RenderFaceBatch(FaceBatch faceBatch)
{
    glBindVertexArray(faceBatch.idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, faceBatch.idVBO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, faceBatch.sharedTexture.id);
    glDrawArrays(GL_TRIANGLES, 0, faceBatch.verticesCount);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void RenderGameFaceBatch(const GPUShader shader, GameFaceBatch Batch)
{
    glBindVertexArray(Batch.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, Batch.VBO);
    
    GLBind1i(shader, "ColorTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Batch.ColorTexture.id);

    GLBind1i(shader, "LightMap", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, Batch.LightMapTexture.id);

    glDrawArrays(GL_TRIANGLES, 0, Batch.VerticesCount);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLHasErrors();
}

#include "UTILITY.CPP"
#include "COLLISION.CPP"


GPUShader GameLevelShader;
GPUShader PatchesIDShader;
GPUShader EditorShader_Scene;
GPUShader EditorShader_Wireframe;
GPUShader EditorShader_FaceSelected;
GPUShader FinalPassShader;
GPUFrameBuffer RenderTargetGame;
GPUFrameBuffer RenderTargetGUI;
GPUMeshIndexed FinalRenderOutputQuad;
float GAMEPROJECTION_NEARCLIP = 4.f; // even 2 works fine to remove z fighting
float GAMEPROJECTION_FARCLIP = 32000.f;
bool DoPrimitivesDepthTest = false;



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
std::vector<GameFaceBatch> GameLevelFaceBatches;

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

#include "physics.cpp"
#include "resources.cpp"
#include "shaders.cpp"

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

// Create mapping table from object layer to broadphase layer
BPLayerInterfaceImpl broad_phase_layer_interface;
// Create class that filters object vs broadphase layers
ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
// Create class that filters object vs object layers
ObjectLayerPairFilterImpl object_vs_object_layer_filter;
// Now we can create the actual physics system.
JPH::PhysicsSystem physics_system;
JPH::BodyInterface *body_interface;
JPH::Character *mCharacter;

static void Stuff2()
{
    // Lets make the fucking level mesh collider bitch

    JPH::RegisterDefaultAllocator();
    JPH::Trace = TraceImpl;
    JPH_IF_ENABLE_ASSERTS(AssertFailed = AssertFailedImpl;)
    JPH::Factory::sInstance = new Factory();
    JPH::RegisterTypes();

    const uint cMaxBodies = 65536;
    const uint cNumBodyMutexes = 0;
    const uint cMaxBodyPairs = 65536;
    const uint cMaxContactConstraints = 10240;

    physics_system.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
    physics_system.SetGravity(physics_system.GetGravity() * 32.f);

    // A body activation listener gets notified when bodies activate and go to sleep
    // Note that this is called from a job so whatever you do here needs to be thread safe.
    // Registering one is entirely optional.
    static MyBodyActivationListener body_activation_listener;
    physics_system.SetBodyActivationListener(&body_activation_listener);

    // A contact listener gets notified when bodies (are about to) collide, and when they separate again.
    // Note that this is called from a job so whatever you do here needs to be thread safe.
    // Registering one is entirely optional.
    static MyContactListener contact_listener;
    physics_system.SetContactListener(&contact_listener);

    // The main way to interact with the bodies in the physics system is through the body interface. There is a locking and a non-locking
    // variant of this. We're going to use the locking version (even though we're not planning to access bodies from multiple threads)
    body_interface = &physics_system.GetBodyInterface();

    // // Next we can create a rigid body to serve as the floor, we make a large box
    // // Create the settings for the collision volume (the shape).
    // // Note that for simple shapes (like boxes) you can also directly construct a BoxShape.
    // BoxShapeSettings floor_shape_settings(Vec3(100.0f, 1.0f, 100.0f));
    // floor_shape_settings.SetEmbedded(); // A ref counted object on the stack (base class RefTarget) should be marked as such to prevent it from being freed when its reference count goes to 0.

    static JPH::TriangleList triangles;
    for (FlatPolygonCollider& collider : GameLevelColliders)
    {
        vec3 first = collider.pointCloudPtr[0];
        for (u32 i = 2; i < collider.pointCount; ++i)
        {
            vec3 second = collider.pointCloudPtr[i-1];
            vec3 third = collider.pointCloudPtr[i];

            JPH::Float3 jph_first(first.x, first.y, first.z);
            JPH::Float3 jph_second(second.x, second.y, second.z);
            JPH::Float3 jph_third(third.x, third.y, third.z);

            triangles.push_back(Triangle(jph_first, jph_second, jph_third));
        }
    }
    static JPH::MeshShapeSettings myLevelColliderSettings = JPH::MeshShapeSettings(triangles);
    myLevelColliderSettings.SetEmbedded();

    // Create the shape
    static JPH::ShapeSettings::ShapeResult level_shape_result = myLevelColliderSettings.Create();
    static JPH::ShapeRefC level_shape = level_shape_result.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()
    if(level_shape_result.HasError())
    {
        LogMessage("%s", level_shape_result.GetError().c_str());
    }

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    static JPH::BodyCreationSettings level_settings(level_shape, RVec3(0.0_r, 0.0_r, 0.0_r), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
    level_settings.mEnhancedInternalEdgeRemoval = true;

    // Create the actual rigid body
    Body *floor = body_interface->CreateBody(level_settings); // Note that if we run out of bodies this can return nullptr

    // Add it to the world
    body_interface->AddBody(floor->GetID(), EActivation::DontActivate);

    // // Now create a dynamic body to bounce on the floor
    // // Note that this uses the shorthand version of creating and adding a body to the world
    // BodyCreationSettings sphere_settings(new SphereShape(8.f), RVec3(0.0_r, 64.0_r, 0.0_r), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
    // BodyID sphere_id = body_interface->CreateAndAddBody(sphere_settings, EActivation::Activate);

    // // Now you can interact with the dynamic body, in this case we're going to give it a velocity.
    // // (note that if we had used CreateBody then we could have set the velocity straight on the body before adding it to the physics system)
    // body_interface->SetLinearVelocity(sphere_id, Vec3(0.0f, 0.0f, 0.0f));

    // We simulate the physics world in discrete time steps. 60 Hz is a good rate to update the physics system.


    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    physics_system.OptimizeBroadPhase();

    // // Now we're ready to simulate the body, keep simulating until it goes to sleep
    // uint step = 0;
    // while (body_interface->IsActive(sphere_id))
    // {
    //     // Next step
    //     ++step;

    //     // Output current position and velocity of the sphere
    //     RVec3 position = body_interface->GetCenterOfMassPosition(sphere_id);
    //     Vec3 velocity = body_interface->GetLinearVelocity(sphere_id);
    //     cout << "Step " << step << ": Position = (" << position.GetX() << ", " << position.GetY() << ", " << position.GetZ() << "), Velocity = (" << velocity.GetX() << ", " << velocity.GetY() << ", " << velocity.GetZ() << ")" << endl;

    //     // If you take larger steps than 1 / 60th of a second you need to do multiple collision steps in order to keep the simulation stable. Do 1 collision step per 1 / 60th of a second (round up).
    //     const float cDeltaTime = 1.0f / 60.0f;
    //     const int cCollisionSteps = 1;

    //     // Step the world
    //     physics_system.Update(cDeltaTime, cCollisionSteps, &temp_allocator, &job_system);
    // }

    // // Remove the sphere from the physics system. Note that the sphere itself keeps all of its state and can be re-added at any time.
    // body_interface.RemoveBody(sphere_id);

    // // Destroy the sphere. After this the sphere ID is no longer valid.
    // body_interface.DestroyBody(sphere_id);

    // // Remove and destroy the floor
    // body_interface.RemoveBody(floor->GetID());
    // body_interface.DestroyBody(floor->GetID());

    // // Unregisters all types with the factory and cleans up the default material
    // UnregisterTypes();

    // // Destroy the factory
    // delete Factory::sInstance;
    // Factory::sInstance = nullptr;




    // make the character collider
    static constexpr float  cCharacterHeightStanding = 48.f;
    static constexpr float  cCharacterRadiusStanding = 8.f;
    RefConst<Shape> mStandingShape = RotatedTranslatedShapeSettings(Vec3(0, 0.5f * cCharacterHeightStanding + cCharacterRadiusStanding, 0), Quat::sIdentity(), new CapsuleShape(0.5f * cCharacterHeightStanding, cCharacterRadiusStanding)).Create().Get();

    Ref<CharacterSettings> settings = new CharacterSettings();
    settings->mMaxSlopeAngle = DegreesToRadians(45.0f);
    settings->mLayer = Layers::MOVING;
    settings->mShape = mStandingShape;
    settings->mFriction = 0.5f;
    settings->mSupportingVolume = Plane(Vec3::sAxisY(), -cCharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    mCharacter = new Character(settings, JPH::RVec3(0,32,0), Quat::sIdentity(), 0, &physics_system);
    mCharacter->AddToPhysicsSystem(EActivation::Activate);
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
        for (GameFaceBatch& batch : GameLevelFaceBatches)
            batch.LightMapTexture = Temporary_LightMapVisualizeTex;
    }
    if (KeysPressed[SDL_SCANCODE_F2])
    {
        for (GameFaceBatch& batch : GameLevelFaceBatches)
            batch.LightMapTexture = Temporary_TestTex0;
    }

    // ENEMY 0
    vec3 toPlayer = PlayerControllerRoot - Enemy0.root;
    Enemy0.facing = Normalize(vec3(toPlayer.x, 0.f, toPlayer.z));
    // enemy0.root += enemy0.facing * 32.f * g_DeltaTime;

    // CALCULATE PLAYER FACING DIRECTION
    float camYawDelta = MouseDelta.x * 0.085f;
    float camPitchDelta = MouseDelta.y * 0.085f;
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
    // float moveSpeed = 250.f;
    vec3 desiredMoveDirection;
    if (KeysCurrent[SDL_SCANCODE_W])
        desiredMoveDirection += PlayerWalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_A])
        desiredMoveDirection += -PlayerWalkDirectionRight;
    if (KeysCurrent[SDL_SCANCODE_S])
        desiredMoveDirection += -PlayerWalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_D])
        desiredMoveDirection += PlayerWalkDirectionRight;
    
    static int channelrotationtesting = 0;
    if (KeysPressed[SDL_SCANCODE_SPACE])
    {
        PlayerYVel = 160.f;
        Mix_VolumeChunk(sfx_Jump, 48);
//        Mix_PlayChannel(channelrotationtesting++%3, sfx_Jump, 0);
    }
    PlayerYVel -= 314.f * DeltaTime; // 9.81 m/s * 32 units/m ~= 314 units/s
    PlayerControllerRoot.y += PlayerYVel * DeltaTime;



    // Cancel movement in opposite direction of normal when touching something we can't walk up
    Vec3 movement_direction = Vec3(desiredMoveDirection.x, desiredMoveDirection.y, desiredMoveDirection.z);
    Character::EGroundState ground_state = mCharacter->GetGroundState();
    if (ground_state == Character::EGroundState::OnSteepGround || ground_state == Character::EGroundState::NotSupported)
    {
        Vec3 normal = mCharacter->GetGroundNormal();
        normal.SetY(0.0f);
        float dot = normal.Dot(movement_direction);
        if (dot < 0.0f)
            movement_direction -= (dot * normal) / normal.LengthSq();
    }

    //// Stance switch
    //if (inSwitchStance)
    //    mCharacter->SetShape(mCharacter->GetShape() == mStandingShape ? mCrouchingShape : mStandingShape, 1.5f * mPhysicsSystem->GetPhysicsSettings().mPenetrationSlop);
    const float		sCharacterSpeed = 6.0f*32.f;
    const float		sJumpSpeed = 4.0f*32.f;
    if (/*sControlMovementDuringJump || */ mCharacter->IsSupported())
    {
        // Update velocity
        Vec3 current_velocity = mCharacter->GetLinearVelocity();
        // try printing magnitude of current velocity
        Vec3 desired_velocity = sCharacterSpeed * movement_direction;
        if (!desired_velocity.IsNearZero() || current_velocity.GetY() < 0.0f || !mCharacter->IsSupported())
            desired_velocity.SetY(current_velocity.GetY());
        Vec3 new_velocity = 0.75f * current_velocity + 0.25f * desired_velocity;

        // Jump
        if (KeysPressed[SDL_SCANCODE_SPACE] && ground_state == Character::EGroundState::OnGround)
            new_velocity += Vec3(0, sJumpSpeed, 0);

        // Update the velocity
        mCharacter->SetLinearVelocity(new_velocity);
    }

    // PLAYER RESOLVE COLLISIONS

    static TempAllocatorImpl temp_allocator(10 * 1024 * 1024);
    static JPH::JobSystemThreadPool job_system(cMaxPhysicsJobs, cMaxPhysicsBarriers, thread::hardware_concurrency() - 1);

    const float cDeltaTime = 1.0f / 60.0f;
    const int cCollisionSteps = 1;
    // Step the world
    physics_system.Update(cDeltaTime, cCollisionSteps, &temp_allocator, &job_system);
    mCharacter->PostSimulation(0.05f);

    RVec3 cpos = mCharacter->GetPosition();
    LogMessage("character pos %f, %f, %f", cpos.GetX(), cpos.GetY(), cpos.GetZ());

    PlayerControllerRoot.x = cpos.GetX();
    PlayerControllerRoot.y = cpos.GetY();
    PlayerControllerRoot.z = cpos.GetZ();

    // MeshCollider *pcol = PlayerColliderAtPos(PlayerControllerRoot);
    // Bounds playerBounds = BoundsFromMeshCollider(*pcol);
    // std::unordered_set<Collider*> pcollisionquery = GameLevelCollisionTree.Query(playerBounds);
    // // NOTE(Kevin): If I wanted, I could do something like sort these potential colliders by walls before floors or something
    // for (Collider *levelCollider : pcollisionquery)
    // {
    //     CollisionResult result = GJK(levelCollider, pcol);
    //     if (result.hasCollision)
    //     {
    //         vec3 moveby = result.normal * result.penetrationDepth;
    //         PlayerControllerRoot += moveby;
    //         if (Dot(result.normal, GM_UP_VECTOR) > 0.1f)
    //             PlayerYVel = 0.f;
    //     }
    // }

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
        GameFaceBatch fb = GameLevelFaceBatches.at(i);
        RenderGameFaceBatch(GameLevelShader, fb);
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
    RenderGPUMeshIndexed(FinalRenderOutputQuad);

    // Draw GUI frame
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderTargetGUI.colorTexId);
    RenderGPUMeshIndexed(FinalRenderOutputQuad);

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
    CreateGPUMeshIndexed(&FinalRenderOutputQuad, refQuadVertices, refQuadIndices, 16, 6, 2, 2, 0, GL_STATIC_DRAW);

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
    SDL_GL_SetSwapInterval(1);

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
    // HelloWorld();


    if (!InitializeEverything()) return -1;

    InitGameRenderer();

    Stuff();

    LoadMapEditor();

    // RDOCAPI->LaunchReplayUI(1, "");

// #if SUNLIGHT_TEST
//     EditorDeserializeMap(wd_path("House.emf").c_str());
//     // EditorDeserializeMap(wd_path("IrradianceCachingTest.emf").c_str());
// #else
//     EditorDeserializeMap(wd_path("LightTest.emf").c_str());
// #endif
//     BuildGameMap(wd_path("buildtest.map").c_str());

    if (LoadGameMap(wd_path("buildtest.map").c_str()) == false)
        LogError("failed to load game map");

    Stuff2();

    while (!ProgramShutdownRequested)
    {
        TickTime();

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
        glFinish();
    }

    CloseMapEditor();

    SDL_DestroyWindow(SDLMainWindow);
    SDL_GL_DeleteContext(SDLGLContext);
    SDL_Quit();
    return 0;
}
