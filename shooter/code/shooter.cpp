#include "shooter.h"
#include "shooter_renderer.h"
#include "shooter_shared.h"
#include "shooter_gui.h"
#include "shooter_gui.cpp"
#include "shooter_colliders.cpp"
#define STB_SPRINTF_IMPLEMENTATION
#include <stb_sprintf.h>
#include "shooter_meshes.h"
#include "shooter_random.h"
#include "shooter_gui++.cpp"
#if SHOOTER_COMPILE_WITH_EDITOR
#include "shooter_editor.cpp"
#endif

internal void
GameOutputSound(game_state *GameState, game_sound_output_buffer *SoundBuffer, int ToneHz)
{
    int16 ToneVolume = 4000;
    int WavePeriod = SoundBuffer->SamplesPerSecond / ToneHz; // samples per cycle

    int16 *SampleOut = (int16 *)SoundBuffer->Samples;
    for (int SampleIndex = 0;
        SampleIndex < SoundBuffer->SampleCount;
        ++SampleIndex)
    {
#if 0
        real32 SineValue = sinf(GameState->tSine);
        int16 SampleValue = (int16)(SineValue * ToneVolume);
#else
        int16 SampleValue = 0;
#endif
        *SampleOut++ = SampleValue;
        *SampleOut++ = SampleValue;
#if 0
        GameState->tSine += 2.0f * Pi32 * 1.0f/(real32)WavePeriod;
        if (GameState->tSine > 2.0f * Pi32)
        {
            GameState->tSine -= 2.0f * Pi32;
        }
#endif
    }
}

#pragma pack(push, 1)
struct bitmap_header
{
    uint16  FileType;
    uint32  FileSize;
    uint16  Reserved1;
    uint16  Reserved2;
    uint32  BitmapOffset;
    uint32  Size;
    int32   Width;
    int32   Height;
    uint16  Planes;
    uint16  BitsPerPixel;
    uint32  Compression;
    uint32  SizeOfBitmap;
    int32   HorzResolution;
    int32   VertResolution;
    uint32  ColorsUsed;
    uint32  ColorsImportant;
    uint32  RedMask;
    uint32  GreenMask;
    uint32  BlueMask;
};
#pragma pack(pop)

internal loaded_bitmap
DEBUGLoadBMP(thread_context *Thread, debug_platform_read_entire_file *ReadEntireFile, char *FileName)
{
    loaded_bitmap Result = {};

    // NOTE(Kevin): Casey's bitmaps are saved as AA BB GG RR, bottom up.
    // In little endian -> 0xRRGGBBAA
    // ACTUALLY, his hero bitmaps are BB GG RR AA...
    debug_read_file_result ReadResult = ReadEntireFile(Thread, FileName);

    if (ReadResult.ContentsSize != 0)
    {
        bitmap_header *Header = (bitmap_header *)ReadResult.Contents;
        uint32 *Pixels = (uint32 *)((uint8 *)ReadResult.Contents + Header->BitmapOffset);

        Assert(Header->Compression == 3);

        // NOTE(Kevin): If you are using this generically for some reason,
        // please remember that BMP files can go in either direction and
        // the height can be negative for top-down.
        // (Also, there can be compression, etc... DON'T think this is
        // compelete BMP loading code because it isnt!)

        // NOTE(Kevin): Byte order in memory is determined by the Header itself,
        // so we have to read out the masks and convert the pixels ourselves.
        uint32 RedMask = Header->RedMask;
        uint32 GreenMask = Header->GreenMask;
        uint32 BlueMask = Header->BlueMask;
        uint32 AlphaMask = ~(RedMask | GreenMask | BlueMask);

        bit_scan_result RedScan = FindLeastSignificantSetBit(RedMask);
        bit_scan_result GreenScan = FindLeastSignificantSetBit(GreenMask);
        bit_scan_result BlueScan = FindLeastSignificantSetBit(BlueMask);
        bit_scan_result AlphaScan = FindLeastSignificantSetBit(AlphaMask);

        Assert(RedScan.Found);
        Assert(GreenScan.Found);
        Assert(BlueScan.Found);
        Assert(AlphaScan.Found);

        // In Windows, RGBx is stored as BGRx
        // Pixel in memory: BB GG RR xx
        // Which, following LITTLE ENDIAN, becomes:
        // 0x xx RR GG BB
        int32 AlphaShift = 24 - (int32)AlphaScan.Index;
        int32 RedShift = 16 - (int32)RedScan.Index;
        int32 GreenShift = 8 - (int32)GreenScan.Index;
        int32 BlueShift = 0 - (int32)BlueScan.Index;

        uint32 *SourceDest = Pixels;
        for (int32 Y = 0; Y < Header->Height; ++Y)
        {
            for (int32 X = 0; X < Header->Width; ++X)
            {
                uint32 C = *SourceDest;
                *SourceDest++ = (RotateLeft(C & RedMask, RedShift) |
                                 RotateLeft(C & GreenMask, GreenShift) |
                                 RotateLeft(C & BlueMask, BlueShift) |
                                 RotateLeft(C & AlphaMask, AlphaShift));
            }
        }

        Result.Pixels = Pixels;
        Result.Width = Header->Width;
        Result.Height = Header->Height;
    }

    return Result;
}

internal create_font_result
DEBUGCreateBitmapFont(loaded_bitmap Bitmap, renderer *RendererAPI)
{
    u32 *Pixels = Bitmap.Pixels;
    for (s32 Y = 0; Y < Bitmap.Height; ++Y)
    {
        for (s32 X = 0; X < Bitmap.Width; ++X)
        {
            if (((*Pixels >> 16) & 0xFF) > 0)
            {
                *Pixels = 0xFFFFFFFF;
            }
            else
            {
                *Pixels = 0x00000000;
            }
            ++Pixels;
        }
    }

    create_font_result Result = RendererAPI->DEBUGCreateBitmapFont_Monospace(Bitmap.Pixels, Bitmap.Width, 
                                                                             Bitmap.Height, BGRA_U8, 3);

    return Result;
}

internal void
PlaceCamera(camera *Camera, vec3 Pos, vec3 EulerRot)
{
    Camera->P = Pos;
    Camera->EulerRot = EulerRot;
    Camera->Orientation = QuatFromEuler_YZX(Camera->EulerRot);
    Camera->Forward = Normalize(DirectionFromOrientation(Camera->Orientation));
    Camera->Right = Normalize(Cross(Camera->Forward, GMATH_UP));
    Camera->Up = Normalize(Cross(Camera->Right, Camera->Forward));

    Camera->ViewPosition = Camera->P;
    Camera->ViewForward = Camera->Forward;
    Camera->ViewUp = Camera->Up;
}

internal void
PointCamera(camera *Camera, vec3 Pos, vec3 Forward)
{
    Forward = Normalize(Forward);

    Camera->P = Pos;
    LookRotation2(Forward, &Camera->EulerRot.y, &Camera->EulerRot.z);
    Camera->Orientation = QuatFromEuler_YZX(Camera->EulerRot);
    Camera->Forward = Forward;
    Camera->Right = Normalize(Cross(Camera->Forward, GMATH_UP));
    Camera->Up = Normalize(Cross(Camera->Right, Camera->Forward));

    Camera->ViewPosition = Camera->P;
    Camera->ViewForward = Camera->Forward;
    Camera->ViewUp = Camera->Up;
}

internal void
UpdateCameraStrafeTilt(camera *Camera, vec2 dHorizontalP, real32 MaxHorizontalSpeed, real32 dt)
{
    // Strafe tilt
    if (MaxHorizontalSpeed > 0.0f)
    {
        real32 TiltSpeed = 15.0f;
        real32 MaxTilt = 0.015f;
        real32 DesiredCamTilt = 0.0f;
        vec2 CamRight = Normalize(vec2(Camera->Right.x, Camera->Right.z));

        real32 StrafeMagnitude = Dot(dHorizontalP, CamRight);
        StrafeMagnitude = StrafeMagnitude / MaxHorizontalSpeed;

        DesiredCamTilt = StrafeMagnitude * MaxTilt;
        quat FromTo = RotationFromTo(Camera->ViewUp, Camera->Right);

#if 0
        quat Tilt = Slerp(quat(), FromTo, DesiredCamTilt);
        vec3 CameraUpWithSway = RotateVector(Camera->ViewUp, Tilt);

        Camera->ViewUp = CameraUpWithSway;
#else
        // NOTE(Kevin): There is already gradual acceleration to the tilt from using
        //              the strafe magnitude, but I like this extra tilt speed smoothing.
        //              Makes it feel like I'm speeding up even more than I am.
        //              Actually...can't tell if I'm getting motion sick...
        Camera->StrafeTilt = Blend(Camera->StrafeTilt, DesiredCamTilt, dt * TiltSpeed);
        quat Tilt = Slerp(quat(), FromTo, Camera->StrafeTilt);
        vec3 CameraUpWithSway = RotateVector(Camera->ViewUp, Tilt);

        Camera->ViewUp = CameraUpWithSway;
#endif
    }
}

internal mat4
CameraSpaceFromWorldSpace(camera *Camera)
{
    mat4 ViewMatrix = CameraSpaceMatrixLookAt(Camera->ViewPosition,
                                              Camera->ViewPosition + Camera->ViewForward, 
                                              Camera->ViewUp);
    return ViewMatrix;
}

internal void
MouseLookCamera(camera *Camera, real32 MouseAxisX, real32 MouseAxisY)
{
    real32 PitchConstraint = GMATH_HALFPI-0.0175f;
    real32 YawDelta = MouseAxisX;
    real32 PitchDelta = MouseAxisY;
    Camera->EulerRot.y -= YawDelta;
    Camera->EulerRot.z -= PitchDelta;
    if (Camera->EulerRot.z > PitchConstraint)
        Camera->EulerRot.z = PitchConstraint;
    if (Camera->EulerRot.z < -PitchConstraint)
        Camera->EulerRot.z = -PitchConstraint;

    PlaceCamera(Camera, Camera->P, Camera->EulerRot);
}

inline mat4
TransformMatrix(vec3 P, quat O, vec3 S)
{
    return TranslationMatrix(P) * RotationMatrix(O) * ScaleMatrix(S);
}

internal mesh_collider
GetTransformedEntityCollider(entity *Entity, vec3 P, quat O, vec3 S, memory_arena *Storage)
{
    mat4 Transform = TransformMatrix(P, O, S);

    vec3 *ColliderPoints = (vec3 *)CurrentPtr(Storage);
    for (u32 Index = 0; Index < Entity->ColliderPointsCount; ++Index)
    {
        vec3 *ColliderPoint = PushStruct(Storage, vec3);
        *ColliderPoint = (Transform*vec4(Entity->ColliderPoints[Index],1.0f)).XYZ;
    }

    mesh_collider Result;
    Result.PointCloud = ColliderPoints;
    Result.PointCount = Entity->ColliderPointsCount;
    return Result;
}

struct collision_result
{
    vec3 Normal;
    real32 Depth;
    bool32 HasCollision;
};

internal collision_result
DoGJKRoutine(collider_base *BodyA, collider_base *BodyB)
{
    collision_result CollisionResult = {};

    gjk_simplex CaptureSimplex;

    CollisionResult.HasCollision = GJK(BodyA, BodyB, &CaptureSimplex);

    if (CollisionResult.HasCollision)
    {
        epa_result EPAResult = EPA(BodyA, BodyB, &CaptureSimplex);

        CollisionResult.Normal = EPAResult.NearestFaceNormal;
        CollisionResult.Depth = EPAResult.NearestFaceDist;
    }

    return CollisionResult;
}

internal vec3
ResolveEntityCollisionsWithWorld(game_state *GameState, game_map *Map, vec3 TestP, entity *Entity)
{
    vec3 Result = {};

    quat O = Entity->Orientation;
    vec3 S = Entity->ColliderScale;

    temporary_memory TemporaryColliderMemory = BeginTemporaryMemory(&GameState->FrameArena);

    mesh_collider EntityCollider = GetTransformedEntityCollider(Entity, TestP, O, S, &GameState->FrameArena);

    for (u32 MapColliderIndex = 0; 
         MapColliderIndex < Map->CollidersCount;
         ++MapColliderIndex)
    {
        flatpoly_collider MapFaceCollider = Map->Colliders[MapColliderIndex];

        collision_result CollisionResult = DoGJKRoutine((collider_base *)&MapFaceCollider,
                                                        (collider_base *)&EntityCollider);

        if (CollisionResult.HasCollision)
        {
            vec3 MapFaceP0 = MapFaceCollider.PointCloud[0];
            vec3 MapFaceP1 = MapFaceCollider.PointCloud[1];
            vec3 MapFaceP2 = MapFaceCollider.PointCloud[2];
            vec3 MapFaceN = Normalize(Cross(MapFaceP1-MapFaceP0, MapFaceP2-MapFaceP0));

            vec3 ResolutionVector = CollisionResult.Normal * CollisionResult.Depth;
            bool32 ResolvingSameDirAsFaceNormal = Dot(MapFaceN, CollisionResult.Normal) > 0.001f;
            if (ResolvingSameDirAsFaceNormal)
            {
                // Registered
                Result += ResolutionVector;
                TestP += ResolutionVector;

                EndTemporaryMemory(TemporaryColliderMemory);
                TemporaryColliderMemory = BeginTemporaryMemory(&GameState->FrameArena);
                EntityCollider = GetTransformedEntityCollider(Entity, TestP, O, S, &GameState->FrameArena);
            }
            else
            {
                // Ignored
            }
        }
    }

    EndTemporaryMemory(TemporaryColliderMemory);

    return Result;
}

internal entity_handle
AddEntity(game_state *GameState, bool32 MakeAlive)
{
    entity_handle EntityHandle = 0;

    if (GameState->FreeEntityHandlesCount > 0)
    {
        EntityHandle = GameState->FreeEntityHandles[--GameState->FreeEntityHandlesCount];

        entity *Entity = &(GameState->Entities[EntityHandle]);
        ZeroStruct(*Entity);
        if (MakeAlive)
        {
            Entity->Alive = true;
            Entity->AliveSeconds = 0.0f;
        }
        Entity->Orientation = quat();
        Entity->RenderScale = vec3(1,1,1);
    }

    return EntityHandle;
}

internal entity *
GetEntity(game_state *GameState, entity_handle Index)
{
    entity *Result = 0;

    if (Index > 0 && Index < GameState->EntitiesCount)
    {
        Result = &(GameState->Entities[Index]);
    }
    else
    {
        // TODO(Kevin): Logging warning
        Assert(0);

        Result = &(GameState->Entities[0]);
    }

    return Result;
}

internal void
MoveEntity(game_state *GameState, entity *Entity, vec2 ddHorizontalP, real32 DragFactor, real32 dt,
           bool32 ApplyGravity, real32 *OutTravelDist, vec3 *OutTravelVector)
{
    //
    // Updating velocity
    //
    vec2 ddP = ddHorizontalP;
    ddP = ddP - DragFactor*Entity->dHorizontalP; // Hack instead of ODE
    vec2 PlayerDelta = (0.5f*ddP*Square(dt) + Entity->dHorizontalP*dt);
    Entity->dHorizontalP = ddP*dt + Entity->dHorizontalP;
    // NOTE(Kevin): When velocity is sufficiently small set it to zero, otherwise
    //              I'll keep moving with increasingly small increments until NAN
    if (Magnitude(Entity->dHorizontalP) < 1.f)
    {
        Entity->dHorizontalP = vec2();
    }

    real32 TravelDist = Magnitude(PlayerDelta);
    if (OutTravelDist)
    {
        *OutTravelDist = TravelDist;
    }
    vec3 TravelVector = Normalize(vec3(PlayerDelta.X, 0.0f, PlayerDelta.Y));
    if (OutTravelVector)
    {
        *OutTravelVector = TravelVector;
    }

    //
    // Collision checking
    //
    vec3 TestP = Entity->P;

    // NOTE(Kevin): Moving in smaller increments before resolving collision to
    // avoid tunneling and erroneous collision resolution vectors (that resolve
    // into walls).
    // TODO(Kevin): Auto calculate number of steps based on collider radius and velocity
    // TODO(Kevin): Should horizontal and vertical checks be separate like this?
    u32 NumSteps = 4;
    real32 MoveStepVelocity = TravelDist/(real32)NumSteps;
    for (u32 Step = 0; Step < NumSteps; ++Step)
    {
        TestP += TravelVector * MoveStepVelocity;
        vec3 ResolutionVector = ResolveEntityCollisionsWithWorld(GameState, &GameState->Map, TestP, Entity);
        TestP += ResolutionVector;
        // TODO(Kevin): Horizontal velocity is not nullified after hitting wall!
    }

    if (ApplyGravity)
    {
        Entity->IsSupported = false;
        Entity->dVerticalP -= 314.f * dt; // 9.81 m/s * 32 units/m ~= 314 units/s
    }
    TestP.y += Entity->dVerticalP * dt;
    vec3 ResolutionVector = ResolveEntityCollisionsWithWorld(GameState, &GameState->Map, TestP, Entity);
    TestP += ResolutionVector;

    real32 ResolutionLenAlongUp = Dot(Normalize(ResolutionVector), GMATH_UP);
    if (AbsoluteValue(ResolutionLenAlongUp) > 0.1f)
    {
        if (ResolutionLenAlongUp > 0.1f)
        {
            // NOTE(Kevin): Since we set IsSupported after vertical movement, we are not able
            //              to start jumping until the frame after we touch the floor.
            Entity->IsSupported = true;
        }

        Entity->dVerticalP = 0.0f;
    }

    Entity->P = TestP;
}

internal void
DeleteEntity(entity *Entity)
{
    Entity->Alive = false;
    Entity->Visible = false;
}

internal entity_handle
SpawnShooterEntity(game_state *GameState, vec3 SpawnP)
{
    entity_handle EntityIndex = AddEntity(GameState, true);

    entity *Entity = GetEntity(GameState, EntityIndex);
    Entity->Type = Entity_Shooter;
    Entity->P = SpawnP;
    Entity->State = EnemyState_Chasing;

    Entity->Visible = true;
    Entity->RenderMesh = GameState->CylinderMesh;
    Entity->RenderAlbedo = GameState->TexWireframeRed;
    Entity->RenderScale = vec3(16.f, 47.f, 16.f);
    Entity->QueryLightCache = true;

    Entity->ColliderPoints = GameState->Cylinder8SidedPointCloud;
    Entity->ColliderPointsCount = ArrayCount(GameState->Cylinder8SidedPointCloud);
    Entity->ColliderScale = vec3(16.f, 47.f, 16.f);

    return EntityIndex;
}

internal entity_handle
SpawnFireballEntity(game_state *GameState, vec3 SpawnP, vec3 DirToPlayer)
{
    entity_handle EntityIndex = AddEntity(GameState, true);

    // NOTE(Kevin): The fastest an entity can travel per second without tunneling
    // through planar walls is:
    // 
    //     R = Radius of the entity collider at its thinnest section
    //     TR = Tick rate
    //     Steps = Number of steps to take towards destination i.e. collision checks per tick
    //     Fastest Units Per Second = R * TR * Steps
    //
    // This guarantees the collider will intersect the planar wall on at least one step.
    // TODO(Kevin): If cheaper, might be better to project the hull (cast rays) to check
    // intersection with wall for continuous collision detection. Could even use artifical
    // broadphase hull.

    DirToPlayer = Normalize(DirToPlayer);
    vec2 dHorizontalP = { DirToPlayer.x, DirToPlayer.z };
    real32 dVerticalP = DirToPlayer.y;
    real32 FireballSpeed = 1000.f;
    dHorizontalP *= FireballSpeed;
    dVerticalP *= FireballSpeed;

#if 1
    quat Rotation = LookRotation3(DirToPlayer);
#else
    quat Rotation = LookRotation1(DirToPlayer, GMATH_UP);
#endif

    entity *Entity = GetEntity(GameState, EntityIndex);
    Entity->Type = Entity_Fireball;
    Entity->P = SpawnP;
    Entity->Orientation = Rotation;
    Entity->State = EnemyState_Chasing;
    Entity->dHorizontalP = dHorizontalP;
    Entity->dVerticalP = dVerticalP;

    Entity->Visible = true;
    Entity->RenderMesh = GameState->CubeMesh;
    Entity->RenderAlbedo = GameState->TexWireframeRed;
    Entity->RenderScale = vec3(12.0f,12.0f,12.0f);
    Entity->QueryLightCache = false; // TODO(Kevin): Force flat lighting

    Entity->ColliderPoints = GameState->CubePointCloud;
    Entity->ColliderPointsCount = ArrayCount(GameState->CubePointCloud);
    Entity->ColliderScale = vec3(12.0f, 12.0f, 12.0f);

    return EntityIndex;
}

extern "C" GAME_UPDATE_AND_RENDER(GameUpdateAndRender)
{
    Assert((&Input->Controllers[0].Terminator - &Input->Controllers[0].Buttons[0]) == 
           ArrayCount(Input->Controllers[0].Buttons));
    Assert(sizeof(game_state) <= Memory->PermanentStorageSize);

    game_state *GameState = (game_state *)Memory->PermanentStorage;
    gui_state *GuiState = (gui_state *)((uint8 *)Memory->PermanentStorage + sizeof(game_state));
    ResetArena(&GameState->FrameArena);
    real32 dt = Input->dtForFrame;

#if HANDMADE_INTERNAL
    GJKCallCounter = 0;
    EPACallCounter = 0;
#endif

    if (!Memory->IsInitialized)
    {
        InitializeArena(&GameState->PermanentArena, Memory->PermanentStorageSize - sizeof(game_state) - sizeof(gui_state),
                        (uint8 *)Memory->PermanentStorage + sizeof(game_state) + sizeof(gui_state));

        InitializeArena(&GameState->FrameArena, Memory->TransientStorageSize,
                        (uint8 *)Memory->TransientStorage);

        GameState->EntitiesCount = ArrayCount(GameState->Entities);
        GameState->FreeEntityHandlesCount = ArrayCount(GameState->FreeEntityHandles);
        Assert(ArrayCount(GameState->Entities) == ArrayCount(GameState->FreeEntityHandles));
        PopulateFreeHandles(GameState->FreeEntityHandles, GameState->FreeEntityHandlesCount);
        AddEntity(GameState, false); // NOTE(Kevin): Reserve entity slot 0 for the null entity

        GameState->Backdrop = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "test/test_background.bmp");

        GameState->CubeMesh = RendererAPI->UploadMeshData(CubeVB, ArrayCount(CubeVB), CubeIB, ArrayCount(CubeIB));

        // TODO(Kevin): Release bitmap memory
        // TODO(Kevin): Load this default white lightmap from inside renderer code! Not from game code!
        // NOTE(Kevin): White_1x1 should be loaded first to have texture handle 0
        loaded_bitmap White_1x1 = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "white_1x1.bmp");
        RendererAPI->WhiteTexture_1x1 = RendererAPI->UploadTexture2DData(White_1x1.Pixels, White_1x1.Width, White_1x1.Height, BGRA_U8, false);

        loaded_bitmap TestBmp = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "test_1.bmp");
        texture_handle Test1Texture = RendererAPI->UploadTexture2DData(TestBmp.Pixels, TestBmp.Width, TestBmp.Height, BGRA_U8, false);

        loaded_bitmap TestBmp2 = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "default_8x8.bmp");
        texture_handle DefaultTexture = RendererAPI->UploadTexture2DData(TestBmp2.Pixels, TestBmp2.Width, TestBmp2.Height, BGRA_U8, false);

        texture_handle BackdropTexture = RendererAPI->UploadTexture2DData(GameState->Backdrop.Pixels, GameState->Backdrop.Width, GameState->Backdrop.Height, BGRA_U8, false);

        loaded_bitmap BmpWireframeRed = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "wireframe_red.bmp");
        GameState->TexWireframeRed = RendererAPI->UploadTexture2DData(BmpWireframeRed.Pixels, BmpWireframeRed.Width, BmpWireframeRed.Height, BGRA_U8, true);

        loaded_bitmap CustomFont0Bmp = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "fonts/anikki_8x8.bmp");
        create_font_result RenderFont0 = DEBUGCreateBitmapFont(CustomFont0Bmp, RendererAPI);
        GuiState->Fonts[GuiFontAnikki8x8] = RenderFont0;

        loaded_bitmap CustomFont1Bmp = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "fonts/custom_font_0_6x9.bmp");
        create_font_result RenderFont1 = DEBUGCreateBitmapFont(CustomFont1Bmp, RendererAPI);
        GuiState->Fonts[GuiFontCurses6x9] = RenderFont1;

        create_font_result RenderFont2 = RendererAPI->DEBUGCreateBitmapFont_TrueType(Thread, Memory->PlatformAPI, 40,
                                                                                     "fonts/cambria.ttc");
        GuiState->Fonts[GuiFontCambria40] = RenderFont2;

        BuildCylinderMesh8Sided(GameState->Cylinder8SidedPointCloud, 
                                GameState->Cylinder8SidedVB, GameState->Cylinder8SidedIB);
        for (u32 Index = 0; Index < ArrayCount(GameState->CubePointCloud); ++Index)
        {
            GameState->CubePointCloud[Index] = CubePoints[Index];
        }
        GameState->CylinderMesh = RendererAPI->UploadMeshData(GameState->Cylinder8SidedVB, CylinderMesh8SidedVertexCount, 
                                                              GameState->Cylinder8SidedIB, CylinderMesh8SidedIndexCount);

#if !SHOOTER_COMPILE_WITH_EDITOR
        char *MapFilename = "../../old_game/wd/buildtest.map";
        debug_read_file_result MapReadResult = Memory->PlatformAPI.DEBUGPlatformReadEntireFile(Thread, MapFilename);
        if (MapReadResult.ContentsSize > 0)
        {
            buffer MapBytes;
            MapBytes.Data = (uint8 *)MapReadResult.Contents;
            MapBytes.Count = MapReadResult.ContentsSize;
            umm Position = 0;

#define ByteBufferRead(Buffer, PositionPtr, Type, WritePtr)\
            do {\
                if ((WritePtr))\
                {\
                    Type *WritePtrCasted = (Type *)(WritePtr);\
                    *WritePtrCasted = *(Type *)((Buffer).Data + *(PositionPtr));\
                }\
                *(PositionPtr) += sizeof(Type);\
            } while (0);

#define ByteBufferReadBulk(Buffer, PositionPtr, BytesToRead, WritePtr)\
            do {\
                if ((WritePtr))\
                {\
                    Copy((WritePtr), ((Buffer).Data + *(PositionPtr)), BytesToRead);\
                }\
                *(PositionPtr) += BytesToRead;\
            } while (0);

            ByteBufferRead(MapBytes, &Position, vec3, 0); //&GameState->PlayerStartPosition);
            ByteBufferRead(MapBytes, &Position, vec3, 0); //&GameState->PlayerStartRotation);

            int32 LightmapW, LightmapH;
            ByteBufferRead(MapBytes, &Position, int32, &LightmapW);
            ByteBufferRead(MapBytes, &Position, int32, &LightmapH);
            uint32 LightmapDimensions = (uint32)(LightmapW*LightmapH);
            float *LightmapData = (float *)malloc(LightmapDimensions*sizeof(float));
            ByteBufferReadBulk(MapBytes, &Position, LightmapDimensions*sizeof(float), LightmapData);
            // TODO(Kevin): Yeah don't create texture resources here...just load all the data first
            // LightmapData should be part of e.g. map_data
            texture_handle MapGeometryLightmap = 
                RendererAPI->UploadTexture2DData(LightmapData, SafeCastToU32(LightmapW), 
                                                 SafeCastToU32(LightmapH), R_Float, true);

            // Light cache volume
            uint32 TotalCubeCount;
            ByteBufferRead(MapBytes, &Position, uint64, 0);
            ByteBufferRead(MapBytes, &Position, vec3, 0);//&Start);
            ByteBufferRead(MapBytes, &Position, vec3, 0);//&End);
            ByteBufferRead(MapBytes, &Position, int, 0);//&CountX);
            ByteBufferRead(MapBytes, &Position, int, 0);//&CountY);
            ByteBufferRead(MapBytes, &Position, int, 0);//&CountZ);
            ByteBufferRead(MapBytes, &Position, ivec3, 0);//&LightCubePlacementInterval);
            ByteBufferRead(MapBytes, &Position, uint32, &TotalCubeCount);
            ByteBufferReadBulk(MapBytes, &Position, sizeof(vec3)*TotalCubeCount, 0);//CubePositions.data);
            ByteBufferReadBulk(MapBytes, &Position, sizeof(float)*6*TotalCubeCount, 0);//AmbientCubes.data);
            ByteBufferReadBulk(MapBytes, &Position, sizeof(short)*4*TotalCubeCount, 0);//SignificantLightIndices.data);

            // DeserializeMapLightData
            ByteBufferRead(MapBytes, &Position, uint64, 0);//&SerializeStartMarker);
            ByteBufferRead(MapBytes, &Position, vec3, 0);//&MapInfo->DirectionToSun);
            //MapInfo->DirectionToSun = Normalize(MapInfo->DirectionToSun);
            uint32 PointLightsCount;
            ByteBufferRead(MapBytes, &Position, uint32, &PointLightsCount);
            // MapInfo->PointLights = fixed_array<static_point_light_t>(PointLightsCount, MemoryType::Level);
            // MapInfo->PointLights.setlen(PointLightsCount);
            ByteBufferReadBulk(MapBytes, &Position, sizeof(float)*5*PointLightsCount, 0);//MapInfo->PointLights.data,

            // TODO(Kevin): SerializeArray function that takes pointer, element size, element count
            // TODO(Kevin): These would need to pointers from game_map
            // - Lightmap data
            // - Light cache volume
            // - PointLights probably
            // - Colliders
            // - Vertex buffers

            // collider shit
            size_t numColliderPoints, numColliderSpans;
            ByteBufferRead(MapBytes, &Position, size_t, &numColliderPoints);
            ByteBufferRead(MapBytes, &Position, size_t, &numColliderSpans);
            GameState->Map.ColliderPointsCount = SafeTruncateU64(numColliderPoints);

            u32 *Scratch = (u32 *)CurrentPtr(&GameState->FrameArena);
            ByteBufferReadBulk(MapBytes, &Position, sizeof(vec3)*numColliderPoints,
                               GameState->Map.ColliderPoints);
            ByteBufferReadBulk(MapBytes, &Position, sizeof(u32)*numColliderSpans,
                               Scratch);
            for (memory_index SpanOffset = 0;
                 SpanOffset < numColliderPoints;
                 ++Scratch)
            {
                u32 Span = *Scratch;

                flatpoly_collider Collider;
                Collider.PointCloud = (GameState->Map.ColliderPoints + SpanOffset);
                Collider.PointCount = Span;
                GameState->Map.Colliders[GameState->Map.CollidersCount++] = Collider;

                SpanOffset += Span;
            }

            // vertex buffers
            size_t numVertexBufs;
            ByteBufferRead(MapBytes, &Position, size_t, &numVertexBufs);
            for (int i = 0; i < numVertexBufs; ++i)
            {
                uint32 texturePersistId;
                size_t FloatValuesCount;
                ByteBufferRead(MapBytes, &Position, uint32, &texturePersistId);
                ByteBufferRead(MapBytes, &Position, size_t, &FloatValuesCount);
                uint32 VertexCount = (uint32)(FloatValuesCount / (sizeof(PNUV2_vertex) / sizeof(float)));
                // TODO(Kevin): This is test code! won't be using malloc ever.
                PNUV2_vertex *vb = (PNUV2_vertex *)malloc(sizeof(float)*FloatValuesCount);
                uint32 *ib = (uint32 *)malloc(sizeof(uint32)*VertexCount);
                ByteBufferReadBulk(MapBytes, &Position, sizeof(float)*FloatValuesCount, vb);

                // TODO(Kevin): Since this is a hack to work w vertex buffers without index buffers,
                // maybe have one array with uint32s from 0 ... some big N to reuse instead of creating
                // a new one every time - could be useful for vertex streaming
                for (uint32 VertexIndex = 0; VertexIndex < VertexCount; ++VertexIndex)
                {
                    ib[VertexIndex] = VertexIndex;
                }

                mesh_handle MapGeometry = RendererAPI->UploadMeshData(vb, VertexCount, ib, VertexCount);
                free(vb);
                free(ib);

                // TODO(Kevin): Map geometry does not need to be entity if I don't want
                entity_handle EntityIndex = AddEntity(GameState, false);
                entity *Entity = GetEntity(GameState, EntityIndex);
                Entity->Visible = true;
                Entity->RenderMesh = MapGeometry;
                Entity->RenderAlbedo = DefaultTexture;
                Entity->RenderLightmap = MapGeometryLightmap;
            }

            Memory->PlatformAPI.DEBUGPlatformFreeFileMemory(Thread, MapReadResult.Contents);
        }
#endif

        PointCamera(&GameState->Camera, vec3(-100, 60, 0), vec3(1,0,0));

#if !SHOOTER_COMPILE_WITH_EDITOR
        GameState->PlayerEntityHandle = AddEntity(GameState, true);
        entity *Player = GetEntity(GameState, GameState->PlayerEntityHandle);
        Player->P = vec3(-100, 0.1f, 0);
        Player->HealthPoints = 100.0f;
        Player->ColliderPoints = GameState->Cylinder8SidedPointCloud;
        Player->ColliderPointsCount = ArrayCount(GameState->Cylinder8SidedPointCloud);
        Player->ColliderScale = vec3(8.f,47.f,8.f);

        for (u32 i = 0; i < 10; ++i)
        {
            entity_handle EntityIndex = AddEntity(GameState, false);
            entity *Entity = GetEntity(GameState, EntityIndex);
            Entity->Visible = true;
            Entity->RenderMesh = GameState->CubeMesh;
            Entity->RenderAlbedo = i > 4 ? DefaultTexture : Test1Texture;
            Entity->P = vec3(100, 30, ((real32)(i) * 25) - 125);
            Entity->RenderScale = vec3(1,1,1);
            Entity->QueryLightCache = true;
        }

        SpawnShooterEntity(GameState, vec3(-400, 0.01f, 0));
        SpawnShooterEntity(GameState, vec3(400, 0.01f, 0));
        // SpawnShooterEntity(GameState, vec3(300, 0.01f, -200));
#else
        GameState->EditorMode = true;
        Memory->PlatformAPI.CaptureMouseCursor(false);

        Assert(Memory->EditorMemoryStorageSize > 0);
        InitializeMemoryBlocks(&GameState->Editor.EditorMemory,
                               Memory->EditorMemoryStorageSize,
                               Memory->EditorMemoryStorage);
#endif

        Memory->IsInitialized = true;
    }

    //
    // Input
    //
    for (int ControllerIndex = 0;
         ControllerIndex < ArrayCount(Input->Controllers);
         ++ControllerIndex)
    {
        game_controller_input *Controller = GetController(Input, ControllerIndex);

        if (!Controller->IsConnected)
        {
            continue;
        }

        if (ControllerIndex == 0)
        {
            real32 MouseLookSensitivity = 0.001f;
            vec2 AdjustedMouseAxes = Input->RawMouseDelta * MouseLookSensitivity;

            bool32 FlyCam = false;

#if SHOOTER_COMPILE_WITH_EDITOR
            if (GameState->EditorMode)
            {
                FlyCam = true;

                if (Input->MouseButtons[1].EndedDown/* && !GuiState->Hovered*/)
                {
                    Memory->PlatformAPI.CaptureMouseCursor(true);
                    MouseLookCamera(&GameState->Camera, AdjustedMouseAxes.x, AdjustedMouseAxes.y);
                }
                else
                {
                    Memory->PlatformAPI.CaptureMouseCursor(false);
                }
            }
#else
            MouseLookCamera(&GameState->Camera, AdjustedMouseAxes.x, AdjustedMouseAxes.y);
#endif
            if (FlyCam)
            {
                vec3 PositionDelta;
                real32 MoveSpeed = 350.f;
                real32 MoveIncrement = MoveSpeed * dt;
                if (Controller->MoveUp.EndedDown)
                {
                    PositionDelta += GameState->Camera.Forward * MoveIncrement;
                }
                if (Controller->MoveDown.EndedDown)
                {
                    PositionDelta += -GameState->Camera.Forward * MoveIncrement;
                }
                if (Controller->MoveLeft.EndedDown)
                {
                    PositionDelta += -GameState->Camera.Right * MoveIncrement;
                }
                if (Controller->MoveRight.EndedDown)
                {
                    PositionDelta += GameState->Camera.Right * MoveIncrement;
                }
                if (Controller->LeftShoulder.EndedDown)
                {
                    PositionDelta += -GMATH_UP * MoveIncrement;
                }
                if (Controller->RightShoulder.EndedDown)
                {
                    PositionDelta += GMATH_UP * MoveIncrement;
                }
                GameState->Camera.P += PositionDelta;
                GameState->Camera.ViewPosition = GameState->Camera.P;
            }
            else if (!GameState->EditorMode)
            {
                entity *Player = GetEntity(GameState, GameState->PlayerEntityHandle);
                // NOTE(Kevin): Camera.Right should already be updated from MouseLook!
                Player->PlayerWalkDirectionRight = GameState->Camera.Right;
                Player->PlayerWalkDirectionForward = Normalize(Cross(GMATH_UP, GameState->Camera.Right));

                bool32 MoveForward = Controller->MoveUp.EndedDown;
                bool32 MoveBack = Controller->MoveDown.EndedDown;
                bool32 StrafeLeft = Controller->MoveLeft.EndedDown;
                bool32 StrafeRight = Controller->MoveRight.EndedDown;
                bool32 Jump = Controller->ActionDown.EndedDown && Controller->ActionDown.HalfTransitionCount > 0;
                bool32 MovingHorizontal = MoveForward | MoveBack | StrafeLeft | StrafeRight;

                vec3 PlayerHorizontalVector;
                if (MoveForward)
                {
                    PlayerHorizontalVector += Player->PlayerWalkDirectionForward;
                }
                if (MoveBack)
                {
                    PlayerHorizontalVector += -Player->PlayerWalkDirectionForward;
                }
                if (StrafeLeft)
                {
                    PlayerHorizontalVector += -Player->PlayerWalkDirectionRight;
                }
                if (StrafeRight)
                {
                    PlayerHorizontalVector += Player->PlayerWalkDirectionRight;
                }
                if (Jump && Player->IsSupported)
                {
                    Player->dVerticalP = 160.f;
                }

                vec2 ddP = { PlayerHorizontalVector.x, PlayerHorizontalVector.z };
                ddP = Normalize(ddP);
                bool32 ChangeToOppositeDirection = Dot(ddP, Normalize(Player->dHorizontalP)) < 0.0f;
                real32 PlayerHorizontalAcceleration = 3200.0f; // units/s^2
                real32 MaxHorizontalSpeed = 350.0f;
                ddP *= PlayerHorizontalAcceleration;
                real32 DragFactor = PlayerHorizontalAcceleration / MaxHorizontalSpeed;

                // Bob view up and down while walking
                real32 MaxBobOffset = -4.6f;
                // NOTE(Kevin): ddPlayerViewBobOffset (a) scales linearly with MaxBobOffset (d)
                //              given constant time    d = 1/2(at^2) -> (2/t^2)d = a
                real32 ddPlayerViewBobOffset = 20.f*MaxBobOffset;
                GameState->PlayerViewBobGoingDown &= MovingHorizontal;

                // NOTE(Kevin): Spamming A D back and forth should nullify the bob
                if (ChangeToOppositeDirection && GameState->PlayerViewBobGoingDown)
                {
                    GameState->PlayerViewBobGoingDown = false;
                }

                if (GameState->PlayerViewBobGoingDown && Player->IsSupported)
                {
                    GameState->PlayerViewBobOffset += GameState->dPlayerViewBobOffset * dt;
                    GameState->dPlayerViewBobOffset += ddPlayerViewBobOffset * dt;

                    if (GameState->PlayerViewBobOffset < MaxBobOffset)
                    {
                        GameState->PlayerViewBobOffset = MaxBobOffset;
                        GameState->PlayerViewBobGoingDown = false;
                    }
                }
                else
                {
                    // Bob offset recovery
                    GameState->PlayerViewBobOffset = Blend(GameState->PlayerViewBobOffset, (-MaxBobOffset*0.2f), 0.062f);
                    if (GameState->PlayerViewBobOffset >= 0.0f)
                    {
                        GameState->PlayerViewBobOffset = 0.0f;
                        GameState->dPlayerViewBobOffset = 0.0f;

                        if (MovingHorizontal)
                        {
                            GameState->PlayerViewBobGoingDown = true;
                        }
                    }
                }

                real32 BaseCamOffsetFromPlayerRoot = 40.0f;
                real32 CamYOffset = BaseCamOffsetFromPlayerRoot + GMATH_Min(0.0f, GameState->PlayerViewBobOffset);
                PointCamera(&GameState->Camera, Player->P + vec3(0,CamYOffset,0), GameState->Camera.Forward);
                UpdateCameraStrafeTilt(&GameState->Camera, Player->dHorizontalP,
                                       GameState->PlayerMaxUnitsTravelledPerSecond, dt);

                real32 TravelDist;
                MoveEntity(GameState, Player, ddP, DragFactor, dt, true, &TravelDist, NULL);
                GameState->PlayerMaxUnitsTravelledPerSecond = MaxHorizontalSpeed;
                GameState->PlayerUnitsTravelledPerSecond = TravelDist / dt;
            }
        }
    }

    uint32 RenderItemsCount = 0;
    uint8 *RenderItemsStart = CurrentPtr(&GameState->FrameArena);

    //
    // Test Code! Entity UpdateAndRender
    //
    for (entity_handle EntityIndex = 1;
         EntityIndex < GameState->EntitiesCount;
         ++EntityIndex)
    {
        entity *Entity = GetEntity(GameState, EntityIndex);

        if (Entity->Alive && !GameState->EditorMode)
        {
            Entity->AliveSeconds += dt;

            switch (Entity->Type)
            {
                default: break;

                case Entity_Fireball:
                {
                    vec3 OldEntityPos = Entity->P;
                    real32 TravelDist;
                    MoveEntity(GameState, Entity, vec2(0,0), 0, dt, false, &TravelDist, 0);                    
                    if (Magnitude(Entity->P - OldEntityPos) < AbsoluteValue(TravelDist - 0.01f))
                    {
                        // Entity->dHorizontalP = {};
                        DeleteEntity(Entity);
                    }
                    else
                    {
                        if (Entity->AliveSeconds > 3.0f)
                        {
                            DeleteEntity(Entity);
                        }
                        else
                        {
                            entity *Player = GetEntity(GameState, GameState->PlayerEntityHandle);

                            temporary_memory TemporaryColliders = BeginTemporaryMemory(&GameState->FrameArena);
                            mesh_collider A = GetTransformedEntityCollider(Entity, Entity->P, Entity->Orientation,
                                                                           Entity->ColliderScale, &GameState->FrameArena);
                            mesh_collider B = GetTransformedEntityCollider(Player, Player->P, Player->Orientation,
                                                                           Player->ColliderScale, &GameState->FrameArena);
                            EndTemporaryMemory(TemporaryColliders);

                            bool32 Collided = GJK((collider_base *)&A, (collider_base *)&B, NULL);
                            if (Collided)
                            {
                                Player->HealthPoints -= 30.f;
                                DeleteEntity(Entity);
                            }
                        }
                    }
                } break;

                case Entity_Shooter:
                {
                    entity *Player = GetEntity(GameState, GameState->PlayerEntityHandle);
                    vec3 ToPlayer = Player->P - Entity->P;
                    vec3 DirToPlayer = Normalize(ToPlayer);
                    vec3 ToPlayerHorizontal = vec3(ToPlayer.x, 0, ToPlayer.z);

                    real32 HorizontalAcceleration = 1000.f;
                    real32 MaxHorizontalSpeed = 150.f;
                    real32 DragFactor = HorizontalAcceleration / MaxHorizontalSpeed;
                    vec2 ddHorizontalP;

                    switch (Entity->State)
                    {
                        case EnemyState_Shooting:
                        {
                            quat PlayerFacingO = OrientationFromDirection(Normalize(ToPlayerHorizontal));
                            Entity->Orientation = PlayerFacingO;

                            if (Entity->tShoot > 0.0f)
                            {
                                Entity->tShoot -= dt;
                            }
                            if (Entity->tShoot <= 0.0f)
                            {
                                vec3 FireballSpawnP = Entity->P + (DirToPlayer*10.f) + vec3(0,23,0);
                                SpawnFireballEntity(GameState, FireballSpawnP, DirToPlayer);
                                Entity->tShoot = 0.4f;
                            }
                        } break;
                        case EnemyState_Chasing:
                        {
                            quat PlayerFacingO = OrientationFromDirection(Normalize(ToPlayerHorizontal));
                            Entity->Orientation = PlayerFacingO;

                            ddHorizontalP = Normalize(vec2(ToPlayer.x, ToPlayer.z));
                            ddHorizontalP *= HorizontalAcceleration;
                        } break;
                    }

                    MoveEntity(GameState, Entity, ddHorizontalP, DragFactor, dt, true, 0, 0);

                    if (Entity->State == EnemyState_Chasing)
                    {
                        // TODO(Kevin): These local_persist do not survive DLL reload
                        local_persist u32 RandomNumberIndex = 0;
                        u32 RandomChoices = 200;
                        u32 RandomChoice = RandomNumberTable[(RandomNumberIndex++)%ArrayCount(RandomNumberTable)] % RandomChoices;
                        if (RandomChoice <= 1)
                        {
                            Entity->State = EnemyState_Shooting;
                        }
                    }
                    else if (Entity->State == EnemyState_Shooting)
                    {
                        local_persist u32 RandomNumberIndex = 0;
                        u32 RandomChoices = 100;
                        u32 RandomChoice = RandomNumberTable[(RandomNumberIndex++)%ArrayCount(RandomNumberTable)] % RandomChoices;
                        if (RandomChoice <= 1)
                        {
                            Entity->State = EnemyState_Chasing;
                        }
                    }

                } break;
            }
        }

        if (Entity->Visible)
        {
            render_item *RenderItem = PushStruct(&GameState->FrameArena, render_item);

            *RenderItem = {};
            RenderItem->Descriptor.Mesh = Entity->RenderMesh;
            RenderItem->Descriptor.Albedo = Entity->RenderAlbedo;
            RenderItem->Descriptor.Lightmap = Entity->RenderLightmap;
            RenderItem->UseLightmap = Entity->RenderLightmap > 0;
            RenderItem->UseLightCubes = Entity->QueryLightCache;
            RenderItem->WorldPosition = Entity->P;
            RenderItem->WorldRotation = Entity->Orientation;
            RenderItem->WorldScale = Entity->RenderScale;

            ++RenderItemsCount;
        }
    }

    mat4 ViewFromWorld = CameraSpaceFromWorldSpace(&GameState->Camera);
    RendererAPI->PushViewMatrix(ViewFromWorld);
    RendererAPI->PushRenderItems((render_item *)RenderItemsStart, RenderItemsCount);

    uint32 CanvasW, CanvasH;
    RendererAPI->GetRenderTargetDimensions(&CanvasW, &CanvasH);
    if (CanvasW > 0 && CanvasW < UINT32_MAX && 
        CanvasH > 0 && CanvasH < UINT32_MAX)
    {
        GuiBeginFrame(GuiState, CanvasW, CanvasH, &GameState->FrameArena,
                      Input->MouseX, Input->MouseY, Input->MouseButtons[0], Input->MouseButtons[1]);

        if (!GameState->EditorMode)
        {
            u32 CrosshairDotWidth = 6;
            gui_rect CrosshairOuterRect = {SafeCastToS32(CanvasW / 2 - (CrosshairDotWidth/2)),
                                           SafeCastToS32(CanvasH / 2 - (CrosshairDotWidth/2)),
                                           CrosshairDotWidth, CrosshairDotWidth};
            gui_rect CrosshairInnerRect = {SafeCastToS32(CanvasW / 2 - (CrosshairDotWidth/2) + 1),
                                           SafeCastToS32(CanvasH / 2 - (CrosshairDotWidth/2) + 1),
                                           CrosshairDotWidth-2, CrosshairDotWidth-2};
            GuiPanel(GuiState, CrosshairOuterRect, vec4(0.f,0.f,0.f,0.8f));
            GuiPanel(GuiState, CrosshairInnerRect, vec4(0.8f,0.8f,0.8f,0.8f));

            GuiTextFmt(GuiState, 100, CanvasH-200, GuiFontAnikki8x8,
                       6.0f+4.0f*(GameState->PlayerUnitsTravelledPerSecond/GameState->PlayerMaxUnitsTravelledPerSecond),
                       Align_Left, false, vec4(1,1,1,1),
                       "%d u/s", RoundReal32ToInt32(GameState->PlayerUnitsTravelledPerSecond));

            entity *Player = GetEntity(GameState, GameState->PlayerEntityHandle);

            GuiTextFmt(GuiState, 100, CanvasH-100, GuiFontAnikki8x8, 
                       6.0f, Align_Left, false, vec4(1,1,1,1),
                       "HP: %f", Player->HealthPoints);

            // 6 sides * 6 cubes * 5 steps = 180 GJK calls per entity
            GuiTextFmt(GuiState, 40, 500, GuiFontCurses6x9, 3.0f, Align_Left, false, vec4(0,1,0,1), "GJK: %d calls", GJKCallCounter);
            GuiTextFmt(GuiState, 40, 530, GuiFontCurses6x9, 3.0f, Align_Left, false, vec4(0,1,0,1), "EPA: %d calls", EPACallCounter);

            // if (Player->IsSupported)
            // {
            //     GuiPanel(GuiState, {40, 40, 128, 128 }, GameState->TexWireframeRed);
            // }
        }
        else
        {
            window_state Window = BeginWindow(GuiState, { 10, 80, 400, 600 });

            WindowText(&Window, "== Some Title ==");
            WindowText(&Window, "Hello world!");

            EndWindow(&Window);

#if 0
            GuiPanel(GuiState, { GameState->Editor.EditorMouseX-5, 
                     GameState->Editor.EditorMouseY-5, 10, 10 }, 0);
#endif
#if 0
            if (GameState->Editor.IsInitialized)
            {
                GuiTextFmt(GuiState, 500, 50, GuiFontCambria40, 1.4f, Align_Left, false, vec4(1,1,1,1),
                           "SP: %d", GameState->Editor.UndoStack->StackP);
                GuiTextFmt(GuiState, 500, 100, GuiFontCambria40, 1.4f, Align_Left, false, vec4(1,1,1,1),
                           "HS: %d", GameState->Editor.UndoStack->HistoryStartP);
                GuiTextFmt(GuiState, 500, 150, GuiFontCambria40, 1.4f, Align_Left, false, vec4(1,1,1,1),
                           "MR: %d", GameState->Editor.UndoStack->MostRecentP);
            }
#endif

            if (GuiState->TooltipTimer > 0.f)
            {
                GuiState->TooltipTimer -= dt;

                if (GuiState->TooltipLength)
                {
                    real32 TextWidth, _;
                    real32 TextScale = 2.0f;
                    VT_GetTextBoundingBoxDimensions(GuiState->Fonts[GuiFontCambria40].FontInfo, 
                                                    GuiState->Tooltip, TextScale, &TextWidth, &_,
                                                    GuiState->TooltipLength);
                    s32 X = (s32)(CanvasW - RoundReal32ToUInt32(TextWidth)) / 2;
                    GuiText(GuiState, X, (s32)CanvasH - 100, GuiFontCambria40, TextScale,
                            Align_Left, false, { 1.f, 1.f, 1.f, 1.f }, GuiState->Tooltip);
                }
            }
        }

#if 0
        GuiPanel(GuiState, {99, 299, 514, 514}, GetDebugColor4(10));
        GuiPanel(GuiState, {100, 300, 512, 512}, GuiState->Fonts[GuiFontCambria40].FontAtlasHandle);
#endif

        RendererAPI->PushGuiDrawRequests(&GameState->FrameArena, GuiState->DrawRequests, GuiState->DrawRequestsCount);
    }

#if SHOOTER_COMPILE_WITH_EDITOR
    if (GameState->EditorMode)
    {
        EditorUpdateAndRender(Thread, GameState, GuiState, Input, RendererAPI);
    }
#endif
}

extern "C" GAME_GET_SOUND_SAMPLES(GameGetSoundSamples)
{
    game_state *GameState = (game_state *)Memory->PermanentStorage;
    GameOutputSound(GameState, SoundBuffer, 256);
}


