#include "shooter.h"
#include "shooter_renderer.h"
#include "shooter_shared.h"
#include "shooter_gui.h"
#include "shooter_gui.cpp"
#define STB_SPRINTF_IMPLEMENTATION
#include <stb_sprintf.h>

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

internal font_handle
CreateBitmapFont(loaded_bitmap Bitmap, renderer *RendererAPI)
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

    font_handle Result = RendererAPI->CreateBitmapFont_ExtendedASCII(Bitmap.Pixels, Bitmap.Width, Bitmap.Height, BGRA_U8, 3);
    return Result;
}

internal void
PlaceCamera(camera *Camera, vec3 Pos, vec3 Forward)
{
    vec3 WorldUp = GM_UP_VECTOR;
    vec3 WorldRight = GM_RIGHT_VECTOR;

    Camera->P = Pos;
    Camera->Forward = Normalize(Forward);
    Camera->Right = Normalize(Cross(Camera->Forward, WorldUp));
    Camera->Up = Normalize(Cross(Camera->Right, Camera->Forward));

    quat Pitch = RotationFromTo(WorldUp, Camera->Up);
    quat Yaw = RotationFromTo(WorldRight, Camera->Right);
    vec3 PitchEuler = QuatToEuler(Pitch);
    vec3 YawEuler = QuatToEuler(Yaw);

    Camera->EulerRot = PitchEuler + YawEuler;
    Camera->Orientation = CombineRotations(Pitch, Yaw);

    Camera->ViewPosition = Camera->P;
    Camera->ViewForward = Camera->Forward;
    Camera->ViewUp = Camera->Up;
}

internal mat4
CameraSpaceFromWorldSpace(camera *Camera)
{
    mat4 ViewMatrix = ViewMatrixLookAt(Camera->ViewPosition,
                                       Camera->ViewPosition + Camera->ViewForward, 
                                       Camera->ViewUp);
    return ViewMatrix;
}

internal void
MouseLookCamera(camera *Camera, real32 MouseAxisX, real32 MouseAxisY)
{
    real32 PitchConstraint = GM_HALFPI-0.0175f;
    real32 YawDelta = MouseAxisX;
    real32 PitchDelta = MouseAxisY;
    Camera->EulerRot.y -= YawDelta;
    Camera->EulerRot.z -= PitchDelta;
    if (Camera->EulerRot.z > PitchConstraint)
        Camera->EulerRot.z = PitchConstraint;
    if (Camera->EulerRot.z < -PitchConstraint)
        Camera->EulerRot.z = -PitchConstraint;

    Camera->Orientation = EulerToQuat(Camera->EulerRot);
    Camera->Forward = Normalize(OrientationToDirection(Camera->Orientation));
    Camera->Right = Normalize(Cross(Camera->Forward, GM_UP_VECTOR));
    Camera->Up = Normalize(Cross(Camera->Right, Camera->Forward));

    Camera->ViewPosition = Camera->P;
    Camera->ViewForward = Camera->Forward;
    Camera->ViewUp = Camera->Up;
}


extern "C" GAME_UPDATE_AND_RENDER(GameUpdateAndRender)
{
    Assert((&Input->Controllers[0].Terminator - &Input->Controllers[0].Buttons[0]) == 
           ArrayCount(Input->Controllers[0].Buttons));
    Assert(sizeof(game_state) <= Memory->PermanentStorageSize);

    game_state *GameState = (game_state *)Memory->PermanentStorage;
    ResetArena(&GameState->ScratchArena);
    gui_state *GuiState = (gui_state *)Memory->TransientStorage;
    if (!Memory->IsInitialized)
    {
        GameState->Backdrop = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "test/test_background.bmp");

        PNUV2_vertex CubeVB[] = {
            { { 16.0f, -16.0f, -16.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } }, // +X face
            { { 16.0f,  16.0f, -16.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { { 16.0f,  16.0f,  16.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { 16.0f, -16.0f, -16.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } },
            { { 16.0f,  16.0f,  16.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { 16.0f, -16.0f,  16.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 0.0f } },
            { { -16.0f, -16.0f,  16.0f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } }, // -X face
            { { -16.0f,  16.0f,  16.0f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { { -16.0f,  16.0f, -16.0f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { -16.0f, -16.0f,  16.0f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } },
            { { -16.0f,  16.0f, -16.0f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { -16.0f, -16.0f, -16.0f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 0.0f } },
            { { -16.0f, 16.0f, -16.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } }, // +Y face
            { { -16.0f, 16.0f,  16.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { {  16.0f, 16.0f,  16.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { -16.0f, 16.0f, -16.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } },
            { {  16.0f, 16.0f,  16.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { {  16.0f, 16.0f, -16.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 0.0f } },
            { { -16.0f, -16.0f,  16.0f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } }, // -Y face
            { { -16.0f, -16.0f, -16.0f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { {  16.0f, -16.0f, -16.0f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { -16.0f, -16.0f,  16.0f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } },
            { {  16.0f, -16.0f, -16.0f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { {  16.0f, -16.0f,  16.0f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 0.0f } },
            { { -16.0f, -16.0f, 16.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } }, // +Z face
            { {  16.0f, -16.0f, 16.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 0.0f } },
            { {  16.0f,  16.0f, 16.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { -16.0f, -16.0f, 16.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } },
            { {  16.0f,  16.0f, 16.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { { -16.0f,  16.0f, 16.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { {  16.0f, -16.0f, -16.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } }, // -Z face
            { { -16.0f, -16.0f, -16.0f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f }, { 1.0f, 0.0f } },
            { { -16.0f,  16.0f, -16.0f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { {  16.0f, -16.0f, -16.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } },
            { { -16.0f,  16.0f, -16.0f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f } },
            { {  16.0f,  16.0f, -16.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f }, { 0.0f, 1.0f } },
        };
        uint32 CubeIB[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 };
        RendererAPI->UploadMeshData(CubeVB, ArrayCount(CubeVB), CubeIB, ArrayCount(CubeIB));

        InitializeArena(&GameState->WorldArena, Memory->PermanentStorageSize - sizeof(game_state),
                        (uint8 *)Memory->PermanentStorage + sizeof(game_state));

        InitializeArena(&GameState->ScratchArena, Memory->TransientStorageSize - sizeof(gui_state),
                        (uint8 *)Memory->TransientStorage + sizeof(gui_state));

        char *MapFilename = "../../wd/buildtest.map";
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
            RendererAPI->UploadTexture2DData(LightmapData, SafeCastToU32(LightmapW), SafeCastToU32(LightmapH),
                                          R_Float, true);

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
            ByteBufferReadBulk(MapBytes, &Position, sizeof(vec3)*numColliderPoints, 0);//MapInfo->LoadingLevelColliderPoints.data, 
            ByteBufferReadBulk(MapBytes, &Position, sizeof(uint32)*numColliderSpans, 0);//MapInfo->LoadingLevelColliderSpans.data, 
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
                uint32 *ib = (uint32 *)malloc(sizeof(uint32)*FloatValuesCount);
                ByteBufferReadBulk(MapBytes, &Position, sizeof(float)*FloatValuesCount, vb);

                // face_batch_t FaceBatch;
                // FaceBatch.ColorTexture = Assets.GetTextureById(texturePersistId).gputex;
                // FaceBatch.LightMapTexture = LevelLightmapTexture;
                // CreateFaceBatch(&FaceBatch);
                // RebindFaceBatch(&FaceBatch, sizeof(float)*VertexCount, vb.data());
                // MapInfo->GameLevelFaceBatches.push_back(FaceBatch);

                // TODO(Kevin): Since this is a hack to work w vertex buffers without index buffers,
                // maybe have one array with uint32s from 0 ... some big N to reuse instead of creating
                // a new one every time - could be useful for vertex streaming
                for (uint32 VertexIndex = 0; VertexIndex < VertexCount; ++VertexIndex)
                {
                    ib[VertexIndex] = VertexIndex;
                }

                RendererAPI->UploadMeshData(vb, VertexCount, ib, VertexCount);
                free(vb);
                free(ib);
            }


            Memory->PlatformAPI.DEBUGPlatformFreeFileMemory(Thread, MapReadResult.Contents);
        }

        loaded_bitmap TestBmp = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "test_1.bmp");
        RendererAPI->UploadTexture2DData(TestBmp.Pixels, TestBmp.Width, TestBmp.Height, BGRA_U8, false);

        loaded_bitmap TestBmp2 = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "default_8x8.bmp");
        RendererAPI->UploadTexture2DData(TestBmp2.Pixels, TestBmp2.Width, TestBmp2.Height, BGRA_U8, false);

        // TODO(Kevin): Load this default white lightmap from inside renderer code! Not from game code!
        // White_1x1 should have texture handle 0 so it is the default texture
        loaded_bitmap White_1x1 = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "white_1x1.bmp");
        RendererAPI->WhiteTexture_1x1 = 
            RendererAPI->UploadTexture2DData(White_1x1.Pixels, White_1x1.Width, White_1x1.Height, BGRA_U8, false);

        RendererAPI->UploadTexture2DData(GameState->Backdrop.Pixels, GameState->Backdrop.Width, GameState->Backdrop.Height, BGRA_U8, false);

        loaded_bitmap CustomFont0Bmp = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "fonts/anikki_8x8.bmp");
        font_handle RenderFont0 = CreateBitmapFont(CustomFont0Bmp, RendererAPI);
        GuiState->Fonts[GuiFontCurses6x9] = RenderFont0;

        loaded_bitmap CustomFont1Bmp = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "fonts/custom_font_0_6x9.bmp");
        font_handle RenderFont1 = CreateBitmapFont(CustomFont1Bmp, RendererAPI);
        GuiState->Fonts[GuiFontAnikki8x8] = RenderFont1;

        PlaceCamera(&GameState->Camera, vec3(-100, 60, 0), vec3(1,-0.1f,0));

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
    }
    real32 MouseLookSensitivity = 0.002f;
    vec2 AdjustedMouseAxes = Input->NormalizedMouseDelta * MouseLookSensitivity;
    MouseLookCamera(&GameState->Camera, AdjustedMouseAxes.x, AdjustedMouseAxes.y);

    //
    // This is all renderer and gui test code
    //
    uint32 RenderItemsCount = 12;
    uint8 *RenderItemsStart = ArenaCurrentAddress(&GameState->ScratchArena);

    for (uint32 i = 0; i < RenderItemsCount - 2; ++i)
    {
        render_item *Item = PushStruct(&GameState->ScratchArena, render_item);
        *Item = {};
        Item->Descriptor.Mesh = 0;
        Item->Descriptor.Albedo = i > 4 ? 2 : 1;
        Item->UseLightCubes = true;
        Item->UseLightmap = false;
        Item->WorldPosition = vec3(100, 30, (float(i) * 25) - 125);
        Item->WorldRotation = quat();
        Item->WorldScale = vec3(0.5f, 0.5f, 0.5f);
    }

    render_item *Item0 = PushStruct(&GameState->ScratchArena, render_item);
    *Item0 = {};
    Item0->Descriptor.Mesh = 1;
    Item0->Descriptor.Albedo = 2;
    Item0->Descriptor.Lightmap = 0;
    Item0->UseLightCubes = false;
    Item0->UseLightmap = true;
    Item0->WorldPosition = vec3();
    Item0->WorldRotation = quat();
    Item0->WorldScale = vec3(1.0f,1.0f,1.0f);

    render_item *Item1 = PushStruct(&GameState->ScratchArena, render_item);
    *Item1 = {};
    Item1->Descriptor.Mesh = 2;
    Item1->Descriptor.Albedo = 2;
    Item1->Descriptor.Lightmap = 0;
    Item1->UseLightCubes = false;
    Item1->UseLightmap = true;
    Item1->WorldPosition = vec3();
    Item1->WorldRotation = quat();
    Item1->WorldScale = vec3(1.0f,1.0f,1.0f);

    mat4 ViewFromWorld = CameraSpaceFromWorldSpace(&GameState->Camera);
    RendererAPI->PushViewMatrix(ViewFromWorld);
    RendererAPI->PushRenderItems((render_item *)RenderItemsStart, RenderItemsCount);

    uint32 CanvasW, CanvasH;
    RendererAPI->GetRenderTargetDimensions(&CanvasW, &CanvasH);
    GuiBeginFrame(GuiState, CanvasW, CanvasH, &GameState->ScratchArena, 
                  Input->MouseX, Input->MouseY, Input->MouseButtons[0], Input->MouseButtons[1]);
    GuiPushScissor(GuiState, { 250, 250, 1000, 100 });
    GuiPanel(GuiState, { 200, 200, 100, 100 }, 2);
    GuiPushScissor(GuiState, { 200, 200, 600, 125 });
    GuiPanel(GuiState, { 300, 300, 100, 100 }, vec4(1.0f, 0.0f, 1.0f, 1.0f));
    GuiPanel(GuiState, { 500, 100, 800, 400 }, 4);
    GuiPopScissor(GuiState);
    GuiPopScissor(GuiState);
    GuiPanel(GuiState, { 400, 400, 100, 100 }, vec4(0.0f, 1.0f, 1.0f, 1.0f));
    GuiPanel(GuiState, { Input->MouseX, Input->MouseY, 16, 16 }, vec4(1.0f, 1.0f, 0.0f, 1.0f));
    GuiText(GuiState, 50, 160, 36, GuiFontAnikki8x8, Align_Left, false, vec4(0,1,0,1), "Hello world");
    GuiText(GuiState, 50, 200, 36, GuiFontCurses6x9, Align_Left, false, vec4(0,1,1,0.5f), "Hello world");

    GuiPanel(GuiState, { 500, 100, 600, 600 }, 6);

    if (GuiButton(GuiState, 1, {100, 500, 200, 200}, vec4(0.0f,0.7f,0.0f,1.0f), 
                  vec4(0.0f,0.5f,0.0f,1.0f), vec4(0.0f,1.0f,0.0f,1.0f)))
    {
        GuiPanel(GuiState, { 500, 500, 200, 200 }, vec4(0,0,1,1));
    }

    RendererAPI->PushGuiDrawRequests(&GameState->ScratchArena, GuiState->DrawRequests, GuiState->DrawRequestsCount);
}

extern "C" GAME_GET_SOUND_SAMPLES(GameGetSoundSamples)
{
    game_state *GameState = (game_state *)Memory->PermanentStorage;
    GameOutputSound(GameState, SoundBuffer, 256);
}


