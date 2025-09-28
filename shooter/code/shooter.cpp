#include "shooter.h"
#include "shooter_renderer.h"
#include "shooter_shared.h"

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

extern "C" GAME_UPDATE_AND_RENDER(GameUpdateAndRender)
{
    Assert((&Input->Controllers[0].Terminator - &Input->Controllers[0].Buttons[0]) == 
           ArrayCount(Input->Controllers[0].Buttons));
    Assert(sizeof(game_state) <= Memory->PermanentStorageSize);

    uint32 Result = Renderer->SomeTestServiceToGame(64);

    game_state *GameState = (game_state *)Memory->PermanentStorage;
    if (!Memory->IsInitialized)
    {
        GameState->Backdrop = DEBUGLoadBMP(Thread, Memory->PlatformAPI.DEBUGPlatformReadEntireFile, "test/test_background.bmp");

        InitializeArena(&GameState->WorldArena, Memory->PermanentStorageSize - sizeof(game_state),
                        (uint8 *)Memory->PermanentStorage + sizeof(game_state));

        // InitializeArena(&GameState->ScratchArena, Memory->TransientStorageSize,
        //                 (uint8 *)Memory->TransientStorage);

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
            Renderer->UploadTexture2DData(LightmapData, SafeCastToUInt32(LightmapW), SafeCastToUInt32(LightmapH),
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

                Renderer->UploadMeshData(vb, VertexCount, ib, VertexCount);
                free(vb);
                free(ib);
            }


            Memory->PlatformAPI.DEBUGPlatformFreeFileMemory(Thread, MapReadResult.Contents);
        }

        Memory->IsInitialized = true;
    }

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
}

extern "C" GAME_GET_SOUND_SAMPLES(GameGetSoundSamples)
{
    game_state *GameState = (game_state *)Memory->PermanentStorage;
    GameOutputSound(GameState, SoundBuffer, 256);
}


