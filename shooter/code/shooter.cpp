#include "shooter.h"

#include "shooter_renderer.h"

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

internal void
DrawRectangle(game_offscreen_buffer *Buffer, vec2 vMin, vec2 vMax, real32 R, real32 G, real32 B)
{
    int32 MinX = RoundReal32ToInt32(vMin.x);
    int32 MinY = RoundReal32ToInt32(vMin.y);
    int32 MaxX = RoundReal32ToInt32(vMax.x);
    int32 MaxY = RoundReal32ToInt32(vMax.y);

    if (MinX < 0)
        MinX = 0;
    if (MinY < 0)
        MinY = 0;
    if (MaxX > Buffer->Width)
        MaxX = Buffer->Width;
    if (MaxY > Buffer->Height)
        MaxY = Buffer->Height;

    uint32 Color = 0xFF000000 |
                   ((uint32)RoundReal32ToInt32(R * 255.0f)) << 16 |
                   ((uint32)RoundReal32ToInt32(G * 255.0f)) << 8 |
                   ((uint32)RoundReal32ToInt32(B * 255.0f)) << 0;

    uint8 *Row = ((uint8 *)Buffer->Memory +
                  MinX*Buffer->BytesPerPixel +
                  MinY*Buffer->Pitch);

    for (int Y = MinY; Y < MaxY; ++Y)
    {
        uint32 *Pixel = (uint32 *)Row;
        for (int X = MinX; X < MaxX; ++X)
        {
            *Pixel++ = Color;
        }
        Row += Buffer->Pitch;
    }
}

internal void
DrawBitmap(game_offscreen_buffer *Buffer, loaded_bitmap *Bitmap,
           real32 RealX, real32 RealY,
           int32 AlignX = 0, int32 AlignY = 0)
{
    RealX -= (real32)AlignX;
    RealY -= (real32)AlignY;

    int32 MinX = RoundReal32ToInt32(RealX);
    int32 MinY = RoundReal32ToInt32(RealY);
    int32 MaxX = RoundReal32ToInt32(RealX + (real32)Bitmap->Width);
    int32 MaxY = RoundReal32ToInt32(RealY + (real32)Bitmap->Height);

    int32 SourceOffsetX = 0;
    int32 SourceOffsetY = 0;
    if (MinX < 0)
    {
        SourceOffsetX = -MinX;
        MinX = 0;
    }
    if (MinY < 0)
    {
        SourceOffsetY = -MinY;
        MinY = 0;
    }
    if (MaxX > Buffer->Width)
    {
        MaxX = Buffer->Width;
    }
    if (MaxY > Buffer->Height)
    {
        MaxY = Buffer->Height;
    }

    // TODO(Kevin): SourceRow needs to be changed based on clipping.
    uint32 *SourceRow = Bitmap->Pixels + (Bitmap->Width * (Bitmap->Height - 1));
    SourceRow += -SourceOffsetY * Bitmap->Width + SourceOffsetX;
    uint8 *DestRow = (uint8 *)Buffer->Memory +
                     MinX * Buffer->BytesPerPixel +
                     MinY * Buffer->Pitch;
    for (int Y = MinY; Y < MaxY; ++Y)
    {
        uint32 *Source = SourceRow;
        uint32 *Dest = (uint32 *)DestRow;
        for (int X = MinX; X < MaxX; ++X)
        {
            real32 A = (real32)((*Source >> 24) & 0xFF) / 255.0f;
            real32 DR = (real32)((*Dest >> 16) & 0xFF);
            real32 DG = (real32)((*Dest >> 8) & 0xFF);
            real32 DB = (real32)((*Dest >> 0) & 0xFF);
            real32 SR = (real32)((*Source >> 16) & 0xFF);
            real32 SG = (real32)((*Source >> 8) & 0xFF);
            real32 SB = (real32)((*Source >> 0) & 0xFF);
            
            // TODO(Kevin): Premultiplied alpha - this is not premultiplied alpha
            real32 R = (1.0f - A)*DR + A*SR;
            real32 G = (1.0f - A)*DG + A*SG;
            real32 B = (1.0f - A)*DB + A*SB;

            *Dest = ((uint32)(R + 0.5f) << 16) |
                    ((uint32)(G + 0.5f) << 8) |
                    ((uint32)(B + 0.5f) << 0);

            ++Dest;
            ++Source;
        }
        SourceRow -= Bitmap->Width;
        DestRow += Buffer->Pitch;
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

    game_state *GameState = (game_state *)Memory->PermanentStorage;
    if (!Memory->IsInitialized)
    {
        GameState->Backdrop = DEBUGLoadBMP(Thread, Memory->DEBUGPlatformReadEntireFile, "test/test_background.bmp");

        InitializeArena(&GameState->WorldArena, Memory->PermanentStorageSize - sizeof(game_state),
                        (uint8 *)Memory->PermanentStorage + sizeof(game_state));
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

    // DrawRectangle(Buffer, 0.f, 0.f, (real32)Buffer->Width, (real32)Buffer->Height, 1.f, 0.f, 1.f);
    DrawBitmap(Buffer, &GameState->Backdrop, 0, 0);
}

extern "C" GAME_GET_SOUND_SAMPLES(GameGetSoundSamples)
{
    game_state *GameState = (game_state *)Memory->PermanentStorage;
    GameOutputSound(GameState, SoundBuffer, 256);
}


