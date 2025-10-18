#ifndef SHOOTER_MESHES_H
#define SHOOTER_MESHES_H

#include "shooter_intrinsics.h"

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

// NOTE(Kevin): Having sort of cylindrical mesh appears to help with avoiding ghost collisions along walls.
#define CylinderMesh8SidedVertexCount 66 // 4*8+2+2*8*2
#define CylinderMesh8SidedIndexCount 96 //6*8+3*8*2
#define CylinderMesh8SidedPointCount 16 // 2*8

void BuildCylinderMesh8Sided(vec3 *PointCloud, PNUV2_vertex *Vertices, u32 *Indices)
{
    /*
        NOTE(Kevin): Builds the point cloud, vertices, and indices for a cylinder
        with its points relative to the center of its base with radius and height
        of length 1.
    */

    u32 Sides = 8;
    real32 R = 1.0f;
    real32 H = 1.0f;
    real32 Step = GMATH_TWOPI / (real32)Sides;

    u32 PointCount = 0;
    u32 VertexCount = 0;
    u32 IndexCount = 0;

    for (u32 Side = 0; Side < Sides; ++Side)
    {
        real32 Hypotenuse = R;
        real32 X0 = Sin(Step*Side)*Hypotenuse;
        real32 Z0 = Cos(Step*Side)*Hypotenuse;
        real32 X1 = Sin(Step*(Side+1))*Hypotenuse;
        real32 Z1 = Cos(Step*(Side+1))*Hypotenuse;
        real32 YBot = 0.f;
        real32 YTop = H;

        vec3 P0 = { X0, YBot, Z0 };
        vec3 P1 = { X1, YBot, Z1 };
        vec3 P2 = { X1, YTop, Z1 };
        vec3 P3 = { X0, YTop, Z0 };

        vec3 N = Normalize(Cross(P1-P0, P3-P0));

        vec2 UV0 = { 0.0f, 0.0f };
        vec2 UV1 = { 1.0f, 0.0f };
        vec2 UV2 = { 1.0f, 1.0f };
        vec2 UV3 = { 0.0f, 1.0f };

        if (PointCloud)
        {
            PointCloud[PointCount++] = P0;
            PointCloud[PointCount++] = P3;
        }

        Vertices[VertexCount++] = { P0, N, UV0, {0,0} };
        Vertices[VertexCount++] = { P1, N, UV1, {0,0} };
        Vertices[VertexCount++] = { P2, N, UV2, {0,0} };
        Vertices[VertexCount++] = { P3, N, UV3, {0,0} };

        u32 I0 = VertexCount - 4;
        u32 I1 = VertexCount - 3;
        u32 I2 = VertexCount - 2;
        u32 I3 = VertexCount - 1;

        Indices[IndexCount++] = I0;
        Indices[IndexCount++] = I1;
        Indices[IndexCount++] = I2;
        Indices[IndexCount++] = I0;
        Indices[IndexCount++] = I2;
        Indices[IndexCount++] = I3;
    }

    vec3 CenterBot = vec3();
    vec3 CenterTop = vec3(0,H,0);
    vec3 NBot = vec3(0,-1,0);
    vec3 NTop = vec3(0,1,0);
    Vertices[VertexCount++] = { CenterBot, NBot, { 0.0f, 0.0f }, {0,0} };
    Vertices[VertexCount++] = { CenterTop, NTop, { 0.0f, 0.0f }, {0,0} };
    u32 ICenterBot = VertexCount - 2;
    u32 ICenterTop = VertexCount - 1;

    for (u32 Side = 0; Side < Sides; ++Side)
    {
        real32 Hypotenuse = R;
        real32 X0 = Sin(Step*Side)*Hypotenuse;
        real32 Z0 = Cos(Step*Side)*Hypotenuse;
        real32 X1 = Sin(Step*(Side+1))*Hypotenuse;
        real32 Z1 = Cos(Step*(Side+1))*Hypotenuse;
        real32 YBot = 0.f;
        real32 YTop = H;

        vec3 P0 = { X0, YBot, Z0 };
        vec3 P1 = { X1, YBot, Z1 };

        vec3 P2 = { X0, YTop, Z0 };
        vec3 P3 = { X1, YTop, Z1 };

        vec2 UV0 = { X0, Z0 };
        vec2 UV1 = { X1, Z1 };

        Vertices[VertexCount++] = { P0, NBot, UV0, {0,0} };
        Vertices[VertexCount++] = { P1, NBot, UV1, {0,0} };
        Vertices[VertexCount++] = { P2, NTop, UV0, {0,0} };
        Vertices[VertexCount++] = { P3, NTop, UV1, {0,0} };

        u32 IBot0 = VertexCount - 4;
        u32 IBot1 = VertexCount - 3;
        u32 ITop0 = VertexCount - 2;
        u32 ITop1 = VertexCount - 1;

        Indices[IndexCount++] = ICenterBot;
        Indices[IndexCount++] = IBot1;
        Indices[IndexCount++] = IBot0;
        Indices[IndexCount++] = ICenterTop;
        Indices[IndexCount++] = ITop0;
        Indices[IndexCount++] = ITop1;
    }
}

#endif // SHOOTER_MESHES_H
