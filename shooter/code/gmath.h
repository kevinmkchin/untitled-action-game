/** 

    // TODO(Kevin): Test the few remaining untested quaternion functions
    // TODO(Kevin): SIMD
    // TODO(Kevin): review where the handedness is baked into the implementation...
    //              can I modify it?
    //              - In camera space having -Z as forward, +X as right, +Y as up
    //              - and subsequently clip space transformation using the same handedness
    //              - The only other thing is rotation I believe? yes
    // TODO(Kevin): What am I actually using from math? sinf, cosf, sqrtf, asinf, acosf, atan2f
    // TODO(Kevin): Slerp tests are shit - comparing against incorrect implementation...also result even 
    //              seems to vary b/w correct implementations

    gmath.h

    3D math and linear algebra library for games (https://kevin.gd/)

Features:
        - Vectors
        - Matrices
        - Quaternions
        - Helpers for Transform, View, and Projection matrices

Standards:
        - RIGHT-HANDED system with orthogonal axes XYZ
        - Theoretically, "Forward", "Right", and "Up" semantics can be assigned to any of
          the axes provided they are orthogonal, but I've not tested this
        - By default, +X is forward, +Y is up, +Z is right
          Cross(Forward, Up) = Right; Cross(Up, Right) = Forward; Cross(Right, Forward) = Up
        - "Roll", "Yaw", "Pitch" are semantically Euler rotations around the "Forward, "Up",
          and "Right" axes respectively. All Euler rotations are in radians.
        - Vectors and matrices are stored in COLUMN-MAJOR order.

*/
#ifndef _INCLUDE_GMATH_LIBRARY_H_
#define _INCLUDE_GMATH_LIBRARY_H_

// #include <math.h>
#include "shooter_intrinsics.h"

#define GMATH_X_AXIS        vec3(1.f,0.f,0.f)
#define GMATH_Y_AXIS        vec3(0.f,1.f,0.f)
#define GMATH_Z_AXIS        vec3(0.f,0.f,1.f)
#define GMATH_FORWARD       GMATH_X_AXIS
#define GMATH_BACKWARD      -GMATH_FORWARD
#define GMATH_UP            GMATH_Y_AXIS
#define GMATH_DOWN          -GMATH_UP
#define GMATH_RIGHT         GMATH_Z_AXIS
#define GMATH_LEFT          -GMATH_RIGHT
#define GMATH_PI            3.141592653589f
#define GMATH_ONEOVERPI     0.318309886f
#define GMATH_TWOPI         6.28318530718f
#define GMATH_ONEOVERTWOPI  0.159154943f
#define GMATH_HALFPI        1.570796f
#define GMATH_QUARTERPI     0.785398f
#define GMATH_D2R           GMATH_DEG2RAD
#define GMATH_R2D           GMATH_RAD2DEG
#define GMATH_DEG2RAD       0.0174532925f // Degrees * GMATH_DEG2RAD = Radians
#define GMATH_RAD2DEG       57.2958f      // Radians * GMATH_RAD2DEG = Degrees

#define GMATH_min(A, B) ((A) < (B) ? (A) : (B))
#define GMATH_max(A, B) ((A) > (B) ? (A) : (B))
#define GMATH_abs(A)    ((A) < 0.f ? (-(A)) : (A))
#define GMATH_sign(X)   ((X > 0) - (X < 0))
#define GMATH_clamp(X, Lower, Upper) GMATH_max((Lower), GMATH_min((Upper), (X)))
#define GMATH_Clamp01(t) GMATH_clamp(t, 0.0f, 1.0f)

inline float Square(float A)
{
    float Result = A*A;
    return Result;
}

inline float Blend(float A, float B, float t)
{
    return A + t * (B - A);
}

struct vec4;
union vec3;
struct vec2;
union mat3;
union mat4;
struct quat;

union vec3
{
    struct { float X, Y, Z; };
    struct { float x, y, z; };

    vec3()
        : x(0.f)
        , y(0.f)
        , z(0.f)
    {}

    vec3(float X, float Y, float Z)
        : x(X)
        , y(Y)
        , z(Z)
    {}

    vec3(vec2 A, float B);

    float &operator[] (int Row) { float *Data = (float *)this; return Data[Row]; }
};

struct vec4
{
    union
    {
        struct { float X, Y, Z; };
        struct { float x, y, z; };
        union { vec3 XYZ; vec3 xyz; };
    };
    union { float W; float w; };

    vec4()
        : x(0.f)
        , y(0.f)
        , z(0.f)
        , w(0.f)
    {}

    vec4(float X, float Y, float Z, float W)
        : x(X)
        , y(Y)
        , z(Z)
        , w(W)
    {}

    vec4(vec3 XYZ, float W)
        : xyz(XYZ)
        , w(W)
    {}

    float &operator[] (int Row) { float *Data = (float *)this; return Data[Row]; }
};

struct vec2
{
    union
    {
        struct { float X, Y; };
        struct { float x, y; };
    };

    vec2() : x(0.f), y(0.f) {}
    vec2(float X, float Y) : x(X), y(Y) {}

    float &operator[] (int Row) { float *Data = (float *)this; return Data[Row]; }
};

struct ivec3
{
    union
    {
        struct { int X, Y, Z; };
        struct { int x, y, z; };
    };

    ivec3() : x(0), y(0), z(0) {}
    ivec3(int X, int Y, int Z) : x(X), y(Y), z(Z) {}
};

struct ivec2
{
    union
    {
        struct { int X, Y; };
        struct { int x, y; };
    };

    ivec2() : x(0), y(0) {}
    ivec2(int X, int Y) : x(X), y(Y) {}
};

/*
    Matrices

    Column-major order
        [00][04][08][12]
        [01][05][09][13]
        [02][06][10][14]
        [03][07][11][15]

    Access like so: mat4[col][row]
    Memory layout like so:
        0x????0000  col 0 row 0  float
        0x????0004  col 0 row 1  float
        0x????0008  col 0 row 2  float
        0x????000c  col 0 row 3  float
        0x????0010  col 1 row 0  float
        0x????0014  col 1 row 1  float
        ...
*/
union mat4
{
    float e[16];
    vec4 columns[4];

    /** Constructs a 4x4 identity matrix */
    mat4();
    mat4(float e00, float e01, float e02, float e03,
         float e10, float e11, float e12, float e13,
         float e20, float e21, float e22, float e23,
         float e30, float e31, float e32, float e33);
    mat4(vec4 A, vec4 B, vec4 C, vec4 D);

    inline void Clear() { *this = {}; }
    inline mat4 GetTranspose();
    inline mat4 GetInverse();

    float *ptr() { return((float*)this); }
    vec4 &operator[] (int col) { return columns[col]; }
};

union mat3
{
    float e[9];
    vec3 columns[3];

    /** Constructs a 3x3 identity matrix */
    mat3();
    mat3(float e00, float e01, float e02,
         float e10, float e11, float e12,
         float e20, float e21, float e22);
    mat3(vec3 A, vec3 B, vec3 C);

    inline void Clear() { *this = {}; }
    inline mat3 GetTranspose();
    inline mat3 GetInverse();

    float *ptr() { return((float*)this); }
    vec3 &operator[] (int col) { return columns[col]; }
};

struct quat
{
    union
    {
        struct { float W, X, Y, Z; };
        struct { float w, x, y, z; };
    };

    /* Construct an identity quaternion */
    quat() : w(1.f), x(0.f), y(0.f), z(0.f) {}
    quat(float W, float X, float Y, float Z);
    /* Construct a quaternion representing a single axis rotation */
    quat(float Angle, vec3 Axis);

    /* Gets the conjugate of given quaternion. Doesn't check that Magnitude is 1. */
    inline quat GetConjugate();
    /* Gets the conjugate of the given quaternion with Magnitude 1 */
    inline quat GetInverseUnitQuaternion();
};

/*
    Vectors

        Dot (vec, vec -> float)
        Cross (vec, vec -> vec)
        Length (vec -> float)
        Normalize (vec -> vec)
        Blend (vec, vec, float -> vec)

    Examples

        vec2 A;
        vec2 NegA = -A;
        vec3 LightPos, SurfacePos, SurfaceNormal, Forward, Up;
        vec3 VectorToLight = LightPos - SurfacePos;
        float Lambertian = Dot(Normalize(VectorToLight), SurfaceNormal);
        vec3 Right = Cross(Forward, Up);
        A += vec2(x,y);
        A -= NegA;
        if (A != NegA || A == A)
        {
            VectorToLight /= Length(VectorToLight);
        }

*/

inline vec2 Add(vec2 A, vec2 B);
inline vec3 Add(vec3 A, vec3 B);
inline vec4 Add(vec4 A, vec4 B);
inline vec2 Sub(vec2 A, vec2 B);
inline vec3 Sub(vec3 A, vec3 B);
inline vec4 Sub(vec4 A, vec4 B);
inline vec2 Mul(vec2 A, float B);
inline vec3 Mul(vec3 A, float B);
inline vec4 Mul(vec4 A, float B);
inline vec2 Div(vec2 A, float B);
inline vec3 Div(vec3 A, float B);
inline vec4 Div(vec4 A, float B);
inline mat3 Mul(mat3 &A, mat3 &B);
inline vec3 Mul(mat3 &A, vec3 V);
inline mat4 Mul(mat4 &A, mat4 &B);
inline vec4 Mul(mat4 &A, vec4 V);

inline float Dot(vec2 A, vec2 B);
inline float Dot(vec3 A, vec3 B);
inline float Dot(vec4 A, vec4 B);
inline float Cross(vec2 A, vec2 B);
inline vec3 Cross(vec3 A, vec3 B);
inline float Length(vec2 A);
inline float Length(vec3 A);
inline float Length(vec4 A);
inline float Magnitude(vec2 A);
inline float Magnitude(vec3 A);
inline float Magnitude(vec4 A);
inline vec2 Normalize(vec2 A);
inline vec3 Normalize(vec3 A);
inline vec4 Normalize(vec4 A);
inline vec3 Blend(vec3 A, vec3 B, float t);
inline vec4 Blend(vec4 A, vec4 B, float t);

inline ivec2 operator-(ivec2 A);
inline ivec2 operator+(ivec2 A, ivec2 B);
inline ivec2 operator-(ivec2 A, ivec2 B);
inline ivec2 operator*(ivec2 A, float B);
inline ivec2 operator*(float B, ivec2 A);
inline ivec2 operator/(ivec2 A, float B);
inline ivec2 &operator+=(ivec2 &A, ivec2 B);
inline ivec2 &operator-=(ivec2 &A, ivec2 B);
inline ivec2 &operator*=(ivec2 &A, float B);
inline ivec2 &operator/=(ivec2 &A, float B);
inline bool operator==(ivec2 &A, ivec2 &B);
inline bool operator!=(ivec2 &A, ivec2 &B);

inline ivec3 operator-(ivec3 A);
inline ivec3 operator+(ivec3 A, ivec3 B);
inline ivec3 operator-(ivec3 A, ivec3 B);
inline ivec3 operator*(ivec3 A, int B);
inline ivec3 operator*(int B, ivec3 A);
inline ivec3 operator/(ivec3 A, int B);
inline ivec3 &operator+=(ivec3& A, ivec3 B);
inline ivec3 &operator-=(ivec3& A, ivec3 B);
inline ivec3 &operator*=(ivec3& A, int B);
inline ivec3 &operator/=(ivec3& A, int B);
inline bool operator==(ivec3& A, ivec3& B);
inline bool operator!=(ivec3& A, ivec3& B);

inline vec2 operator-(vec2 A);
inline vec2 operator+(vec2 A, vec2 B);
inline vec2 operator-(vec2 A, vec2 B);
inline vec2 operator*(vec2 A, float B);
inline vec2 operator*(float B, vec2 A);
inline vec2 operator/(vec2 A, float B);
inline vec2 &operator+=(vec2& A, vec2 B);
inline vec2 &operator-=(vec2& A, vec2 B);
inline vec2 &operator*=(vec2& A, float B);
inline vec2 &operator/=(vec2& A, float B);
inline bool operator==(vec2& A, vec2& B);
inline bool operator!=(vec2& A, vec2& B);

inline vec3 operator-(vec3 A);
inline vec3 operator+(vec3 A, vec3 B);
inline vec3 operator-(vec3 A, vec3 B);
inline vec3 operator*(vec3 A, float B);
inline vec3 operator*(float B, vec3 A);
inline vec3 operator/(vec3 A, float B);
inline vec3 &operator+=(vec3& A, vec3 B);
inline vec3 &operator-=(vec3& A, vec3 B);
inline vec3 &operator*=(vec3& A, float B);
inline vec3 &operator/=(vec3& A, float B);
inline bool operator==(vec3& A, vec3& B);
inline bool operator!=(vec3& A, vec3& B);

inline vec4 operator-(vec4 A);
inline vec4 operator+(vec4 A, vec4 B);
inline vec4 operator-(vec4 A, vec4 B);
inline vec4 operator*(vec4 A, float B);
inline vec4 operator*(float B, vec4 A);
inline vec4 operator/(vec4 A, float B);
inline vec4 &operator+=(vec4& A, vec4 B);
inline vec4 &operator-=(vec4& A, vec4 B);
inline vec4 &operator*=(vec4& A, float B);
inline vec4 &operator/=(vec4& A, float B);
inline bool operator==(vec4& A, vec4& B);
inline bool operator!=(vec4& A, vec4& B);

/*
    Matrices

    Examples

        vec4 WorldPoint = ScaleMatrix(S) * RotationMatrix(R) * TranslationMatrix(X,Y,Z) * vec4(ModelPoint, 1);

*/
inline vec3 operator*(mat3 A, vec3 V);
inline mat3 operator*(mat3 A, mat3 B);
inline mat3 &operator*=(mat3 &A, mat3 B);
inline vec4 operator*(mat4 A, vec4 V);
inline mat4 operator*(mat4 A, mat4 B);
inline mat4 &operator*=(mat4 &A, mat4 B);

inline mat3 OuterProduct(vec3 A, vec3 B);
inline mat4 OuterProduct(vec4 A, vec4 B);
inline mat3 Mat3FromMat4(mat4 From);
inline mat4 Mat4FromMat3(mat3 From);

/*
    Quaternions

    Examples

        quat BoneRotationFrame0 = BoneRotations[AnimFrame0].Rotation;
        quat BoneRotationFrame1 = BoneRotations[AnimFrame1].Rotation;
        Slerp(BoneRotationFrame0, BoneRotationFrame1, tBetweenAnimationFrames);

        vec3 DefaultLightDirection = { 0, -1, 0 }; // pointing -Y direction when no rotation
        quat LightRotation = RotationFromTo(DefaultLightDirection, DesiredLightDirection);

*/
// inline quat Add(quat A, quat B);
// inline quat Sub(quat A, quat B);
// inline quat Scale(quat A, float Scale);
inline quat Div(quat A, float Scale);
inline quat Normalize(quat A);

/* Combine rotations in the order A then B */
inline quat HamiltonProduct(quat B, quat A);
inline quat operator*(quat B, quat A) { return(HamiltonProduct(B, A)); }
/* Cosine of the angle between the two rotations represented by the unit quaternions. */
inline float Dot(quat UnitQuaternionA, quat UnitQuaternionB);
/* Checks if Dot product of A and B is within +/- epsilon */
inline bool Similar(quat A, quat B, float Epsilon = 0.0001f);
/* Construct a 3x3 rotation matrix from the given quaternion */
inline mat3 Mat3FromQuat(quat Rotation);
/* Construct a 4x4 rotation matrix from the given quaternion */
inline mat4 Mat4FromQuat(quat Rotation);
/* Constructs an orientation quaternion from the rotation matrix */
inline quat QuatFromMat4(mat4 RotationMatrix);
/* Apply Euler angle rotations around three orthogonal axes in right handed system in
   the order X->Z->Y. The order in the function name is applied right to left.
   Equivalent to Qy * (Qz * Qx) where each Q represent single axis rotation. */
inline quat QuatFromEuler_YZX(float X, float Y, float Z);
inline quat QuatFromEuler_YZX(vec3 EulerAnglesRadian);
/* Rotates vector by the quaternion represented rotation. Rotates around the origin. */
inline vec3 RotateVector(vec3 Vector, quat Rotation); // TODO(Kevin): NOT TESTED
/* Creates a unit quaternion which rotates from FromDirection to ToDirection
   Use this to rotate a transform so that one of its axes, such as the y-axis,
   follows a target direction, ToDirection. */
inline quat RotationFromTo(vec3 FromDirection, vec3 ToDirection); // TODO(Kevin): NOT TESTED
/* Finds the difference such that b = difference * a */
// inline quat RotationDifference(quat a, quat b); // TODO(Kevin): NOT TESTED
/* Spherical linear interpolation */
inline quat Slerp(quat From, quat To, float t); // TODO(Kevin): NOT TESTED -> cuz i made changes to the helpers
/* Returns a yaw-pitch only rotation with the specified forward and upwards directions. */
inline quat LookRotation1(vec3 LookDirection, vec3 Up);
inline void LookRotation2(vec3 LookDirection, real32 *Yaw, real32 *Pitch);
inline quat LookRotation3(vec3 LookDirection);
/* Returns an orientation that faces the given direction. The return value represents
   the rotation from the world's forward direction (forward vector) in order to face
   the same direction as the given direction. The resulting orientation will point in
   the desired direction but the degree of rotation around the desired direction may vary. */
inline quat OrientationFromDirection(vec3 Direction);
/* Returns the direction of this orientation. The world's forward direction (forward vector)
   rotated by the orientation is the direction. */
inline vec3 DirectionFromOrientation(quat Orientation);

// TODO(Kevin): Implement
/* This will not produce consistent results. There is more than one way to represent
   a rotation using Euler angles.
   This will not work when the Z euler angle is within [90, 270] degrees. This is a
   limitation with euler angles: euler angles (of any type) have a singularity. Unity's
   Quaternion.eulerAngle also experiences the same limitation, so I don't think there is
   anything I can do about it. Just whenever possible, avoid using euler angles. */
// inline vec3 EulerFromQuat(quat R);


/*
    Translation matrix

    Generates translation matrix for given delta x delta y delta z
    https://en.wikipedia.org/wiki/Translation_(geometry)#Matrix_resentation
*/
inline mat4 TranslationMatrix(float X, float Y, float Z);
inline mat4 TranslationMatrix(vec3 Translation);
inline mat3 TranslationMatrix2D(vec2 Translation);

/*
    Rotation matrix

    Generates rotation matrix for given quaternion represented rotation
    https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
*/
inline mat4 RotationMatrix(quat Rotation);
inline mat3 RotationMatrix2D(float Rotation);

/*
    Scaling matrix

    Generates scaling matrix for given x y z scales
    https://en.wikipedia.org/wiki/Scaling_(geometry)#Using_homogeneous_coordinates
*/
inline mat4 ScaleMatrix(float Scale);
inline mat4 ScaleMatrix(float ScaleX, float ScaleY, float ScaleZ);
inline mat4 ScaleMatrix(vec3 Scale);
inline mat3 ScaleMatrix2D(vec2 Scale);

/*
    View

    This matrix takes points into camera-relative coordinate space.

    In camera/view space:
        Right = +X
        Up = +Y
        Forward = -Z

*/
inline mat4 CameraSpaceMatrixLookAt(vec3 Eye, vec3 Target, vec3 EyeUpVector);

/*
    Projection

    Clip space is the coordinate space before perspective divide (W) where the viewing frustum
    is defined by the clipping planes at -W <= X <= W, -W <= Y <= W, Z_NEAR_CLIP <= Z <= W.
    Z_NEAR_CLIP is 0 in DirectX and Vulkan, -W in OpenGL.

    See CameraSpaceMatrixLookAt for camera/view space definition. Uses the same handedness as camera space (RH).

    Perspective divide is performed by the GPU to get Normalized Device Coordinates.

    VerticalFOV:  Specifies the field of view angle in the y direction. Expressed in radians.
    AspectRatio:  Specifies the aspect ratio that determines the field of view in the x direction. (width / height).
    NearClip:     Specifies the distance from the viewer to the near clipping plane (always positive).
    FarClip:      Specifies the distance from the viewer to the far clipping plane (always positive).
    NearClipZero: Sets Z_NEAR_CLIP to 0 instead of -W
*/
inline mat4 ClipSpaceFromCameraSpace_PerspectiveProjection(float VerticalFOV, float AspectRatio,
                                                           float NearClip, float FarClip, bool NearClipZero);

// TODO(Kevin): Add NearClipZero flag for this orthographic projection matrix
// /** Creates a 4x4 matrix for an orthographic parallel viewing volume, using right-handed coordinates.
//     https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml
//     The near and far clip planes correspond to z normalized device coordinates of -1 and +1
//     respectively (OpenGL clip volume definition). */
// inline mat4 ProjectionMatrixOrthographic(float Left, float Right, float Bottom, float Top,
//                                          float NearClip, float FarClip)

/*  Creates a 4x4 matrix for projecting two-dimensional coordinates onto the screen.
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluOrtho2D.xml
    Left, Right: Specify the coordinates for the Left and Right vertical clipping planes.
    Bottom, Top: Specify the coordinates for the Bottom and Top horizontal clipping planes.
    e.g. ProjectionMatrixOrthographicNoZ(0.f, 1920.f, 1080.f, 0.f);
    e.g. ProjectionMatrixOrthographicNoZ(0.f, 1920.f, 0.f, 1080.f); <-- vertical flip */
inline mat4 ProjectionMatrixOrthographicNoZ(float Left, float Right, float Bottom, float Top);

/*  Creates a 3x3 matrix for projecting two-dimensional coordinates onto the screen.
    This is used for transforming vec3s where vec3.x is the X coordinate, vec3.y is
    the Y coordinate, and vec3.z is 1 for points and 0 for vectors. This is different
    from ProjectionMatrixOrthographicNoZ which gives a 4x4 matrix used for transforming
    vec4s which includes a Z coordinate which is discarded. */
inline mat3 ProjectionMatrixOrthographic2D(float Left, float Right, float Bottom, float Top);

#include "gmath.inl"

#endif // _INCLUDE_GMATH_LIBRARY_H_

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2025 Myungkeon Kevin Chin
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/
