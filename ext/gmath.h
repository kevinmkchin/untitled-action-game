/** 

    gmath.h

    A 3D math and linear algebra library for games written by Kevin Chin (https://kevch.in/)
    Just include and use:
        #include "gmath.h"

Features:
        - Vectors & vector operations
        - Matrices & matrix operations
        - Quaternions & quaternion operations
        - Methods to create transformation matrices
        - Methods to create projection matrices
        - Method to create view matrix
        - Spherical linear interpolation & vector linear interpolation

STANDARDS:
    This library imposes some standards and rules for 3D math:
        - Positive X axis is forward vector. Positive Y axis is up vector. Positive Z
          axis is right vector. Therefore, the orientation of space uses RIGHT-HANDedness:
            Cross(Forward, Up) = Right; Cross(Up, Right) = Forward; Cross(Right, Forward) = Up
        - Roll is rotation around X axis, Pitch is rotation around Z axis, and Yaw is rotation
          around Y axis. Euler angles follow RIGHT-HANDedness. When converting from Euler angles
          to a Quaternion orientation, the Euler angles are applied in X-Z-Y order (this is
          application order not hierarchy order - the conversion behaviour applies X/Roll, then
          applies Z/Pitch, then applies Y/Yaw).
        - Vectors and matrices are stored in COLUMN-MAJOR order.
        - Quaternions: gmath uses Quaternions to represent all rotations.
          They are compact, don't suffer from gimbal lock and can easily be interpolated.

*/
#ifndef _INCLUDE_GMATH_LIBRARY_H_
#define _INCLUDE_GMATH_LIBRARY_H_

#include <cstdlib>
#include <cmath>

#define GM_FORWARD_VECTOR vec3(1.f,0.f,0.f)
#define GM_BACKWARD_VECTOR (-GM_FORWARD_VECTOR)
#define GM_UP_VECTOR vec3(0.f,1.f,0.f)
#define GM_DOWN_VECTOR (-GM_UP_VECTOR)
#define GM_RIGHT_VECTOR vec3(0.f,0.f,1.f)
#define GM_LEFT_VECTOR (-GM_RIGHT_VECTOR)
#define GM_PI 3.141592653589f
#define GM_ONEOVERPI 0.318309886f
#define GM_TWOPI 6.28318530718f
#define GM_ONEOVERTWOPI 0.159154943f
#define GM_HALFPI 1.570796f
#define GM_QUARTERPI 0.785398f
#define GM_D2R GM_DEG2RAD
#define GM_R2D GM_RAD2DEG
#define GM_DEG2RAD 0.0174532925f  // degrees * GM_DEG2RAD = radians
#define GM_RAD2DEG 57.2958f       // radians * GM_RAD2DEG = degrees

#define GM_min(a, b) ((a) < (b) ? (a) : (b))
#define GM_max(a, b) ((a) > (b) ? (a) : (b))
#define GM_abs(a) ((a) < 0.f ? (-(a)) : (a))
#define GM_sign(x) ((x > 0) - (x < 0))
#define GM_clamp(x, lower, upper) GM_max((lower), GM_min((upper), (x)))


struct vec4;
union vec3;
struct vec2;
union mat3;
union mat4;
struct quat;

union vec3
{
    struct
    {
        float x, y, z;
    };
    struct
    {
        float roll, yaw, pitch;
    };

    vec3()
        : x(0.f)
        , y(0.f)
        , z(0.f)
    {}

    vec3(float xVal, float yVal, float zVal)
        : x(xVal)
        , y(yVal)
        , z(zVal)
    {}

    vec3(vec2 a, float b);

    float& operator[] (int row)
    {
        float* address = (float*)this;
        return address[row];
    }
    const float& operator[] (int row) const
    {
        float* address = (float*)this;
        return address[row];
    }
};

struct vec4
{
    union
    {
        struct
        {
            float x, y, z;
        };
        vec3 xyz;
    };
    float w;

    vec4()
        : x(0.f)
        , y(0.f)
        , z(0.f)
        , w(0.f)
    {}

    vec4(float xVal, float yVal, float zVal, float wVal)
        : x(xVal)
        , y(yVal)
        , z(zVal)
        , w(wVal)
    {}

    vec4(vec3 v3, float wVal)
        : xyz(v3)
        , w(wVal)
    {}

    float& operator[] (int row)
    {
        float* address = (float*)this;
        return address[row];
    }
    const float& operator[] (int row) const
    {
        float* address = (float*)this;
        return address[row];
    }
};

struct vec2
{
    float x;
    float y;

    vec2()
        : x(0.f)
        , y(0.f)
    {}

    vec2(float xVal, float yVal)
        : x(xVal)
        , y(yVal)
    {}

    vec2(vec3 a)
        : x(a.x)
        , y(a.y)
    {}

    float& operator[] (int row)
    {
        float* address = (float*)this;
        return address[row];
    }
    const float& operator[] (int row) const
    {
        float* address = (float*)this;
        return address[row];
    }
};

struct ivec3
{
    int x;
    int y;
    int z;

    ivec3()
        : x(0)
        , y(0)
        , z(0)
    {}

    ivec3(int xVal, int yVal, int zVal)
        : x(xVal)
        , y(yVal)
        , z(zVal)
    {}
};

struct ivec2
{
    int x;
    int y;

    ivec2()
        : x(0)
        , y(0)
    {}

    ivec2(int xVal, int yVal)
        : x(xVal)
        , y(yVal)
    {}
};

/** - Matrices -
    Column-major order
    Access like so: mat4[col][row]
    Laid in memory like so:
    0x????0000  col 0 row 0 : float
    0x????0004  col 0 row 1 : float
    0x????0008  col 0 row 2 : float
    0x????000c  col 0 row 3 : float
    0x????0010  col 1 row 0 : float
    0x????0014  col 1 row 1 : float
    0x????0018  col 1 row 2 : float
    0x????001c  col 1 row 3 : float
    0x????0020  col 2 row 0 : float
    0x????0024  col 2 row 1 : float
    0x????0028  col 2 row 2 : float
    0x????002c  col 2 row 3 : float
    0x????0030  col 3 row 0 : float
    0x????0034  col 3 row 1 : float
    0x????0038  col 3 row 2 : float
    0x????003c  col 3 row 3 : float
    Get float array address through ptr()
    Can use initializer list like mat4 m = { 00, 01, ... , 33 };

    [00][04][08][12]
    [01][05][09][13]
    [02][06][10][14]
    [03][07][11][15]
*/

union mat4
{
    float e[16];
    vec4 columns[4];

    /** Constructs a 4x4 identity matrix */
    mat4()
    {
        columns[0] = vec4(1.f, 0.f, 0.f, 0.f);
        columns[1] = vec4(0.f, 1.f, 0.f, 0.f);
        columns[2] = vec4(0.f, 0.f, 1.f, 0.f);
        columns[3] = vec4(0.f, 0.f, 0.f, 1.f);
    }

    mat4(float e00, float e01, float e02, float e03,
         float e10, float e11, float e12, float e13,
         float e20, float e21, float e22, float e23,
         float e30, float e31, float e32, float e33);

    mat4(const mat3& from);

    mat4(vec4 a, vec4 b, vec4 c, vec4 d)
    {
        columns[0] = a;
        columns[1] = b;
        columns[2] = c;
        columns[3] = d;
    }

    /** Construct a 4x4 rotation matrix from the given quaternion */
    mat4(quat q);

    inline void Empty()
    {
        *this = { 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
    }

    inline mat4 GetTranspose() const;

    inline mat4 GetInverse() const;

    /** Returns a float pointer to the memory layout of the matrix. Useful
        for uploading data to graphics API. OpenGL uses column-major order.*/
    float* ptr() const { return((float*)this); }

    vec4& operator[] (int col) { return columns[col]; }
    const vec4& operator[] (int col) const { return columns[col]; }
};

union mat3
{
    float e[9];
    vec3 columns[3];

    /** Constructs a 3x3 identity matrix */
    mat3()
    {
        columns[0] = vec3(1.f, 0.f, 0.f);
        columns[1] = vec3(0.f, 1.f, 0.f);
        columns[2] = vec3(0.f, 0.f, 1.f);
    }

    mat3(float e00, float e01, float e02,
         float e10, float e11, float e12,
         float e20, float e21, float e22);

    mat3(const mat4& from);

    mat3(vec3 a, vec3 b, vec3 c)
    {
        columns[0] = a;
        columns[1] = b;
        columns[2] = c;
    }

    /** Construct 3x3 rotation matrix from given quaternion */
    mat3(quat q);

    inline void Empty()
    {
        *this = { 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
    }

    inline mat3 GetTranspose() const;

    inline mat3 GetInverse() const;

    /** Returns a float pointer to the memory layout of the matrix. Useful
        for uploading data to graphics API. OpenGL uses column-major order.*/
    float* ptr() { return((float*)this); }

    vec3& operator[] (int col) { return columns[col]; }
    const vec3& operator[] (int col) const { return columns[col]; }
};

/** - Quaternions -
    You shouldn't modify quaternions values directly--use functions
*/
struct quat
{
    float w = 0.f;
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    /* Constructs an identity quaternion */
    quat()
        : w(1.f)
        , x(0.f)
        , y(0.f)
        , z(0.f)
    {}

    quat(float W, float X, float Y, float Z);
    quat(float angleInRadians, vec3 axisOfRotation);

    /** Gets the conjugate of given quaternion. Doesn't check that Magnitude is 1. */
    inline quat GetConjugate() const;

    /** Identical to conjugate */
    inline quat GetInverse() const;

    /** Gets the conjugate of the given quaternion with Magnitude 1 */
    inline quat GetInverseUnitQuaternion();
};

/**
 *
 * Basic Vector Operations
 *
 */
inline vec2 Add(vec2 a, vec2 b);
inline vec3 Add(vec3 a, vec3 b);
inline vec4 Add(vec4 a, vec4 b);
inline vec2 Sub(vec2 a, vec2 b);
inline vec3 Sub(vec3 a, vec3 b);
inline vec4 Sub(vec4 a, vec4 b);
inline vec2 Mul(vec2 a, float b);
inline vec3 Mul(vec3 a, float b);
inline vec4 Mul(vec4 a, float b);
inline vec2 Div(vec2 a, float b);
inline vec3 Div(vec3 a, float b);
inline vec4 Div(vec4 a, float b);
inline float Dot(vec2 a, vec2 b);
inline float Dot(vec3 a, vec3 b);
inline float Dot(vec4 a, vec4 b);
inline vec3 Cross(vec3 a, vec3 b);
inline float Length(vec2 a);
inline float Length(vec3 a);
inline float Length(vec4 a);
inline float Magnitude(vec2 a);
inline float Magnitude(vec3 a);
inline float Magnitude(vec4 a);
inline vec2 Normalize(vec2 a);
inline vec3 Normalize(vec3 a);
inline vec4 Normalize(vec4 a);
inline vec3 Lerp(vec3 from, vec3 to, float ratio);
inline vec4 Lerp(vec4 from, vec4 to, float ratio);

inline ivec2 operator-(ivec2 a);
inline ivec2 operator+(ivec2 a, ivec2 b);
inline ivec2 operator-(ivec2 a, ivec2 b);
inline ivec2 operator*(ivec2 a, float b);
inline ivec2 operator*(float b, ivec2 a);
inline ivec2 operator/(ivec2 a, float b);
inline ivec2 &operator+=(ivec2 &a, ivec2 b);
inline ivec2 &operator-=(ivec2 &a, ivec2 b);
inline ivec2 &operator*=(ivec2 &a, float b);
inline ivec2 &operator/=(ivec2 &a, float b);
inline bool operator==(const ivec2 &lhs, const ivec2 &rhs);
inline bool operator!=(const ivec2 &lhs, const ivec2 &rhs);

inline ivec3 operator-(ivec3 a);
inline ivec3 operator+(ivec3 a, ivec3 b);
inline ivec3 operator-(ivec3 a, ivec3 b);
inline ivec3 operator*(ivec3 a, int b);
inline ivec3 operator*(int b, ivec3 a);
inline ivec3 operator/(ivec3 a, int b);
inline ivec3 &operator+=(ivec3& a, ivec3 b);
inline ivec3 &operator-=(ivec3& a, ivec3 b);
inline ivec3 &operator*=(ivec3& a, int b);
inline ivec3 &operator/=(ivec3& a, int b);
inline bool operator==(const ivec3& lhs, const ivec3& rhs);
inline bool operator!=(const ivec3& lhs, const ivec3& rhs);

inline vec2 operator-(vec2 a);
inline vec2 operator+(vec2 a, vec2 b);
inline vec2 operator-(vec2 a, vec2 b);
inline vec2 operator*(vec2 a, float b);
inline vec2 operator*(float b, vec2 a);
inline vec2 operator/(vec2 a, float b);
inline vec2 &operator+=(vec2& a, vec2 b);
inline vec2 &operator-=(vec2& a, vec2 b);
inline vec2 &operator*=(vec2& a, float b);
inline vec2 &operator/=(vec2& a, float b);
inline bool operator==(const vec2& lhs, const vec2& rhs);
inline bool operator!=(const vec2& lhs, const vec2& rhs);

inline vec3 operator-(vec3 a);
inline vec3 operator+(vec3 a, vec3 b);
inline vec3 operator-(vec3 a, vec3 b);
inline vec3 operator*(vec3 a, float b);
inline vec3 operator*(float b, vec3 a);
inline vec3 operator/(vec3 a, float b);
inline vec3 &operator+=(vec3& a, vec3 b);
inline vec3 &operator-=(vec3& a, vec3 b);
inline vec3 &operator*=(vec3& a, float b);
inline vec3 &operator/=(vec3& a, float b);
inline bool operator==(const vec3& lhs, const vec3& rhs);
inline bool operator!=(const vec3& lhs, const vec3& rhs);

inline vec4 operator-(vec4 a);
inline vec4 operator+(vec4 a, vec4 b);
inline vec4 operator-(vec4 a, vec4 b);
inline vec4 operator*(vec4 a, float b);
inline vec4 operator*(float b, vec4 a);
inline vec4 operator/(vec4 a, float b);
inline vec4 &operator+=(vec4& a, vec4 b);
inline vec4 &operator-=(vec4& a, vec4 b);
inline vec4 &operator*=(vec4& a, float b);
inline vec4 &operator/=(vec4& a, float b);
inline bool operator==(const vec4& lhs, const vec4& rhs);
inline bool operator!=(const vec4& lhs, const vec4& rhs);

/**
 *
 * Basic Matrix Operations
 *
 */
inline mat3 Mul(const mat3& a, const mat3& b);
inline vec3 Mul(const mat3& A, vec3 v);
inline mat4 Mul(const mat4& a, const mat4& b);
inline vec4 Mul(const mat4& A, vec4 v);
inline vec3 operator*(mat3 A, vec3 v);
inline mat3 operator*(mat3 a, mat3 b);
inline mat3& operator*=(mat3& a, mat3 b);
inline vec4 operator*(mat4 A, vec4 v);
inline mat4 operator*(mat4 a, mat4 b);
inline mat4& operator*=(mat4& a, mat4 b);

/**
 *
 *  Basic Quaternion Operations
 *
 */
inline quat Add(quat a, quat b);
inline quat Sub(quat a, quat b);
inline float Dot(quat a, quat b);
inline quat Mul(quat a, quat b); // Combines rotation a and b such that a is applied first then b
inline quat Mul(quat a, float scale);
inline quat Div(quat a, float scale);
inline float Magnitude(quat a);
inline quat Normalize(quat a);

/**
 *
 * Advanced Operations
 *
 */
/** Generates translation matrix for given delta x delta y delta z
    https://en.wikipedia.org/wiki/Translation_(geometry)#Matrix_resentation */
inline mat4 TranslationMatrix(float x, float y, float z);
inline mat4 TranslationMatrix(vec3 translation);
inline mat3 TranslationMatrix2D(vec2 translation);

/** Generates rotation matrix for given quaternion represented rotation
    https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions */
inline mat4 RotationMatrix(quat q);

inline mat3 RotationMatrix2D(float rotationInRadians);

/** Generates scaling matrix for given x y z scales
    https://en.wikipedia.org/wiki/Scaling_(geometry)#Using_homogeneous_coordinates */
inline mat4 ScaleMatrix(float scale);
inline mat4 ScaleMatrix(float x_scale, float y_scale, float z_scale);
inline mat4 ScaleMatrix(vec3 scale);
inline mat3 ScaleMatrix2D(vec2 scale);

/** Creates a 4x4 matrix for a symetric perspective-view frustum based on the default handedness and default near and far clip planes definition.
    fovy: Specifies the field of view angle in the y direction. Expressed in radians.
    aspect: Specifies the aspect ratio that determines the field of view in the x direction. The aspect ratio is the ratio of x (width) to y (height).
    nearclip: Specifies the distance from the viewer to the near clipping plane (always positive).
    farclip: Specifies the distance from the viewer to the far clipping plane (always positive).
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml */
inline mat4 ProjectionMatrixPerspective(float fovy, float aspect, float nearclip, float farclip);

/** Creates a 4x4 matrix for an orthographic parallel viewing volume, using right-handed coordinates.
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml
    The near and far clip planes correspond to z normalized device coordinates of -1 and +1
    respectively (OpenGL clip volume definition). */
inline mat4 ProjectionMatrixOrthographic(float left, float right, float bottom, float top, float z_near, float z_far);

/** Creates a 4x4 matrix for projecting two-dimensional coordinates onto the screen.
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluOrtho2D.xml
    left, right: Specify the coordinates for the left and right vertical clipping planes.
    bottom, top: Specify the coordinates for the bottom and top horizontal clipping planes.
    e.g. ProjectionMatrixOrthographicNoZ(0.f, 1920.f, 1080.f, 0.f);
    e.g. ProjectionMatrixOrthographicNoZ(0.f, 1920.f, 0.f, 1080.f); <-- vertical flip */
inline mat4 ProjectionMatrixOrthographicNoZ(float left, float right, float bottom, float top);

/** Creates a 3x3 matrix for projecting two-dimensional coordinates onto the screen.
    This is used for transforming vec3s where vec3.x is the X coordinate, vec3.y is
    the Y coordinate, and vec3.z is 1 for points and 0 for vectors. This is different
    from ProjectionMatrixOrthographicNoZ which gives a 4x4 matrix used for transforming
    vec4s which includes a Z coordinate which is discarded. */
inline mat3 ProjectionMatrixOrthographic2D(float left, float right, float bottom, float top);

/** Creates a 4x4 ViewMatrix at eye looking at target. */
inline mat4 ViewMatrixLookAt(vec3 const& eye, vec3 const& target, vec3 const& eyeUpVector);

/** Checks if Dot product of a and b is within +/- tolerance */
inline bool Similar(quat a, quat b, float tolerance = 0.0001f);

/** Combines rotations represented by quaternions. Equivalent to second * first. */
inline quat CombineRotations(quat firstRotation, quat secondRotation);

/** Convert Quaternion to Euler angles IN RADIANS. When you read the .eulerAngles property,
    Unity converts the Quaternion's internal representation of the rotation to Euler angles.
    Because, there is more than one way to represent any given rotation using Euler angles,
    the values you read back out may be quite different from the values you assigned. This
    can cause confusion if you are trying to gradually increment the values to produce animation.
    To avoid these kinds of problems, the recommended way to work with rotations is to avoid
    relying on consistent results when reading .eulerAngles particularly when attempting to
    gradually increment a rotation to produce animation.
    This will not work when the Z euler angle is within [90, 270] degrees. This is a
    limitation with euler angles: euler angles (of any type) have a singularity. Unity's
    Quaternion.eulerAngle also experiences the same limitation, so I don't think there is
    anything I can do about it. Just whenever possible, avoid using euler angles.
*/
inline vec3 QuatToEuler(quat q);

/** Convert Euler angles IN RADIANS to a rotation Quaternion representing a rotation
    x/roll degrees around the x-axis, z/pitch degrees around the z-axis, and y/yaw degrees
    around the y-axis; APPLIED IN THAT ORDER.
    See https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
    The following wikipedia page uses a different order of rotation, but still helpful:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_to_quaternion_conversion
*/
inline quat EulerToQuat(float roll, float yaw, float pitch);
inline quat EulerToQuat(vec3 eulerAnglesInRadians);

/** Returns an orientation that faces the given direction. The return value represents
    the rotation from the world's forward direction (forward vector) in order to face
    the same direction as the given direction. The resulting orientation will point in
    the desired direction but the degree of rotation around the desired direction may vary. */
inline quat DirectionToOrientation(vec3 direction);

/** Returns the direction of this orientation. The world's forward direction (forward vector)
    rotated by the orientation is the direction. */
inline vec3 OrientationToDirection(quat orientation);

/** Creates a rotation which rotates from fromDirection to toDirection
    Similar to https://docs.unity3d.com/ScriptReference/Quaternion.FromToRotation.html
    Usually you use this to rotate a transform so that one of its axes eg. the y-axis -
    follows a target direction toDirection in world space.
*/
inline quat RotationFromTo(vec3 fromDirection, vec3 toDirection);

/** Finds the difference such that b = difference * a */
inline quat RotationDifference(quat a, quat b);

/** Rotates given vector by given quaternion represented rotation.
    The center of rotation is the origin. If you want to rotate around another point,
    translate the vector before calling RotateVector, then inverse translate (translate back). */
inline vec3 RotateVector(vec3 vector, quat rotation);

/** Converts quaternion to a 3x3 matrix representing the rotation */
inline mat3 QuatToMat3(quat q);

/** Converts quaternion to a 4x4 matrix representing the rotation */
inline mat4 QuatToMat4(quat q);

/** Spherically interpolates between quaternions from and to by ratio. The parameter ratio is clamped to the range [0, 1].
    Use this to create a rotation which smoothly interpolates between the first quaternion a to the second quaternion b,
    based on the value of the interpolation ratrio.
    from : Start value, returned when t = 0.
    to : End value, returned when t = 1.
    ratio : Interpolation ratio.
    https://www.youtube.com/watch?v=x1aCcyD0hqE&ab_channel=JorgeRodriguez
*/
inline quat Slerp(const quat from, const quat to, const float ratio);

/** Linearly interpolate between two floats */
inline float Lerp(float from, float to, float ratio);


// ==================================================================================================================
//                                               IMPLEMENTATION
// ==================================================================================================================

inline vec3::vec3(vec2 a, float b)
    : x(a.x)
    , y(a.y)
    , z(b)
{}

inline mat4::mat4(float e00, float e01, float e02, float e03,
           float e10, float e11, float e12, float e13,
           float e20, float e21, float e22, float e23,
           float e30, float e31, float e32, float e33)
{
    e[0] = e00;
    e[1] = e01;
    e[2] = e02;
    e[3] = e03;
    e[4] = e10;
    e[5] = e11;
    e[6] = e12;
    e[7] = e13;
    e[8] = e20;
    e[9] = e21;
    e[10] = e22;
    e[11] = e23;
    e[12] = e30;
    e[13] = e31;
    e[14] = e32;
    e[15] = e33;
}

inline mat4::mat4(const mat3& from)
{
    columns[0][0] = from[0][0];
    columns[0][1] = from[0][1];
    columns[0][2] = from[0][2];
    columns[0][3] = 0.f;
    columns[1][0] = from[1][0];
    columns[1][1] = from[1][1];
    columns[1][2] = from[1][2];
    columns[1][3] = 0.f;
    columns[2][0] = from[2][0];
    columns[2][1] = from[2][1];
    columns[2][2] = from[2][2];
    columns[2][3] = 0.f;
    columns[3][0] = 0.f;
    columns[3][1] = 0.f;
    columns[3][2] = 0.f;
    columns[3][3] = 1.f;
}

inline mat3::mat3(float e00, float e01, float e02,
           float e10, float e11, float e12,
           float e20, float e21, float e22)
{
    e[0] = e00;
    e[1] = e01;
    e[2] = e02;
    e[3] = e10;
    e[4] = e11;
    e[5] = e12;
    e[6] = e20;
    e[7] = e21;
    e[8] = e22;
}


inline mat3::mat3(const mat4& from)
{
    columns[0][0] = from[0][0];
    columns[0][1] = from[0][1];
    columns[0][2] = from[0][2];
    columns[1][0] = from[1][0];
    columns[1][1] = from[1][1];
    columns[1][2] = from[1][2];
    columns[2][0] = from[2][0];
    columns[2][1] = from[2][1];
    columns[2][2] = from[2][2];
}

inline quat::quat(float W, float X, float Y, float Z)
        : w(W)
        , x(X)
        , y(Y)
        , z(Z)
{
    *this = Normalize(*this);
}

inline quat::quat(float angleInRadians, vec3 axisOfRotation)
{
    axisOfRotation = Normalize(axisOfRotation);
    float half_angle = angleInRadians * 0.5f;
    float s = sinf(half_angle);
    w = cosf(half_angle);
    x = axisOfRotation.x * s;
    y = axisOfRotation.y * s;
    z = axisOfRotation.z * s;
}

inline mat4::mat4(quat q)
{
    *this = QuatToMat4(q);
}

inline mat3::mat3(quat q)
{
    *this = QuatToMat3(q);
}



inline vec2 Add(vec2 a, vec2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

inline vec3 Add(vec3 a, vec3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline vec4 Add(vec4 a, vec4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

inline vec2 Sub(vec2 a, vec2 b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

inline vec3 Sub(vec3 a, vec3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline vec4 Sub(vec4 a, vec4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

inline vec2 Mul(vec2 a, float b)
{
    a.x *= b;
    a.y *= b;
    return a;
}

inline vec3 Mul(vec3 a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

inline vec4 Mul(vec4 a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

inline vec2 Div(vec2 a, float b)
{
    b = 1.f/b;
    a.x *= b;
    a.y *= b;
    return a;
}

inline vec3 Div(vec3 a, float b)
{
    b = 1.f/b;
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

inline vec4 Div(vec4 a, float b)
{
    b = 1.f/b;
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

inline float Dot(vec2 a, vec2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline float Dot(vec3 a, vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float Dot(vec4 a, vec4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline vec3 Cross(vec3 a, vec3 b)
{
    vec3 R;
    R.x = a.y * b.z - b.y * a.z;
    R.y = a.z * b.x - b.z * a.x;
    R.z = a.x * b.y - b.x * a.y;
    return R;
}

inline float Length(vec2 a) { return Magnitude(a); }

inline float Length(vec3 a) { return Magnitude(a); }

inline float Length(vec4 a) { return Magnitude(a); }

inline float Magnitude(vec2 a) { return sqrtf(Dot(a, a)); }

inline float Magnitude(vec3 a) { return sqrtf(Dot(a, a)); }

inline float Magnitude(vec4 a) { return sqrtf(Dot(a, a)); }

inline vec2 Normalize(vec2 a)
{
    if(a.x == 0.f && a.y == 0.f)
    {
        return {0.f,0.f};
    }
    return Div(a, Magnitude(a));
}

inline vec3 Normalize(vec3 a)
{
    if(a.x == 0.f && a.y == 0.f && a.z == 0.f)
    {
        return {0.f,0.f,0.f};
    }
    return Div(a, Magnitude(a));
}

inline vec4 Normalize(vec4 a)
{
    if(a.x == 0.f && a.y == 0.f && a.z == 0.f && a.w == 0.f)
    {
        return {0.f,0.f,0.f,0.f};
    }
    return Div(a, Magnitude(a));
}

inline ivec2 operator-(ivec2 a) { ivec2 r = { -a.x, -a.y }; return(r); }
inline ivec2 operator+(ivec2 a, ivec2 b) { return ivec2(a.x + b.x, a.y + b.y); }
inline ivec2 operator-(ivec2 a, ivec2 b) { return ivec2(a.x - b.x, a.y - b.y); }
inline ivec2 operator*(ivec2 a, int b) { return ivec2(a.x * b, a.y * b); }
inline ivec2 operator*(int b, ivec2 a) { return ivec2(a.x * b, a.y * b); }
inline ivec2 operator/(ivec2 a, int b) { return ivec2(a.x / b, a.y / b); }
inline ivec2 &operator*=(ivec2 &a, int b) { return(a = a * b); }
inline ivec2 &operator/=(ivec2 &a, int b) { return(a = a / b); }
inline bool operator==(const ivec2 &lhs, const ivec2 &rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
inline bool operator!=(const ivec2 &lhs, const ivec2 &rhs) { return !(lhs == rhs); }

inline ivec3 operator-(ivec3 a) { ivec3 r = { -a.x, -a.y, -a.z }; return(r); }
inline ivec3 operator+(ivec3 a, ivec3 b) { return ivec3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline ivec3 operator-(ivec3 a, ivec3 b) { return ivec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline ivec3 operator*(ivec3 a, int b) { return ivec3(a.x * b, a.y * b, a.z * b); }
inline ivec3 operator*(int b, ivec3 a) { return ivec3(a.x * b, a.y * b, a.z * b); }
inline ivec3 operator/(ivec3 a, int b) { return ivec3(a.x / b, a.y / b, a.z / b); }
inline ivec3 &operator+=(ivec3 &a, ivec3 b) { return(a = a + b); }
inline ivec3 &operator-=(ivec3 &a, ivec3 b) { return(a = a - b); }
inline ivec3 &operator*=(ivec3 &a, int b) { return(a = a * b); }
inline ivec3 &operator/=(ivec3 &a, int b) { return(a = a / b); }
inline bool operator==(const ivec3 &lhs, const ivec3 &rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; }
inline bool operator!=(const ivec3 &lhs, const ivec3 &rhs) { return !(lhs == rhs); }

inline vec2 operator-(vec2 a) { vec2 r = { -a.x, -a.y }; return(r); }
inline vec2 operator+(vec2 a, vec2 b) { return Add(a, b); }
inline vec2 operator-(vec2 a, vec2 b) { return Sub(a, b); }
inline vec2 operator*(vec2 a, float b) { return Mul(a, b); }
inline vec2 operator*(float b, vec2 a) { return Mul(a, b); }
inline vec2 operator/(vec2 a, float b) { return Div(a, b); }
inline vec2 &operator+=(vec2& a, vec2 b) { return(a = a + b); }
inline vec2 &operator-=(vec2& a, vec2 b) { return(a = a - b); }
inline vec2 &operator*=(vec2& a, float b) { return(a = a * b); }
inline vec2 &operator/=(vec2& a, float b) { return(a = a / b); }
inline bool operator==(const vec2& lhs, const vec2& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
inline bool operator!=(const vec2& lhs, const vec2& rhs) { return !(lhs == rhs); }

inline vec3 operator-(vec3 a) { vec3 r = { -a.x, -a.y, -a.z }; return(r); }
inline vec3 operator+(vec3 a, vec3 b) { return Add(a, b); }
inline vec3 operator-(vec3 a, vec3 b) { return Sub(a, b); }
inline vec3 operator*(vec3 a, float b) { return Mul(a, b); }
inline vec3 operator*(float b, vec3 a) { return Mul(a, b); }
inline vec3 operator/(vec3 a, float b) { return Div(a, b); }
inline vec3 &operator+=(vec3& a, vec3 b) { return(a = a + b); }
inline vec3 &operator-=(vec3& a, vec3 b) { return(a = a - b); }
inline vec3 &operator*=(vec3& a, float b) { return(a = a * b); }
inline vec3 &operator/=(vec3& a, float b) { return(a = a / b); }
inline bool operator==(const vec3& lhs, const vec3& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; }
inline bool operator!=(const vec3& lhs, const vec3& rhs) { return !(lhs == rhs); }

inline vec4 operator-(vec4 a) { vec4 r = { -a.x, -a.y, -a.z, -a.w }; return(r); }
inline vec4 operator+(vec4 a, vec4 b) { return Add(a, b); }
inline vec4 operator-(vec4 a, vec4 b) { return Sub(a, b); }
inline vec4 operator*(vec4 a, float b) { return Mul(a, b); }
inline vec4 operator*(float b, vec4 a) { return Mul(a, b); }
inline vec4 operator/(vec4 a, float b) { return Div(a, b); }
inline vec4 &operator+=(vec4& a, vec4 b) { return(a = a + b); }
inline vec4 &operator-=(vec4& a, vec4 b) { return(a = a - b); }
inline vec4 &operator*=(vec4& a, float b) { return(a = a * b); }
inline vec4 &operator/=(vec4& a, float b) { return(a = a / b); }
inline bool operator==(const vec4& lhs, const vec4& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w; }
inline bool operator!=(const vec4& lhs, const vec4& rhs) { return !(lhs == rhs); }

inline vec3 Lerp(vec3 from, vec3 to, float ratio) { return((1.0f - ratio) * from + to * ratio); }
inline vec4 Lerp(vec4 from, vec4 to, float ratio) { return((1.0f - ratio) * from + to * ratio); }


inline mat3 Add(const mat3& a, const mat3& b)
{
    mat3 ret;
    ret[0] = a[0] + b[0];
    ret[1] = a[1] + b[1];
    ret[2] = a[2] + b[2];
    return ret;
}

inline mat3 Mul(const mat3& a, const mat3& b)
{
    mat3 res;

    res.columns[0][0] = Dot(vec3(a[0][0], a[1][0], a[2][0]), b[0]);
    res.columns[0][1] = Dot(vec3(a[0][1], a[1][1], a[2][1]), b[0]);
    res.columns[0][2] = Dot(vec3(a[0][2], a[1][2], a[2][2]), b[0]);

    res.columns[1][0] = Dot(vec3(a[0][0], a[1][0], a[2][0]), b[1]);
    res.columns[1][1] = Dot(vec3(a[0][1], a[1][1], a[2][1]), b[1]);
    res.columns[1][2] = Dot(vec3(a[0][2], a[1][2], a[2][2]), b[1]);

    res.columns[2][0] = Dot(vec3(a[0][0], a[1][0], a[2][0]), b[2]);
    res.columns[2][1] = Dot(vec3(a[0][1], a[1][1], a[2][1]), b[2]);
    res.columns[2][2] = Dot(vec3(a[0][2], a[1][2], a[2][2]), b[2]);

    return res;
}

inline vec3 Mul(const mat3& A, vec3 v)
{
    return A[0] * v.x + A[1] * v.y + A[2] * v.z;
}

inline mat3 Mul(const mat3& A, float v) // NEED TEST and maybe operator
{
    mat3 ret;
    ret[0] = A[0] * v;
    ret[1] = A[1] * v;
    ret[2] = A[2] * v;
    return ret;
}

inline mat4 Add(const mat4& a, const mat4& b)
{
    mat4 ret;
    ret[0] = a[0] + b[0];
    ret[1] = a[1] + b[1];
    ret[2] = a[2] + b[2];
    ret[3] = a[3] + b[3];
    return ret;
}

inline mat4 Mul(const mat4& a, const mat4& b)
{
    mat4 res;

    res.columns[0][0] = Dot(vec4(a[0][0], a[1][0], a[2][0], a[3][0]), b[0]);
    res.columns[0][1] = Dot(vec4(a[0][1], a[1][1], a[2][1], a[3][1]), b[0]);
    res.columns[0][2] = Dot(vec4(a[0][2], a[1][2], a[2][2], a[3][2]), b[0]);
    res.columns[0][3] = Dot(vec4(a[0][3], a[1][3], a[2][3], a[3][3]), b[0]);

    res.columns[1][0] = Dot(vec4(a[0][0], a[1][0], a[2][0], a[3][0]), b[1]);
    res.columns[1][1] = Dot(vec4(a[0][1], a[1][1], a[2][1], a[3][1]), b[1]);
    res.columns[1][2] = Dot(vec4(a[0][2], a[1][2], a[2][2], a[3][2]), b[1]);
    res.columns[1][3] = Dot(vec4(a[0][3], a[1][3], a[2][3], a[3][3]), b[1]);

    res.columns[2][0] = Dot(vec4(a[0][0], a[1][0], a[2][0], a[3][0]), b[2]);
    res.columns[2][1] = Dot(vec4(a[0][1], a[1][1], a[2][1], a[3][1]), b[2]);
    res.columns[2][2] = Dot(vec4(a[0][2], a[1][2], a[2][2], a[3][2]), b[2]);
    res.columns[2][3] = Dot(vec4(a[0][3], a[1][3], a[2][3], a[3][3]), b[2]);

    res.columns[3][0] = Dot(vec4(a[0][0], a[1][0], a[2][0], a[3][0]), b[3]);
    res.columns[3][1] = Dot(vec4(a[0][1], a[1][1], a[2][1], a[3][1]), b[3]);
    res.columns[3][2] = Dot(vec4(a[0][2], a[1][2], a[2][2], a[3][2]), b[3]);
    res.columns[3][3] = Dot(vec4(a[0][3], a[1][3], a[2][3], a[3][3]), b[3]);

    return res;
}

inline vec4 Mul(const mat4& A, vec4 v)
{
    return A[0] * v.x + A[1] * v.y + A[2] * v.z + A[3] * v.w;
}

inline mat4 Mul(const mat4& A, float v) // NEED TEST and maybe operator
{
    mat4 ret;
    ret[0] = A[0] * v;
    ret[1] = A[1] * v;
    ret[1] = A[1] * v;
    ret[3] = A[3] * v;
    return ret;
}

inline vec3 operator*(mat3 A, vec3 v) { return(Mul(A, v)); }
inline mat3 operator*(mat3 a, mat3 b) { return(Mul(a, b)); }
inline mat3& operator*=(mat3& a, mat3 b) { a = Mul(a, b); return a; }
inline vec4 operator*(mat4 A, vec4 v) { return(Mul(A, v)); }
inline mat4 operator*(mat4 a, mat4 b) { return(Mul(a, b)); }
inline mat4& operator*=(mat4& a, mat4 b) { a = Mul(a, b); return a; }

inline mat4 TranslationMatrix(float x, float y, float z)
{
    mat4 ret = mat4();
    ret[3][0] = x;
    ret[3][1] = y;
    ret[3][2] = z;
    ret[3][3] = 1.f;
    return ret;
}

inline mat4 TranslationMatrix(vec3 translation)
{
    return TranslationMatrix(translation.x, translation.y, translation.z);
}

inline mat3 TranslationMatrix2D(vec2 translation) // TODO(Kevin): NEED TESTS
{
    mat3 ret = mat3();
    ret[2][0] = translation.x;
    ret[2][1] = translation.y;
    ret[2][2] = 1.f;
    return ret;
}

inline mat4 RotationMatrix(quat q)
{
    return mat4(q);
}

inline mat3 RotationMatrix2D(float rotationInRadians)
{
    mat3 ret;
    float cr = cosf(rotationInRadians);
    float sr = sinf(rotationInRadians);
    ret[0][0] = cr;
    ret[0][1] = sr;
    ret[1][0] = -sr;
    ret[1][1] = cr;
    ret[2][2] = 1.f;
    return ret;
}

inline mat4 ScaleMatrix(float scale)
{
    return ScaleMatrix(scale,scale,scale);
}

inline mat4 ScaleMatrix(float x_scale, float y_scale, float z_scale)
{
    mat4 ret;
    ret[0][0] = x_scale;
    ret[1][1] = y_scale;
    ret[2][2] = z_scale;
    ret[3][3] = 1.f;
    return ret;
}

inline mat4 ScaleMatrix(vec3 scale)
{
    return ScaleMatrix(scale.x, scale.y, scale.z);
}

inline mat3 ScaleMatrix2D(vec2 scale) // TODO(Kevin): NEED TESTS
{
    mat3 ret = mat3();
    ret[0][0] = scale.x;
    ret[1][1] = scale.y;
    ret[2][2] = 1.f;
    return ret;
}

inline mat4 ProjectionMatrixPerspective(float fovy, float aspect, float nearclip, float farclip)
{
    float const tanHalfFovy = tanf(fovy / 2.f);

    mat4 Result;
    Result[0][0] = 1.f / (aspect * tanHalfFovy);
    Result[1][1] = 1.f / (tanHalfFovy);
    Result[2][2] = -(farclip + nearclip) / (farclip - nearclip);
    Result[2][3] = -1.f;
    Result[3][2] = -(2.f * farclip * nearclip) / (farclip - nearclip);
    Result[3][3] = 0.f;
    return Result;
}

inline mat4 ProjectionMatrixOrthographic(float left, float right, float bottom, float top, float z_near, float z_far)
{
    mat4 ret = mat4();
    ret[0][0] = 2.f / (right - left);
    ret[1][1] = 2.f / (top - bottom);
    ret[2][2] = - 2.f / (z_far - z_near);
    ret[3][0] = - (right + left) / (right - left);
    ret[3][1] = - (top + bottom) / (top - bottom);
    ret[3][2] = - (z_far + z_near) / (z_far - z_near);
    return ret;
}

inline mat4 ProjectionMatrixOrthographicNoZ(float left, float right, float bottom, float top)
{
    mat4 ret = mat4();
    ret[0][0] = 2.f / (right - left);
    ret[1][1] = 2.f / (top - bottom);
    ret[2][2] = -1.f;
    ret[3][0] = -(right + left) / (right - left);
    ret[3][1] = -(top + bottom) / (top - bottom);
    return ret;
}

inline mat3 ProjectionMatrixOrthographic2D(float left, float right, float bottom, float top)
{
    mat3 ret = mat3();
    ret[0][0] = 2.f / (right - left);
    ret[1][1] = 2.f / (top - bottom);
    ret[2][0] = -(right + left) / (right - left);
    ret[2][1] = -(top + bottom) / (top - bottom);
    ret[2][2] = 1.f;
    return ret;
}

inline mat4 ViewMatrixLookAt(vec3 const& eye, vec3 const& target, vec3 const& eyeUpVector)
{
    vec3 const direction = Normalize(target - eye);
    vec3 const right = Normalize(Cross(direction, eyeUpVector));
    vec3 const up = Cross(right, direction);

    mat4 ret = mat4();
    ret.Empty();

    ret[0][0] = right.x;
    ret[1][0] = right.y;
    ret[2][0] = right.z;

    ret[0][1] = up.x;
    ret[1][1] = up.y;
    ret[2][1] = up.z;

    ret[0][2] = -direction.x;
    ret[1][2] = -direction.y;
    ret[2][2] = -direction.z;

    ret[3][0] = -Dot(right, eye);
    ret[3][1] = -Dot(up, eye);
    ret[3][2] = Dot(direction, eye);
    ret[3][3] = 1.f;

    return ret;
}

inline mat4 mat4::GetTranspose() const
{
    mat4 ret;
    for(int col = 0; col < 4; ++col)
    {
        for(int row = 0; row < 4; ++row)
        {
            ret[col][row] = columns[row][col];
        }
    }
    return ret;
}

inline mat3 mat3::GetTranspose() const
{
    mat3 ret;
    for(int col = 0; col < 3; ++col)
    {
        for(int row = 0; row < 3; ++row)
        {
            ret[col][row] = columns[row][col];
        }
    }
    return ret;
}

inline mat4 mat4::GetInverse() const
{
    float inv[16], det;
    int i;

    inv[0] = e[5]  * e[10] * e[15] - 
             e[5]  * e[11] * e[14] - 
             e[9]  * e[6]  * e[15] + 
             e[9]  * e[7]  * e[14] +
             e[13] * e[6]  * e[11] - 
             e[13] * e[7]  * e[10];

    inv[4] = -e[4]  * e[10] * e[15] + 
              e[4]  * e[11] * e[14] + 
              e[8]  * e[6]  * e[15] - 
              e[8]  * e[7]  * e[14] - 
              e[12] * e[6]  * e[11] + 
              e[12] * e[7]  * e[10];

    inv[8] = e[4]  * e[9] * e[15] - 
             e[4]  * e[11] * e[13] - 
             e[8]  * e[5] * e[15] + 
             e[8]  * e[7] * e[13] + 
             e[12] * e[5] * e[11] - 
             e[12] * e[7] * e[9];

    inv[12] = -e[4]  * e[9] * e[14] + 
               e[4]  * e[10] * e[13] +
               e[8]  * e[5] * e[14] - 
               e[8]  * e[6] * e[13] - 
               e[12] * e[5] * e[10] + 
               e[12] * e[6] * e[9];

    inv[1] = -e[1]  * e[10] * e[15] + 
              e[1]  * e[11] * e[14] + 
              e[9]  * e[2] * e[15] - 
              e[9]  * e[3] * e[14] - 
              e[13] * e[2] * e[11] + 
              e[13] * e[3] * e[10];

    inv[5] = e[0]  * e[10] * e[15] - 
             e[0]  * e[11] * e[14] - 
             e[8]  * e[2] * e[15] + 
             e[8]  * e[3] * e[14] + 
             e[12] * e[2] * e[11] - 
             e[12] * e[3] * e[10];

    inv[9] = -e[0]  * e[9] * e[15] + 
              e[0]  * e[11] * e[13] + 
              e[8]  * e[1] * e[15] - 
              e[8]  * e[3] * e[13] - 
              e[12] * e[1] * e[11] + 
              e[12] * e[3] * e[9];

    inv[13] = e[0]  * e[9] * e[14] - 
              e[0]  * e[10] * e[13] - 
              e[8]  * e[1] * e[14] + 
              e[8]  * e[2] * e[13] + 
              e[12] * e[1] * e[10] - 
              e[12] * e[2] * e[9];

    inv[2] = e[1]  * e[6] * e[15] - 
             e[1]  * e[7] * e[14] - 
             e[5]  * e[2] * e[15] + 
             e[5]  * e[3] * e[14] + 
             e[13] * e[2] * e[7] - 
             e[13] * e[3] * e[6];

    inv[6] = -e[0]  * e[6] * e[15] + 
              e[0]  * e[7] * e[14] + 
              e[4]  * e[2] * e[15] - 
              e[4]  * e[3] * e[14] - 
              e[12] * e[2] * e[7] + 
              e[12] * e[3] * e[6];

    inv[10] = e[0]  * e[5] * e[15] - 
              e[0]  * e[7] * e[13] - 
              e[4]  * e[1] * e[15] + 
              e[4]  * e[3] * e[13] + 
              e[12] * e[1] * e[7] - 
              e[12] * e[3] * e[5];

    inv[14] = -e[0]  * e[5] * e[14] + 
               e[0]  * e[6] * e[13] + 
               e[4]  * e[1] * e[14] - 
               e[4]  * e[2] * e[13] - 
               e[12] * e[1] * e[6] + 
               e[12] * e[2] * e[5];

    inv[3] = -e[1] * e[6] * e[11] + 
              e[1] * e[7] * e[10] + 
              e[5] * e[2] * e[11] - 
              e[5] * e[3] * e[10] - 
              e[9] * e[2] * e[7] + 
              e[9] * e[3] * e[6];

    inv[7] = e[0] * e[6] * e[11] - 
             e[0] * e[7] * e[10] - 
             e[4] * e[2] * e[11] + 
             e[4] * e[3] * e[10] + 
             e[8] * e[2] * e[7] - 
             e[8] * e[3] * e[6];

    inv[11] = -e[0] * e[5] * e[11] + 
               e[0] * e[7] * e[9] + 
               e[4] * e[1] * e[11] - 
               e[4] * e[3] * e[9] - 
               e[8] * e[1] * e[7] + 
               e[8] * e[3] * e[5];

    inv[15] = e[0] * e[5] * e[10] - 
              e[0] * e[6] * e[9] - 
              e[4] * e[1] * e[10] + 
              e[4] * e[2] * e[9] + 
              e[8] * e[1] * e[6] - 
              e[8] * e[2] * e[5];

    det = e[0] * inv[0] + e[1] * inv[4] + e[2] * inv[8] + e[3] * inv[12];

    if (det == 0.f)
    {
        // Matrix is singular and cannot be inverted
        return mat4();
    }

    det = 1.0f / det;

    mat4 ret;
    for (i = 0; i < 16; i++)
    {
        ret.e[i] = inv[i] * det;
    }

    return ret;
}

inline mat3 mat3::GetInverse() const // TODO(Kevin): test coverage
{
    float a11 = e[0], a21 = e[1], a31 = e[2];
    float a12 = e[3], a22 = e[4], a32 = e[5];
    float a13 = e[6], a23 = e[7], a33 = e[8];

    float det = a11 * (a22 * a33 - a32 * a23)
              - a12 * (a21 * a33 - a31 * a23)
              + a13 * (a21 * a32 - a31 * a22);

    if (det == 0.0f) 
    {
        // Matrix is singular and cannot be inverted
        return mat3();
    }

    float invDet = 1.0f / det;

    mat3 inv = {
        (a22 * a33 - a32 * a23) * invDet,
        -(a21 * a33 - a31 * a23) * invDet,
        (a21 * a32 - a31 * a22) * invDet,
        -(a12 * a33 - a32 * a13) * invDet,
        (a11 * a33 - a31 * a13) * invDet,
        -(a11 * a32 - a31 * a12) * invDet,
        (a12 * a23 - a22 * a13) * invDet,
        -(a11 * a23 - a21 * a13) * invDet,
        (a11 * a22 - a21 * a12) * invDet
    };

    return inv;
}

// returns 3x3 matrix product of a and row major representation of b
inline mat3 MatrixProduct(vec3 a, vec3 b)
{
    mat3 ret;
    ret[0][0] = a.x*b.x;
    ret[0][1] = a.y*b.x;
    ret[0][2] = a.z*b.x;
    ret[1][0] = a.x*b.y;
    ret[1][1] = a.y*b.y;
    ret[1][2] = a.z*b.y;
    ret[2][0] = a.x*b.z;
    ret[2][1] = a.y*b.z;
    ret[2][2] = a.z*b.z;
    return ret;
}

inline mat4 MatrixProduct(vec4 a, vec4 b)
{
    mat3 ret;
    ret[0][0] = a.x*b.x;
    ret[0][1] = a.y*b.x;
    ret[0][2] = a.z*b.x;
    ret[0][3] = a.w*b.x;
    ret[1][0] = a.x*b.y;
    ret[1][1] = a.y*b.y;
    ret[1][2] = a.z*b.y;
    ret[1][3] = a.w*b.y;
    ret[2][0] = a.x*b.z;
    ret[2][1] = a.y*b.z;
    ret[2][2] = a.z*b.z;
    ret[2][3] = a.w*b.z;
    ret[3][0] = a.x*b.w;
    ret[3][1] = a.y*b.w;
    ret[3][2] = a.z*b.w;
    ret[3][3] = a.w*b.w;
    return ret;
}

inline quat Add(quat a, quat b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

inline quat Sub(quat a, quat b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

inline float Dot(quat a, quat b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline quat Mul(quat a, quat b)
{
    quat R;
    R.w = a.w * b.w - Dot(vec3(a.x, a.y, a.z), vec3(b.x, b.y, b.z));
    vec3 va = vec3(a.x, a.y, a.z);
    vec3 vb = vec3(b.x, b.y, b.z);
    vec3 first = b.w * va;
    vec3 second = a.w * vb;
    vec3 third = Cross(va, vb);
    R.x = first.x + second.x + third.x;
    R.y = first.y + second.y + third.y;
    R.z = first.z + second.z + third.z;
    return R;
}

inline quat Mul(quat a, float scale)
{
    a.w *= scale;
    a.x *= scale;
    a.y *= scale;
    a.z *= scale;
    return a;
}

inline quat Div(quat a, float scale)
{
    float one_over_s = 1.0f / scale;
    a.w *= one_over_s;
    a.x *= one_over_s;
    a.y *= one_over_s;
    a.z *= one_over_s;
    return a;
}

inline quat operator*(quat a, quat b) { return(Mul(a, b)); }

inline float Magnitude(quat a)
{
    return(sqrtf(Dot(a, a)));
}

inline quat Normalize(quat a)
{
    return Div(a, Magnitude(a));
}

inline bool Similar(quat a, quat b, float tolerance)
{
    return(GM_abs(Dot(a, b) - 1.f) <= tolerance);
}

inline quat CombineRotations(quat firstRotation, quat secondRotation)
{
    return secondRotation * firstRotation; // order matters!
}

inline quat quat::GetConjugate() const
{
    quat ret;
    ret.w = w;
    ret.x = -x;
    ret.y = -y;
    ret.z = -z;
    return ret;
}

inline quat quat::GetInverse() const
{
    return GetConjugate();
}

inline quat quat::GetInverseUnitQuaternion()
{
    quat ret = Div(GetConjugate(), Dot(*this, *this));
    return ret;
}

inline vec3 QuatToEuler(quat q)
{
    vec3 euler_angles;

    float x2 = q.x * q.x;
    float y2 = q.y * q.y;
    float z2 = q.z * q.z;
    float xy = q.x * q.y;
    float zw = q.z * q.w;
    float xz = q.x * q.z;
    float yw = q.y * q.w;
    float yz = q.y * q.z;
    float xw = q.x * q.w;

    float r10 = 2.0f * (xy + zw);
    float r20 = 2.0f * (xz - yw);
    float r00 = 1.0f - 2.0f * (y2 + z2);
    float r12 = 2.0f * (yz - xw);
    float r11 = 1.0f - 2.0f * (x2 + z2);
    float r21 = 2.0f * (yz + xw) ;
    float r22 = 1.0f - 2.0f * (x2 + y2);

    if(r10 < 1.f)
    {
        if(r10 > -1.f)
        {
            euler_angles.z = asinf(r10);
            euler_angles.y = atan2f(-r20, r00);
            euler_angles.x = atan2f(-r12, r11);
        }
        else // r10 = -1
        {
            euler_angles.z = -GM_PI / 2.f;
            euler_angles.y = -atan2f(r21, r22);
            euler_angles.x = 0;
        }
    }
    else
    {
        euler_angles.z = GM_PI / 2.f;
        euler_angles.y = atan2f(r21, r22);
        euler_angles.x = 0;
    }

    return euler_angles;
}

inline quat EulerToQuat(float roll, float yaw, float pitch)
{
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);

    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);

    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);

    quat ret;

    ret.w = cr * cp * cy - sr * sp * sy;
    ret.x = sr * cp * cy + cr * sp * sy;
    ret.y = cr * cp * sy + sr * sp * cy;
    ret.z = cr * cy * sp - sr * sy * cp;

    return ret;
}

inline quat EulerToQuat(vec3 eulerAnglesInRadians)
{
    return EulerToQuat(eulerAnglesInRadians.x, eulerAnglesInRadians.y, eulerAnglesInRadians.z);
}

inline quat DirectionToOrientation(vec3 direction)
{
    return RotationFromTo(GM_FORWARD_VECTOR, direction);
}

inline vec3 OrientationToDirection(quat orientation)
{
    return RotateVector(GM_FORWARD_VECTOR, orientation);
}

inline quat RotationFromTo(vec3 fromDirection, vec3 toDirection)
{
    vec3 start = Normalize(fromDirection);
    vec3 dest = Normalize(toDirection);

    float cos_theta = Dot(start, dest);
    vec3 rotation_axis;
    quat rotation_quat;

    rotation_axis = Cross(start, dest);
    if (cos_theta >= -1 + 0.0001f)
    {
        float s = sqrtf((1 + cos_theta) * 2);
        float sin_of_half_angle = 1 / s;

        rotation_quat = quat(
            s * 0.5f, // recall cos(theta/2) trig identity
            rotation_axis.x * sin_of_half_angle,
            rotation_axis.y * sin_of_half_angle,
            rotation_axis.z * sin_of_half_angle
        );
    }
    else
    {
        // When vectors in opposite directions, there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to start
        rotation_axis = Cross(vec3(0.0f, 0.0f, 1.0f), start);
        if (Dot(rotation_axis, rotation_axis) < 0.01f) // bad luck, they were parallel, try again!
            rotation_axis = Cross(vec3(1.0f, 0.0f, 0.0f), start);
        rotation_quat = quat(GM_PI, rotation_axis);
    }

    return rotation_quat;
}

inline quat RotationDifference(quat a, quat b)
{
    quat ret = Mul(b, a.GetInverseUnitQuaternion());
    return ret;
}

inline vec3 RotateVector(vec3 vector, quat rotation)
{
    quat vector_quat = {0.f, vector.x, vector.y, vector.z };
    quat rotated_vector = rotation * vector_quat * rotation.GetInverse();
    return vec3(rotated_vector.x, rotated_vector.y, rotated_vector.z);
}

inline mat3 QuatToMat3(quat q)
{
    q = q.GetInverseUnitQuaternion();

    mat3 ret;

    float x2 = q.x * q.x;
    float y2 = q.y * q.y;
    float z2 = q.z * q.z;

    float xy = q.x * q.y;
    float zw = q.z * q.w;
    float xz = q.x * q.z;
    float yw = q.y * q.w;
    float yz = q.y * q.z;
    float xw = q.x * q.w;

    ret.columns[0][0] = 1.0f - 2.0f * (y2 + z2);
    ret.columns[1][0] = 2.0f * (xy + zw);
    ret.columns[2][0] = 2.0f * (xz - yw);

    ret.columns[0][1] = 2.0f * (xy - zw);
    ret.columns[1][1] = 1.0f - 2.0f * (x2 + z2);
    ret.columns[2][1] = 2.0f * (yz + xw);

    ret.columns[0][2] = 2.0f * (xz + yw);
    ret.columns[1][2] = 2.0f * (yz - xw);
    ret.columns[2][2] = 1.0f - 2.0f * (x2 + y2);

    return ret;
}

inline mat4 QuatToMat4(quat q)
{
    return mat4(QuatToMat3(q));
}

inline quat Slerp(const quat from, const quat to, const float ratio)
{
    quat start = Normalize(from);
    quat end = Normalize(to);

    // From Jolt Physics Library Quat.inl

    // Difference at which to LERP instead of SLERP
    const float delta = 0.0001f;

    // Calc cosine
    float sign_scale1 = 1.0f;
    float cos_omega = Dot(start, end);

    // Adjust signs (if necessary)
    if (cos_omega < 0.0f)
    {
        cos_omega = -cos_omega;
        sign_scale1 = -1.0f;
    }

    // Calculate coefficients
    float scale0, scale1;
    if (1.0f - cos_omega > delta)
    {
        // Standard case (slerp)
        float omega = acosf(cos_omega);
        float sin_omega = sinf(omega);
        scale0 = sinf((1.0f - ratio) * omega) / sin_omega;
        scale1 = sign_scale1 * sinf(ratio * omega) / sin_omega;
    }
    else
    {
        // Quaternions are very close so we can do a linear interpolation
        scale0 = 1.0f - ratio;
        scale1 = sign_scale1 * ratio;
    }

    // Interpolate between the two quaternions
    return Normalize(Add(Mul(start, scale0), Mul(end, scale1)));
}

inline float Lerp(float from, float to, float ratio)
{
    return from + ratio * (to - from);
}

#endif // _INCLUDE_GMATH_LIBRARY_H_

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2021 Kevin Chin
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
