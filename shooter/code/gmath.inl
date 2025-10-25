
// ==================================================================================================================
//                                               IMPLEMENTATION
// ==================================================================================================================

inline vec3::vec3(vec2 A, float B)
    : x(A.x)
    , y(A.y)
    , z(B)
{}

inline mat4::mat4()
{
    columns[0] = vec4(1.f, 0.f, 0.f, 0.f);
    columns[1] = vec4(0.f, 1.f, 0.f, 0.f);
    columns[2] = vec4(0.f, 0.f, 1.f, 0.f);
    columns[3] = vec4(0.f, 0.f, 0.f, 1.f);
}

inline mat4::mat4(float e00, float e01, float e02, float e03,
                  float e10, float e11, float e12, float e13,
                  float e20, float e21, float e22, float e23,
                  float e30, float e31, float e32, float e33)
{
    columns[0] = { e00, e01, e02, e03 };
    columns[1] = { e10, e11, e12, e13 };
    columns[2] = { e20, e21, e22, e23 };
    columns[3] = { e30, e31, e32, e33 };
}

inline mat4::mat4(vec4 A, vec4 B, vec4 C, vec4 D)
{
    columns[0] = A;
    columns[1] = B;
    columns[2] = C;
    columns[3] = D;
}

inline mat3::mat3()
{
    columns[0] = vec3(1.f, 0.f, 0.f);
    columns[1] = vec3(0.f, 1.f, 0.f);
    columns[2] = vec3(0.f, 0.f, 1.f);
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

inline mat3::mat3(vec3 A, vec3 B, vec3 C)
{
    columns[0] = A;
    columns[1] = B;
    columns[2] = C;
}

inline quat::quat(float W, float X, float Y, float Z)
        : w(W)
        , x(X)
        , y(Y)
        , z(Z)
{
    *this = Normalize(*this);
}

inline quat::quat(float Angle, vec3 Axis)
{
    Axis = Normalize(Axis);
    float HalfAngle = Angle * 0.5f;
    float S = Sin(HalfAngle);
    W = Cos(HalfAngle);
    X = Axis.x * S;
    Y = Axis.y * S;
    Z = Axis.z * S;
}

inline vec2 Add(vec2 A, vec2 B)
{
    A.x += B.x;
    A.y += B.y;
    return A;
}

inline vec3 Add(vec3 A, vec3 B)
{
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    return A;
}

inline vec4 Add(vec4 A, vec4 B)
{
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    A.w += B.w;
    return A;
}

inline vec2 Sub(vec2 A, vec2 B)
{
    A.x -= B.x;
    A.y -= B.y;
    return A;
}

inline vec3 Sub(vec3 A, vec3 B)
{
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    return A;
}

inline vec4 Sub(vec4 A, vec4 B)
{
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    A.w -= B.w;
    return A;
}

inline vec2 Mul(vec2 A, float B)
{
    A.x *= B;
    A.y *= B;
    return A;
}

inline vec3 Mul(vec3 A, float B)
{
    A.x *= B;
    A.y *= B;
    A.z *= B;
    return A;
}

inline vec4 Mul(vec4 A, float B)
{
    A.x *= B;
    A.y *= B;
    A.z *= B;
    A.w *= B;
    return A;
}

inline vec2 Div(vec2 A, float B)
{
    B = 1.f/B;
    A.x *= B;
    A.y *= B;
    return A;
}

inline vec3 Div(vec3 A, float B)
{
    B = 1.f/B;
    A.x *= B;
    A.y *= B;
    A.z *= B;
    return A;
}

inline vec4 Div(vec4 A, float B)
{
    B = 1.f/B;
    A.x *= B;
    A.y *= B;
    A.z *= B;
    A.w *= B;
    return A;
}

inline float Dot(vec2 A, vec2 B)
{
    return A.x * B.x + A.y * B.y;
}

inline float Dot(vec3 A, vec3 B)
{
    return A.x * B.x + A.y * B.y + A.z * B.z;
}

inline float Dot(vec4 A, vec4 B)
{
    return A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w;
}

inline float Cross(vec2 A, vec2 B)
{
    // Cross product of 2D vectors is the signed area of
    // the parallelogram between them.
    return A.x * B.y - A.y * B.x;
}

inline vec3 Cross(vec3 A, vec3 B)
{
    vec3 R;
    R.x = A.y * B.z - B.y * A.z;
    R.y = A.z * B.x - B.z * A.x;
    R.z = A.x * B.y - B.x * A.y;
    return R;
}

inline float Length(vec2 A) { return Magnitude(A); }

inline float Length(vec3 A) { return Magnitude(A); }

inline float Length(vec4 A) { return Magnitude(A); }

inline float Magnitude(vec2 A) { return SquareRoot(Dot(A, A)); }

inline float Magnitude(vec3 A) { return SquareRoot(Dot(A, A)); }

inline float Magnitude(vec4 A) { return SquareRoot(Dot(A, A)); }

inline vec2 Normalize(vec2 A)
{
    if(A.x == 0.f && A.y == 0.f)
    {
        return {0.f,0.f};
    }
    return Div(A, Magnitude(A));
}

inline vec3 Normalize(vec3 A)
{
    if(A.x == 0.f && A.y == 0.f && A.z == 0.f)
    {
        return {0.f,0.f,0.f};
    }
    return Div(A, Magnitude(A));
}

inline vec4 Normalize(vec4 A)
{
    if(A.x == 0.f && A.y == 0.f && A.z == 0.f && A.w == 0.f)
    {
        return {0.f,0.f,0.f,0.f};
    }
    return Div(A, Magnitude(A));
}

inline ivec2 operator-(ivec2 A) { ivec2 r = { -A.x, -A.y }; return(r); }
inline ivec2 operator+(ivec2 A, ivec2 B) { return ivec2(A.x + B.x, A.y + B.y); }
inline ivec2 operator-(ivec2 A, ivec2 B) { return ivec2(A.x - B.x, A.y - B.y); }
inline ivec2 operator*(ivec2 A, int B) { return ivec2(A.x * B, A.y * B); }
inline ivec2 operator*(int B, ivec2 A) { return ivec2(A.x * B, A.y * B); }
inline ivec2 operator/(ivec2 A, int B) { return ivec2(A.x / B, A.y / B); }
inline ivec2 &operator*=(ivec2 &A, int B) { return(A = A * B); }
inline ivec2 &operator/=(ivec2 &A, int B) { return(A = A / B); }
inline bool operator==(ivec2 &A, ivec2 &B) { return A.x == B.x && A.y == B.y; }
inline bool operator!=(ivec2 &A, ivec2 &B) { return !(A == B); }

inline ivec3 operator-(ivec3 A) { ivec3 r = { -A.x, -A.y, -A.z }; return(r); }
inline ivec3 operator+(ivec3 A, ivec3 B) { return ivec3(A.x + B.x, A.y + B.y, A.z + B.z); }
inline ivec3 operator-(ivec3 A, ivec3 B) { return ivec3(A.x - B.x, A.y - B.y, A.z - B.z); }
inline ivec3 operator*(ivec3 A, int B) { return ivec3(A.x * B, A.y * B, A.z * B); }
inline ivec3 operator*(int B, ivec3 A) { return ivec3(A.x * B, A.y * B, A.z * B); }
inline ivec3 operator/(ivec3 A, int B) { return ivec3(A.x / B, A.y / B, A.z / B); }
inline ivec3 &operator+=(ivec3 &A, ivec3 B) { return(A = A + B); }
inline ivec3 &operator-=(ivec3 &A, ivec3 B) { return(A = A - B); }
inline ivec3 &operator*=(ivec3 &A, int B) { return(A = A * B); }
inline ivec3 &operator/=(ivec3 &A, int B) { return(A = A / B); }
inline bool operator==(ivec3 &A, ivec3 &B) { return A.x == B.x && A.y == B.y && A.z == B.z; }
inline bool operator!=(ivec3 &A, ivec3 &B) { return !(A == B); }

inline vec2 operator-(vec2 A) { vec2 r = { -A.x, -A.y }; return(r); }
inline vec2 operator+(vec2 A, vec2 B) { return Add(A, B); }
inline vec2 operator-(vec2 A, vec2 B) { return Sub(A, B); }
inline vec2 operator*(vec2 A, float B) { return Mul(A, B); }
inline vec2 operator*(float B, vec2 A) { return Mul(A, B); }
inline vec2 operator/(vec2 A, float B) { return Div(A, B); }
inline vec2 &operator+=(vec2 &A, vec2 B) { return(A = A + B); }
inline vec2 &operator-=(vec2 &A, vec2 B) { return(A = A - B); }
inline vec2 &operator*=(vec2 &A, float B) { return(A = A * B); }
inline vec2 &operator/=(vec2 &A, float B) { return(A = A / B); }
inline bool operator==(vec2 &A, vec2 &B) { return A.x == B.x && A.y == B.y; }
inline bool operator!=(vec2 &A, vec2 &B) { return !(A == B); }

inline vec3 operator-(vec3 A) { vec3 r = { -A.x, -A.y, -A.z }; return(r); }
inline vec3 operator+(vec3 A, vec3 B) { return Add(A, B); }
inline vec3 operator-(vec3 A, vec3 B) { return Sub(A, B); }
inline vec3 operator*(vec3 A, float B) { return Mul(A, B); }
inline vec3 operator*(float B, vec3 A) { return Mul(A, B); }
inline vec3 operator/(vec3 A, float B) { return Div(A, B); }
inline vec3 &operator+=(vec3 &A, vec3 B) { return(A = A + B); }
inline vec3 &operator-=(vec3 &A, vec3 B) { return(A = A - B); }
inline vec3 &operator*=(vec3 &A, float B) { return(A = A * B); }
inline vec3 &operator/=(vec3 &A, float B) { return(A = A / B); }
inline bool operator==(vec3 &A, vec3 &B) { return A.x == B.x && A.y == B.y && A.z == B.z; }
inline bool operator!=(vec3 &A, vec3 &B) { return !(A == B); }

inline vec4 operator-(vec4 A) { vec4 r = { -A.x, -A.y, -A.z, -A.w }; return(r); }
inline vec4 operator+(vec4 A, vec4 B) { return Add(A, B); }
inline vec4 operator-(vec4 A, vec4 B) { return Sub(A, B); }
inline vec4 operator*(vec4 A, float B) { return Mul(A, B); }
inline vec4 operator*(float B, vec4 A) { return Mul(A, B); }
inline vec4 operator/(vec4 A, float B) { return Div(A, B); }
inline vec4 &operator+=(vec4 &A, vec4 B) { return(A = A + B); }
inline vec4 &operator-=(vec4 &A, vec4 B) { return(A = A - B); }
inline vec4 &operator*=(vec4 &A, float B) { return(A = A * B); }
inline vec4 &operator/=(vec4 &A, float B) { return(A = A / B); }
inline bool operator==(vec4 &A, vec4 &B) { return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w; }
inline bool operator!=(vec4 &A, vec4 &B) { return !(A == B); }

inline vec3 operator*(mat3 A, vec3 v) { return(Mul(A, v)); }
inline mat3 operator*(mat3 A, mat3 B) { return(Mul(A, B)); }
inline mat3 &operator*=(mat3 &A, mat3 B) { A = Mul(A, B); return A; }
inline vec4 operator*(mat4 A, vec4 v) { return(Mul(A, v)); }
inline mat4 operator*(mat4 A, mat4 B) { return(Mul(A, B)); }
inline mat4 &operator*=(mat4 &A, mat4 B) { A = Mul(A, B); return A; }

inline vec3 Lerp(vec3 From, vec3 to, float ratio) { return((1.0f - ratio) * From + to * ratio); }
inline vec4 Lerp(vec4 From, vec4 to, float ratio) { return((1.0f - ratio) * From + to * ratio); }

inline mat3 Add(mat3& A, mat3& B)
{
    mat3 Result;

    Result[0] = A[0] + B[0];
    Result[1] = A[1] + B[1];
    Result[2] = A[2] + B[2];

    return Result;
}

inline mat3 Mul(mat3& A, mat3& B)
{
    mat3 Result;

    Result.columns[0][0] = Dot(vec3(A[0][0], A[1][0], A[2][0]), B[0]);
    Result.columns[0][1] = Dot(vec3(A[0][1], A[1][1], A[2][1]), B[0]);
    Result.columns[0][2] = Dot(vec3(A[0][2], A[1][2], A[2][2]), B[0]);

    Result.columns[1][0] = Dot(vec3(A[0][0], A[1][0], A[2][0]), B[1]);
    Result.columns[1][1] = Dot(vec3(A[0][1], A[1][1], A[2][1]), B[1]);
    Result.columns[1][2] = Dot(vec3(A[0][2], A[1][2], A[2][2]), B[1]);

    Result.columns[2][0] = Dot(vec3(A[0][0], A[1][0], A[2][0]), B[2]);
    Result.columns[2][1] = Dot(vec3(A[0][1], A[1][1], A[2][1]), B[2]);
    Result.columns[2][2] = Dot(vec3(A[0][2], A[1][2], A[2][2]), B[2]);

    return Result;
}

inline vec3 Mul(mat3& A, vec3 V)
{
    return A[0] * V.x + A[1] * V.y + A[2] * V.z;
}

inline mat4 Add(mat4 &A, mat4 &B)
{
    mat4 Result;

    Result[0] = A[0] + B[0];
    Result[1] = A[1] + B[1];
    Result[2] = A[2] + B[2];
    Result[3] = A[3] + B[3];

    return Result;
}

inline mat4 Mul(mat4 &A, mat4 &B)
{
    mat4 Result;

    Result.columns[0][0] = Dot(vec4(A[0][0], A[1][0], A[2][0], A[3][0]), B[0]);
    Result.columns[0][1] = Dot(vec4(A[0][1], A[1][1], A[2][1], A[3][1]), B[0]);
    Result.columns[0][2] = Dot(vec4(A[0][2], A[1][2], A[2][2], A[3][2]), B[0]);
    Result.columns[0][3] = Dot(vec4(A[0][3], A[1][3], A[2][3], A[3][3]), B[0]);

    Result.columns[1][0] = Dot(vec4(A[0][0], A[1][0], A[2][0], A[3][0]), B[1]);
    Result.columns[1][1] = Dot(vec4(A[0][1], A[1][1], A[2][1], A[3][1]), B[1]);
    Result.columns[1][2] = Dot(vec4(A[0][2], A[1][2], A[2][2], A[3][2]), B[1]);
    Result.columns[1][3] = Dot(vec4(A[0][3], A[1][3], A[2][3], A[3][3]), B[1]);

    Result.columns[2][0] = Dot(vec4(A[0][0], A[1][0], A[2][0], A[3][0]), B[2]);
    Result.columns[2][1] = Dot(vec4(A[0][1], A[1][1], A[2][1], A[3][1]), B[2]);
    Result.columns[2][2] = Dot(vec4(A[0][2], A[1][2], A[2][2], A[3][2]), B[2]);
    Result.columns[2][3] = Dot(vec4(A[0][3], A[1][3], A[2][3], A[3][3]), B[2]);

    Result.columns[3][0] = Dot(vec4(A[0][0], A[1][0], A[2][0], A[3][0]), B[3]);
    Result.columns[3][1] = Dot(vec4(A[0][1], A[1][1], A[2][1], A[3][1]), B[3]);
    Result.columns[3][2] = Dot(vec4(A[0][2], A[1][2], A[2][2], A[3][2]), B[3]);
    Result.columns[3][3] = Dot(vec4(A[0][3], A[1][3], A[2][3], A[3][3]), B[3]);

    return Result;
}

inline vec4 Mul(mat4 &A, vec4 V)
{
    return A[0] * V.x + A[1] * V.y + A[2] * V.z + A[3] * V.w;
}

// returns 3x3 outer product of A and row major representation of B
inline mat3 OuterProduct(vec3 A, vec3 B)
{
    mat3 Result;

    Result[0][0] = A.x*B.x;
    Result[0][1] = A.y*B.x;
    Result[0][2] = A.z*B.x;
    Result[1][0] = A.x*B.y;
    Result[1][1] = A.y*B.y;
    Result[1][2] = A.z*B.y;
    Result[2][0] = A.x*B.z;
    Result[2][1] = A.y*B.z;
    Result[2][2] = A.z*B.z;

    return Result;
}

inline mat4 OuterProduct(vec4 A, vec4 B)
{
    mat3 Result;

    Result[0][0] = A.x*B.x;
    Result[0][1] = A.y*B.x;
    Result[0][2] = A.z*B.x;
    Result[0][3] = A.w*B.x;
    Result[1][0] = A.x*B.y;
    Result[1][1] = A.y*B.y;
    Result[1][2] = A.z*B.y;
    Result[1][3] = A.w*B.y;
    Result[2][0] = A.x*B.z;
    Result[2][1] = A.y*B.z;
    Result[2][2] = A.z*B.z;
    Result[2][3] = A.w*B.z;
    Result[3][0] = A.x*B.w;
    Result[3][1] = A.y*B.w;
    Result[3][2] = A.z*B.w;
    Result[3][3] = A.w*B.w;

    return Mat4FromMat3(Result);
}

inline mat3 Mat3FromMat4(mat4 From)
{
    mat3 Result;

    Result.columns[0][0] = From[0][0];
    Result.columns[0][1] = From[0][1];
    Result.columns[0][2] = From[0][2];
    Result.columns[1][0] = From[1][0];
    Result.columns[1][1] = From[1][1];
    Result.columns[1][2] = From[1][2];
    Result.columns[2][0] = From[2][0];
    Result.columns[2][1] = From[2][1];
    Result.columns[2][2] = From[2][2];

    return Result;
}

inline mat4 Mat4FromMat3(mat3 From)
{
    mat4 Result;
    
    Result.columns[0][0] = From[0][0];
    Result.columns[0][1] = From[0][1];
    Result.columns[0][2] = From[0][2];
    Result.columns[0][3] = 0.f;
    Result.columns[1][0] = From[1][0];
    Result.columns[1][1] = From[1][1];
    Result.columns[1][2] = From[1][2];
    Result.columns[1][3] = 0.f;
    Result.columns[2][0] = From[2][0];
    Result.columns[2][1] = From[2][1];
    Result.columns[2][2] = From[2][2];
    Result.columns[2][3] = 0.f;
    Result.columns[3][0] = 0.f;
    Result.columns[3][1] = 0.f;
    Result.columns[3][2] = 0.f;
    Result.columns[3][3] = 1.f;

    return Result;
}

inline mat4 mat4::GetTranspose()
{
    mat4 Result;

    for(int Col = 0; Col < 4; ++Col)
    {
        for(int Row = 0; Row < 4; ++Row)
        {
            Result[Col][Row] = columns[Row][Col];
        }
    }

    return Result;
}

inline mat3 mat3::GetTranspose()
{
    mat3 Result;

    for(int Col = 0; Col < 3; ++Col)
    {
        for(int Row = 0; Row < 3; ++Row)
        {
            Result[Col][Row] = columns[Row][Col];
        }
    }

    return Result;
}

inline mat4 mat4::GetInverse()
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

    mat4 Result;

    for (i = 0; i < 16; i++)
    {
        Result.e[i] = inv[i] * det;
    }

    return Result;
}

inline mat3 mat3::GetInverse() // TODO(Kevin): NEED TESTS
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

    mat3 inv =
    {
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

inline quat quat::GetConjugate()
{
    quat Result;

    Result.w = w;
    Result.x = -x;
    Result.y = -y;
    Result.z = -z;

    return Result;
}

inline quat quat::GetInverseUnitQuaternion()
{
    quat Result = Div(GetConjugate(), Dot(*this, *this));
    return Result;
}

inline quat Add(quat A, quat B)
{
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    A.w += B.w;
    return A;
}

inline quat Sub(quat A, quat B)
{
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    A.w -= B.w;
    return A;
}

inline quat Div(quat A, float Scale)
{
    float OneOverScale = 1.0f / Scale;
    A.w *= OneOverScale;
    A.x *= OneOverScale;
    A.y *= OneOverScale;
    A.z *= OneOverScale;
    return A;
}

inline quat Normalize(quat Q)
{
    float S = Q.w*Q.w + Q.x*Q.x + Q.y*Q.y + Q.z*Q.z;
    if (S > 0.0f)
    {
        float Inv = 1.0f / sqrtf(S);
        Q.x *= Inv;
        Q.y *= Inv;
        Q.z *= Inv;
        Q.w *= Inv;
    }
    return Q;
}

inline quat HamiltonProduct(quat B, quat A)
{
    // https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

    quat R;

    R.w = B.w*A.w - B.x*A.x - B.y*A.y - B.z*A.z;
    R.x = B.w*A.x + B.x*A.w + B.y*A.z - B.z*A.y;
    R.y = B.w*A.y - B.x*A.z + B.y*A.w + B.z*A.x;
    R.z = B.w*A.z + B.x*A.y - B.y*A.x + B.z*A.w;

    return R;
}

inline float Dot(quat UnitQuaternionA, quat UnitQuaternionB)
{
    return UnitQuaternionA.x * UnitQuaternionB.x + UnitQuaternionA.y * UnitQuaternionB.y + 
           UnitQuaternionA.z * UnitQuaternionB.z + UnitQuaternionA.w * UnitQuaternionB.w;
}

inline bool Similar(quat A, quat B, float Epsilon)
{
    return(GMATH_abs(Dot(A, B) - 1.f) <= Epsilon);
}

inline mat3 Mat3FromQuat(quat Rotation)
{
    // https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    mat3 RotationMatrix;

    float x2 = Rotation.x * Rotation.x;
    float y2 = Rotation.y * Rotation.y;
    float z2 = Rotation.z * Rotation.z;

    float xy = Rotation.x * Rotation.y;
    float zw = Rotation.z * Rotation.w;
    float xz = Rotation.x * Rotation.z;
    float yw = Rotation.y * Rotation.w;
    float yz = Rotation.y * Rotation.z;
    float xw = Rotation.x * Rotation.w;

    RotationMatrix.columns[0][0] = 1.0f - 2.0f * (y2 + z2);
    RotationMatrix.columns[1][0] = 2.0f * (xy - zw);
    RotationMatrix.columns[2][0] = 2.0f * (xz + yw);

    RotationMatrix.columns[0][1] = 2.0f * (xy + zw);
    RotationMatrix.columns[1][1] = 1.0f - 2.0f * (x2 + z2);
    RotationMatrix.columns[2][1] = 2.0f * (yz - xw);

    RotationMatrix.columns[0][2] = 2.0f * (xz - yw);
    RotationMatrix.columns[1][2] = 2.0f * (yz + xw);
    RotationMatrix.columns[2][2] = 1.0f - 2.0f * (x2 + y2);

    return RotationMatrix;
}

inline mat4 Mat4FromQuat(quat Rotation)
{
    return Mat4FromMat3(Mat3FromQuat(Rotation));
}

inline quat QuatFromMat4(mat4 RotationMatrix)
{
    quat R;

    R.w = SquareRoot(1.0f + RotationMatrix[0][0] + RotationMatrix[1][1] + RotationMatrix[2][2]) * 0.5f;
    float w4 = 4.0f * R.w;
    R.x = (RotationMatrix[1][2] - RotationMatrix[2][1]) / w4;
    R.y = (RotationMatrix[2][0] - RotationMatrix[0][2]) / w4;
    R.z = (RotationMatrix[0][1] - RotationMatrix[1][0]) / w4;

    return R;
}

#if 0
inline quat QuatFromEuler_HamiltonProduct_XZY(float X, float Y, float Z)
{
    quat Qx = quat(X, GMATH_FORWARD);
    quat Qy = quat(Y, GMATH_UP);
    quat Qz = quat(Z, GMATH_RIGHT);
    quat Qxzy = HamiltonProduct(Qy, HamiltonProduct(Qz, Qx));
    return Qxzy;
}
#endif

inline quat QuatFromEuler_YZX(float X, float Y, float Z)
{
    float cx = Cos(X * 0.5f);
    float sx = Sin(X * 0.5f);
    float cy = Cos(Y * 0.5f);
    float sy = Sin(Y * 0.5f);
    float cz = Cos(Z * 0.5f);
    float sz = Sin(Z * 0.5f);

    quat Result;

    Result.w =  cx * cz * cy - sx * sz * sy;
    Result.x =  sx * cz * cy + cx * sz * sy;
    Result.y =  sx * sz * cy + cx * cz * sy;
    Result.z =  cx * sz * cy - sx * cz * sy;

    return Result;
}

inline quat QuatFromEuler_YZX(vec3 EulerAnglesRadian)
{
    return QuatFromEuler_YZX(EulerAnglesRadian.x, EulerAnglesRadian.y, EulerAnglesRadian.z);
}

inline vec3 RotateVector(vec3 Vector, quat Rotation)
{
    quat VectorQuaternion = { 0.f, Vector.x, Vector.y, Vector.z };
    quat RotatedVector = HamiltonProduct(Rotation, HamiltonProduct(VectorQuaternion, Rotation.GetConjugate()));
    return vec3(RotatedVector.x, RotatedVector.y, RotatedVector.z);
}

inline quat RotationFromTo(vec3 FromDirection, vec3 ToDirection)
{
    vec3 Start = Normalize(FromDirection);
    vec3 Dest = Normalize(ToDirection);

    float CosTheta = Dot(Start, Dest);
    vec3 RotationAxis;
    quat RotationQuat;

    RotationAxis = Cross(Start, Dest);
    if (CosTheta >= -1 + 0.0001f)
    {
        float s = SquareRoot((1 + CosTheta) * 2);
        float sin_of_half_angle = 1 / s;

        RotationQuat = quat(
            s * 0.5f, // recall cos(theta/2) trig identity
            RotationAxis.x * sin_of_half_angle,
            RotationAxis.y * sin_of_half_angle,
            RotationAxis.z * sin_of_half_angle
        );
    }
    else
    {
        // When vectors in opposite directions, there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to Start
        RotationAxis = Cross(vec3(0.0f, 0.0f, 1.0f), Start);
        if (Dot(RotationAxis, RotationAxis) < 0.01f) // parallel
        {
            RotationAxis = Cross(vec3(1.0f, 0.0f, 0.0f), Start); // try again
        }
        RotationQuat = quat(GMATH_PI, RotationAxis);
    }

    return RotationQuat;
}

// inline quat RotationDifference(quat A, quat B)
// {
//     // TODO(Kevin): What is the correct order here?
//     quat Result = HamiltonProduct(B, A.GetInverseUnitQuaternion());
//     return Result;
// }

inline quat Scale(quat A, float Scale)
{
    A.w *= Scale;
    A.x *= Scale;
    A.y *= Scale;
    A.z *= Scale;
    return A;
}

inline quat Slerp(quat From, quat To, float t)
{
    quat Start = Normalize(From);
    quat End = Normalize(To);

    // From Jolt Physics Library Quat.inl

    // Difference at which to LERP instead of SLERP
    const float Delta = 0.0001f;

    // Calc cosine
    float SignScale1 = 1.0f;
    float CosOmega = Dot(Start, End);

    // Adjust signs (if necessary)
    if (CosOmega < 0.0f)
    {
        CosOmega = -CosOmega;
        SignScale1 = -1.0f;
    }

    // Calculate coefficients
    float Scale0, Scale1;
    if (1.0f - CosOmega > Delta)
    {
        // Standard case (slerp)
        float Omega = ACos(CosOmega);
        float SinOmega = Sin(Omega);
        Scale0 = Sin((1.0f - t) * Omega) / SinOmega;
        Scale1 = SignScale1 * Sin(t * Omega) / SinOmega;
    }
    else
    {
        // Quaternions are very close so we can do a linear interpolation
        Scale0 = 1.0f - t;
        Scale1 = SignScale1 * t;
    }

    // Interpolate between the two quaternions
    return Normalize(Add(Scale(Start, Scale0), Scale(End, Scale1)));
}

inline quat LookRotation1(vec3 LookDirection, vec3 Up)
{
    // NOTE(Kevin): This function has direction semantics built-in
    //              +X = Forward, +Y = Up, +Z = Right

    vec3 Forward = Normalize(LookDirection);
    vec3 Right = Normalize(Cross(LookDirection, Up));
    Up = Cross(Right, LookDirection);

    mat4 RotationMatrix;
    RotationMatrix.columns[0] = vec4(Forward, 0);
    RotationMatrix.columns[1] = vec4(Up, 0);
    RotationMatrix.columns[2] = vec4(Right, 0);
    RotationMatrix.columns[3] = vec4(0,0,0,1);
    quat Rotation = QuatFromMat4(RotationMatrix);
    return Rotation;
}

inline void LookRotation2(vec3 LookDirection, real32 *Yaw, real32 *Pitch)
{
    real32 CartesianX = LookDirection.x;
    real32 CartesianY = -LookDirection.z;
    *Yaw = ATan2(CartesianY, CartesianX);
    real32 HorizontalLength = Length({ LookDirection.x, LookDirection.z });
    *Pitch = ATan(LookDirection.y / HorizontalLength);
}

inline quat LookRotation3(vec3 LookDirection)
{
    real32 Yaw, Pitch;
    LookRotation2(LookDirection, &Yaw, &Pitch);
    quat Rotation = QuatFromEuler_YZX(0, Yaw, Pitch);
    return Rotation;
}

inline quat OrientationFromDirection(vec3 Direction)
{
    // NOTE(Kevin): This function has direction semantics built-in
    //              +X = Forward, +Y = Up, +Z = Right

    return RotationFromTo(GMATH_FORWARD, Direction);
}

inline vec3 DirectionFromOrientation(quat Orientation)
{
    // NOTE(Kevin): This function has direction semantics built-in
    //              +X = Forward, +Y = Up, +Z = Right

    return RotateVector(GMATH_FORWARD, Orientation);
}

// inline vec3 EulerFromQuat(quat q)
// {
//     vec3 euler_angles;

//     float x2 = q.x * q.x;
//     float y2 = q.y * q.y;
//     float z2 = q.z * q.z;
//     float xy = q.x * q.y;
//     float zw = q.z * q.w;
//     float xz = q.x * q.z;
//     float yw = q.y * q.w;
//     float yz = q.y * q.z;
//     float xw = q.x * q.w;

//     float r10 = 2.0f * (xy + zw);
//     float r20 = 2.0f * (xz - yw);
//     float r00 = 1.0f - 2.0f * (y2 + z2);
//     float r12 = 2.0f * (yz - xw);
//     float r11 = 1.0f - 2.0f * (x2 + z2);
//     float r21 = 2.0f * (yz + xw) ;
//     float r22 = 1.0f - 2.0f * (x2 + y2);

//     if(r10 < 1.f)
//     {
//         if(r10 > -1.f)
//         {
//             euler_angles.z = ASin(r10);
//             euler_angles.y = ATan2(-r20, r00);
//             euler_angles.x = ATan2(-r12, r11);
//         }
//         else // r10 = -1
//         {
//             euler_angles.z = -GM_PI / 2.f;
//             euler_angles.y = -ATan2(r21, r22);
//             euler_angles.x = 0;
//         }
//     }
//     else
//     {
//         euler_angles.z = GM_PI / 2.f;
//         euler_angles.y = ATan2(r21, r22);
//         euler_angles.x = 0;
//     }

//     return euler_angles;
// }

inline mat4 TranslationMatrix(float X, float Y, float Z)
{
    mat4 Result = mat4();

    Result[3][0] = X;
    Result[3][1] = Y;
    Result[3][2] = Z;
    Result[3][3] = 1.f;

    return Result;
}

inline mat4 TranslationMatrix(vec3 Translation)
{
    return TranslationMatrix(Translation.x, Translation.y, Translation.z);
}

inline mat3 TranslationMatrix2D(vec2 Translation) // TODO(Kevin): NEED TESTS
{
    mat3 Result = mat3();

    Result[2][0] = Translation.x;
    Result[2][1] = Translation.y;
    Result[2][2] = 1.f;

    return Result;
}

inline mat4 RotationMatrix(quat Rotation)
{
    return Mat4FromQuat(Rotation);
}

inline mat3 RotationMatrix2D(float Rotation)
{
    mat3 Result;

    float cr = Cos(Rotation);
    float sr = Sin(Rotation);
    Result[0][0] = cr;
    Result[0][1] = sr;
    Result[1][0] = -sr;
    Result[1][1] = cr;
    Result[2][2] = 1.f;

    return Result;
}

inline mat4 ScaleMatrix(float Scale)
{
    return ScaleMatrix(Scale,Scale,Scale);
}

inline mat4 ScaleMatrix(float ScaleX, float ScaleY, float ScaleZ)
{
    mat4 Result;
    Result[0][0] = ScaleX;
    Result[1][1] = ScaleY;
    Result[2][2] = ScaleZ;
    Result[3][3] = 1.f;
    return Result;
}

inline mat4 ScaleMatrix(vec3 Scale)
{
    return ScaleMatrix(Scale.x, Scale.y, Scale.z);
}

inline mat3 ScaleMatrix2D(vec2 Scale) // TODO(Kevin): NEED TESTS
{
    mat3 Result = mat3();
    Result[0][0] = Scale.x;
    Result[1][1] = Scale.y;
    Result[2][2] = 1.f;
    return Result;
}

inline mat4 CameraSpaceMatrixLookAt(vec3 Eye, vec3 Target, vec3 EyeUpVector)
{
    vec3 Direction = Normalize(Target - Eye);
    vec3 Right = Normalize(Cross(Direction, EyeUpVector));
    vec3 Up = Cross(Right, Direction); // Magnitude is equal to Sin(Theta between Direction and Right) = 1

    mat4 Result = {};

    // Standard basis vectors for world coordinate space transformed into camera space
    vec4 WorldXAxisInCameraSpace = vec4(Right.x, Up.x, -Direction.x, 0.f);
    vec4 WorldYAxisInCameraSpace = vec4(Right.y, Up.y, -Direction.y, 0.f);
    vec4 WorldZAxisInCameraSpace = vec4(Right.z, Up.z, -Direction.z, 0.f);
    vec4 WorldCameraPosInCameraSpace = vec4(-Dot(Right, Eye), -Dot(Up, Eye), Dot(Direction, Eye), 1.f);
    Result[0] = WorldXAxisInCameraSpace;
    Result[1] = WorldYAxisInCameraSpace;
    Result[2] = WorldZAxisInCameraSpace;
    Result[3] = WorldCameraPosInCameraSpace;

    return Result;
}

inline mat4 ClipSpaceFromCameraSpace_PerspectiveProjection(float VerticalFOV, float AspectRatio,
                                                           float NearClip, float FarClip, bool NearClipZero)
{
    float TanHalfVerticalFOV = Tan(VerticalFOV / 2.0f);

    mat4 Result;

    Result[0][0] = 1.f / (AspectRatio * TanHalfVerticalFOV);
    Result[1][1] = 1.f / (TanHalfVerticalFOV);
    Result[2][3] = -1.f;
    Result[3][3] = 0.f;

    float ZNearClip = NearClipZero ? 0.0f : -1.0f;
    Result[2][2] = (FarClip + (-ZNearClip * NearClip)) / (NearClip - FarClip);
    Result[3][2] = ((FarClip + (-ZNearClip * FarClip)) * NearClip) / (NearClip - FarClip);

    return Result;
}

// inline mat4 ProjectionMatrixOrthographic(float Left, float Right, float Bottom, float Top,
//                                          float NearClip, float FarClip)
// {
//     mat4 Result = mat4();

//     Result[0][0] = 2.f / (Right - Left);
//     Result[1][1] = 2.f / (Top - Bottom);
//     Result[2][2] = - 2.f / (FarClip - NearClip);
//     Result[3][0] = - (Right + Left) / (Right - Left);
//     Result[3][1] = - (Top + Bottom) / (Top - Bottom);
//     Result[3][2] = - (FarClip + NearClip) / (FarClip - NearClip);

//     return Result;
// }

inline mat4 ProjectionMatrixOrthographicNoZ(float Left, float Right, float Bottom, float Top)
{
    mat4 Result = mat4();

    Result[0][0] = 2.f / (Right - Left);
    Result[1][1] = 2.f / (Top - Bottom);
    Result[2][2] = -1.f;
    Result[3][0] = -(Right + Left) / (Right - Left);
    Result[3][1] = -(Top + Bottom) / (Top - Bottom);

    return Result;
}

inline mat3 ProjectionMatrixOrthographic2D(float Left, float Right, float Bottom, float Top)
{
    mat3 Result = mat3();

    Result[0][0] = 2.f / (Right - Left);
    Result[1][1] = 2.f / (Top - Bottom);
    Result[2][0] = -(Right + Left) / (Right - Left);
    Result[2][1] = -(Top + Bottom) / (Top - Bottom);
    Result[2][2] = 1.f;

    return Result;
}

