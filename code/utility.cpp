#include "common.h"
#include "utility.h"

// external
random_series RNG;

vec3 ScreenPointToWorldRay(
    i32 BackBufferWidth,
    i32 BackBufferHeight,
    mat4 &PerspectiveMatrix,
    mat4 &ViewMatrix,
    ivec2 ScreenSpaceCoords)
{
    // https://antongerdelan.net/opengl/raycasting.html
    // assuming 3D world view is taking up the entire window
    // Reversing perspective divide not necessary because this is a vector/direction/ray with no intrinsic depth.
    float x_NDC = ((float)ScreenSpaceCoords.x / (float)BackBufferWidth) * 2.f - 1.f;
    float y_NDC = (float(BackBufferHeight - ScreenSpaceCoords.y) / (float)BackBufferHeight) * 2.f - 1.f;
    vec4 ray_in_clipspace = vec4(x_NDC, y_NDC, -1.f, 1.f);
    vec4 ray_in_viewspace = PerspectiveMatrix.GetInverse() * ray_in_clipspace;
    ray_in_viewspace = vec4(ray_in_viewspace.x, ray_in_viewspace.y, -1.f, 0.f);
    vec3 ray_in_worldspace = Normalize((ViewMatrix.GetInverse() * ray_in_viewspace).xyz);
    return ray_in_worldspace;
}

vec3 ScreenPointToWorldPoint(
    i32 BackBufferWidth,
    i32 BackBufferHeight,
    mat4 &PerspectiveMatrix,
    mat4 &ViewMatrix,
    ivec2 ScreenSpaceCoords,
    float z_NDC)
{
    float x_NDC = ((float)ScreenSpaceCoords.x / (float)BackBufferWidth) * 2.f - 1.f;
    float y_NDC = (float(BackBufferHeight - ScreenSpaceCoords.y) / (float)BackBufferHeight) * 2.f - 1.f;
    vec4 point_in_clipspace = vec4(x_NDC, y_NDC, z_NDC, 1.f);
    // For points, reverse perspective divide after the inverse projection matrix transformation because it's easier that way.
    vec4 point_in_viewspace_before_perspective_divide = PerspectiveMatrix.GetInverse() * point_in_clipspace;
    vec4 point_in_viewspace = point_in_viewspace_before_perspective_divide / point_in_viewspace_before_perspective_divide.w;
    vec4 point_in_worldspace = ViewMatrix.GetInverse() * point_in_viewspace;
    return point_in_worldspace.xyz;
}

vec3 WorldPointToScreenPoint(
    i32 BackBufferWidth,
    i32 BackBufferHeight,
    mat4 &PerspectiveMatrix,
    mat4 &ViewMatrix,
    vec3 CameraPosition,
    vec3 CameraDirection,
    vec3 WorldPosition)
{
    vec4 clipspaceCoordinates = PerspectiveMatrix * ViewMatrix * vec4(WorldPosition, 1.f);
    float screenspaceRatioX = ((clipspaceCoordinates.x / clipspaceCoordinates.w) + 1.f) / 2.f;
    float screenspaceRatioY = 1.f - (((clipspaceCoordinates.y / clipspaceCoordinates.w) + 1.f) / 2.f);
    float internalResolutionWidth = (float)BackBufferWidth;
    float internalResolutionHeight = (float)BackBufferHeight;
    float distanceFromCameraWCS = Dot(WorldPosition - CameraPosition, CameraDirection);
    return vec3(screenspaceRatioX * internalResolutionWidth, screenspaceRatioY * internalResolutionHeight, distanceFromCameraWCS);
}

// only checks in direction from
bool IntersectPlaneAndLine(
    vec3 pointOnPlane, vec3 normalOfPlane,
    vec3 pointOnLine, vec3 directionOfLine,
    vec3 *intersectionPoint)
{
    // vec3 pp = pointOnPlane;
    // vec3 pn = normalOfPlane;
    // vec3 lp = pointOnLine;
    // vec3 lv = directionOfLine;
    // line x = lp.x + lv.x * t, line y = lp.y + lv.y * t, line z = lp.z + lv.z * t
    // plane(X, Y, Z) = pn.x * X + pn.y * Y + pn.z * Z - Dot(pn, pp)
    // pn.x * (lp.x + lv.x * t) + pn.y * (lp.y + lv.y * t) + pn.z * (lp.z + lv.z * t) = Dot(pn, pp)
    // pn.x*lp.x + pn.x*lv.x*t + pn.y*lp.y + pn.y*lv.y*t + pn.z*lp.z + pn.z*lv.z*t = Dot(pn, pp)
    // pn.x*lv.x*t + pn.y*lv.y*t + pn.z*lv.z*t = Dot(pn, pp) - pn.x*lp.x - pn.y*lp.y - pn.z*lp.z
    // t (pn.x*lv.x + pn.y*lv.y + pn.z*lv.z) = Dot(pn, pp) - pn.x*lp.x - pn.y*lp.y - pn.z*lp.z
    // t = (Dot(pn, pp) - pn.x*lp.x - pn.y*lp.y - pn.z*lp.z) / (pn.x*lv.x + pn.y*lv.y + pn.z*lv.z)
    // t = (Dot(pn, pp) - (pn.x*lp.x + pn.y*lp.y + pn.z*lp.z)) / (pn.x*lv.x + pn.y*lv.y + pn.z*lv.z)

    float denominator = Dot(normalOfPlane, directionOfLine);
    if (GM_abs(denominator) < 0.000001f)
        return false;

    float t = (Dot(normalOfPlane, pointOnPlane) - Dot(normalOfPlane, pointOnLine)) / denominator;
    *intersectionPoint = pointOnLine + directionOfLine * t;
    return true;
}

bool IntersectPlaneAndLineWithDirections(
    vec3 pointOnPlane, vec3 normalOfPlane,
    vec3 pointOnLine, vec3 directionOfLine,
    vec3 *intersectionPoint)
{
    float denominator = Dot(normalOfPlane, directionOfLine);
    if (denominator > -0.000001f)
        return false;

    float t = (Dot(normalOfPlane, pointOnPlane) - Dot(normalOfPlane, pointOnLine)) / denominator;
    *intersectionPoint = pointOnLine + directionOfLine * t;
    return true;
}

float HorizontalFOVToVerticalFOV_RadianToRadian(
    float FOVXInRadians, float AspectRatio)
{
    float FOVY = 2.0f * atan(tan(FOVXInRadians / 2.0f) / AspectRatio);
    return FOVY;
}

void BlitRect(
    u8 *A, int AW, int AH,
    u8 *B, int BW, int BH,
    int x, int y, size_t pixelsz)
{
    AW *= (int)pixelsz;
    BW *= (int)pixelsz;
    x *= (int)pixelsz;
    for (int row = 0; row < BH; ++row)
    {
        int desty = y + row;
        if (desty < 0 || desty >= AH)
            continue;
        u8 *dest = A + (desty * AW + x);
        u8 *src  = B + row * BW;
        int cpyw = BW;
        if (x < 0) 
        {
            dest -= x;
            src -= x;
            cpyw += x;
        }
        if (x + BW > AW) 
        {
            cpyw = AW - x;
        }
        memcpy(dest, src, cpyw);
    }
}

i32 ShiftASCII(i32 keycodeASCII, bool shift)
{
    i32 keycode = keycodeASCII;

    if (shift)
    {
        if (97 <= keycode && keycode <= 122)
        {
            keycode -= 32;
        }
        else if (keycode == 50)
        {
            keycode = 64;
        }
        else if (49 <= keycode && keycode <= 53)
        {
            keycode -= 16;
        }
        else if (91 <= keycode && keycode <= 93)
        {
            keycode += 32;
        }
        else
        {
            switch (keycode)
            {
                case 48: { keycode = 41; } break;
                case 54: { keycode = 94; } break;
                case 55: { keycode = 38; } break;
                case 56: { keycode = 42; } break;
                case 57: { keycode = 40; } break;
                case 45: { keycode = 95; } break;
                case 61: { keycode = 43; } break;
                case 39: { keycode = 34; } break;
                case 59: { keycode = 58; } break;
                case 44: { keycode = 60; } break;
                case 46: { keycode = 62; } break;
                case 47: { keycode = 63; } break;
                case 96: { keycode = 126; } break;
            }
        }
    }

    return keycode;
}

std::string& RemoveCharactersFromEndOfString(std::string& str, char c)
{
    while (str.back() == c)
    {
        str.pop_back();
    }
    return str;
}

vec3 HSVToRGB(float h, float s, float v)
{
    // https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    u16 huedegree = u16(h * 359.f);
    float chroma = v*s;
    float m = v-chroma;
    float normalizedx = chroma * (1.f - abs(fmod((h * 359.f) / 60.f, 2.f) - 1.f));

    vec3 rgb;
    if (huedegree < 60)
        rgb = { chroma, normalizedx, 0.f };
    else if (huedegree < 120)
        rgb = { normalizedx, chroma, 0.f };
    else if (huedegree < 180)
        rgb = { 0.f, chroma, normalizedx };
    else if (huedegree < 240)
        rgb = { 0.f, normalizedx, chroma };
    else if (huedegree < 300)
        rgb = { normalizedx, 0.f, chroma };
    else if (huedegree < 360)
        rgb = { chroma, 0.f, normalizedx };
    rgb += vec3(m,m,m);
    return rgb;
}

// normalized rgb to hsv
vec3 RGBToHSV(float r, float g, float b)
{
    // https://en.wikipedia.org/wiki/Hue
    // https://www.rapidtables.com/convert/color/rgb-to-hsv.html

    float h = atan2(1.7320508f * (g - b), 2 * r - g - b);
    if (h < 0)
        h = GM_TWOPI + h;
    h = abs(h) / GM_TWOPI;

    float cmax = GM_max(r, GM_max(g, b));
    float cmin = GM_min(r, GM_min(g, b));
    float cdelta = cmax - cmin;
    float s = cmax == 0.f ? 0.f : cdelta / cmax;

    vec3 hsv = vec3(h, s, cmax);
    return hsv;
}


void ByteBufferInit(ByteBuffer* buffer)
{
    buffer->data = (uint8_t*) malloc(BYTE_BUFFER_DEFAULT_CAPACITY);
    memset(buffer->data, 0, BYTE_BUFFER_DEFAULT_CAPACITY);
    buffer->position = 0;
    buffer->size = 0;
    buffer->capacity = BYTE_BUFFER_DEFAULT_CAPACITY;
}

ByteBuffer ByteBufferNew()
{
    ByteBuffer buffer = {0};
    ByteBufferInit(&buffer);
    return buffer;
}

void ByteBufferFree(ByteBuffer* buffer)
{
    if(buffer && buffer->data) 
    {
        free(buffer->data);
    }
    buffer->size = 0;
    buffer->position = 0;
    buffer->capacity = 0;
}

void ByteBufferClear(ByteBuffer* buffer)
{
    buffer->size = 0;
    buffer->position = 0;   
}

void ByteBufferResize(ByteBuffer* buffer, size_t sz)
{
    uint8_t* data = (uint8_t*)realloc(buffer->data, sz);
    if(data == NULL)
    {
        return;
    }
    buffer->data = data;
    buffer->capacity = (uint32_t)sz;
}

void ByteBufferSeekToStart(ByteBuffer* buffer)
{
    buffer->position = 0;
}

void ByteBufferSeekToEnd(ByteBuffer* buffer)
{
    buffer->position = buffer->size;
}

void ByteBufferAdvancePosition(ByteBuffer* buffer, size_t sz)
{
    buffer->position += (uint32_t)sz; 
}

void __byteBufferWriteImpl(ByteBuffer* buffer, void* data, size_t sz)
{
    size_t totalWriteSize = buffer->position + sz;
    if(totalWriteSize >= buffer->capacity)
    {
        size_t capacity = buffer->capacity ? buffer->capacity * 2 : BYTE_BUFFER_DEFAULT_CAPACITY;
        while(capacity < totalWriteSize)
        {
            capacity *= 2;
        }
        ByteBufferResize(buffer, capacity);
    }
    memcpy(buffer->data + buffer->position, data, sz);
    buffer->position += (uint32_t)sz;
    buffer->size += (uint32_t)sz;
}

int ByteBufferWriteToFile(ByteBuffer* buffer, const char* filePath)
{
    FILE* fp;
    fp = fopen(filePath, "wb");

    if(!fp)
    {
        return 0;
    }

    fwrite(buffer->data, 1, buffer->size, fp);

    fclose(fp);

    return 1;
}

int ByteBufferReadFromFile(ByteBuffer* buffer, const char* filePath)
{
    if(!buffer)
    {
        return 0;
    }

    FILE* fp;
    fp = fopen(filePath, "rb");

    if(!fp)
    {
        return 0;
    }

    if(buffer->data)
    {
        ByteBufferFree(buffer);
    }

    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    ByteBuffer bb = ByteBufferNew();
    if(bb.capacity < sz)
    {
        size_t capacity = bb.capacity;
        while(capacity < sz)
        {
            capacity *= 2;
        }
        ByteBufferResize(&bb, capacity);
    }

    fread(bb.data, 1, sz, fp);
    bb.size = (uint32_t)sz;
    *buffer = bb;

    fclose(fp);

    return 1;
}

