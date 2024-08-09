

#define RGB255TO1(r,g,b) ((float)r)/255.f, ((float)g)/255.f, ((float)b)/255.f
#define RGBHEXTO1(hex) \
    float((hex & 0x00FF00000) >> 16)/255.f,\
    float((hex & 0x0000FF00) >> 8)/255.f,\
    float(hex & 0x000000FF)/255.f

template<typename T, int _count> struct NiceArray
{
    /** Nice array wrapper for when you want to keep track of how many active/relevant
        elements are in the array. Essentially a dynamic sized array/vector with a
        maximum defined capacity (s.t. it can be defined on stack or static storage). */

    T data[_count] = {};
    int count = 0;
    const int capacity = _count;

    // todo maybe Insert and Erase?

    bool Contains(T v)
    {
        for (int i = 0; i < count; ++i)
        {
            if (*(data + i) == v) return true;
        }
        return false;
    }

    void EraseAt(int index)
    {
        if (index < count - 1)
        {
            memmove(data + index, data + index + 1, (count - index - 1) * sizeof(*data));
        }
        --count;
    }

    void EraseFirstOf(T v)
    {
        for (int i = 0; i < count; ++i)
        {
            if (*(data + i) == v)
            {
                EraseAt(i);
                break;
            }
        }
    }

    void EraseAllOf(T v)
    {
        for (int i = 0; i < count; ++i)
        {
            if (*(data + i) == v)
            {
                EraseAt(i);
            }
        }
    }

    bool NotAtCapacity()
    {
        return count < capacity;
    }

    void PushBack(T elem)
    {
        data[count] = elem;
        ++count;
    }

    void PopBack()
    {
        --count;
        memset(data + count, 0, sizeof(*data));
    }

    T& At(int index)
    {
        return *(data + index);
    }

    T& At(unsigned int index)
    {
        return At((int) index);
    }

    T& Back()
    {
        return *(data + count - 1);
    }

    void ResetCount()
    {
        count = 0;
    }

    void ResetToZero()
    {
        memset(data, 0, capacity * sizeof(*data));
    }
};


bool GLHasErrors()
{
    GLenum error = glGetError();

    if (error == GL_NO_ERROR) return false;

    while (error != GL_NO_ERROR)
    {
        const char* error_str = "";
        switch (error)
        {
            case GL_INVALID_OPERATION:
                error_str = "INVALID_OPERATION\n";
                break;
            case GL_INVALID_ENUM:
                error_str = "INVALID_ENUM\n";
                break;
            case GL_INVALID_VALUE:
                error_str = "INVALID_VALUE\n";
                break;
            case GL_OUT_OF_MEMORY:
                error_str = "OUT_OF_MEMORY\n";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error_str = "INVALID_FRAMEBUFFER_OPERATION\n";
                break;
        }

        fprintf(stderr, error_str);
        error = glGetError();
        assert(false);
    }

    return true;
}



// PRIMITIVE DRAWING

static u32 __handleVisualVaoId = 0;
static u32 __handleVisualVboId = 0;

static GPUShader __handle_shader;
static const char* __handle_shader_vs =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "void main() {\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "}\n";
static const char* __handle_shader_fs =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform vec4 handleColour;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "void main() {\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(handleColour.xyz, 0.3);\n"
    "    } else {\n"
    "        colour = handleColour;\n"
    "    }\n"
    "}\n";

enum class PrimitiveDrawMode
{
    Triangles,
    Lines
};

static NiceArray<vec4, 128> __colors;
static NiceArray<PrimitiveDrawMode, 128> __drawMode;
static NiceArray<u16, 128> __verticesCounts; // vertex buffer sizes in number of floats
static NiceArray<float, 256 * 128> __vb;


void PrimitiveInitAll()
{
    glGenVertexArrays(1, &__handleVisualVaoId);
    glBindVertexArray(__handleVisualVaoId);
    glGenBuffers(1, &__handleVisualVboId);
    glBindBuffer(GL_ARRAY_BUFFER, __handleVisualVboId);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLCreateShaderProgram(__handle_shader, __handle_shader_vs, __handle_shader_fs);
}

void PrimitiveDrawAll(const mat4* projectionMatrix, const mat4* viewMatrix, GLuint sceneDepthTextureId, vec2 framebufferSize)
{
    if (__colors.count == 0) return;

    UseShader(__handle_shader);
    GLBindMatrix4fv(__handle_shader, "projectionMatrix", 1, projectionMatrix->ptr());
    GLBindMatrix4fv(__handle_shader, "viewMatrix", 1, viewMatrix->ptr());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
    GLBind1i(__handle_shader, "sceneDepthTexture", 0);

    GLBind2f(__handle_shader, "framebufferSize", framebufferSize.x, framebufferSize.y);

    u32 __vbIndexer = 0;
    for (int i = 0; i < __colors.count; ++i)
    {
        vec4 color = __colors.At(i);
        u16 vertexCount = __verticesCounts.At(i);
        float* vb = __vb.data + __vbIndexer;
        __vbIndexer += vertexCount * 3;

        GLBind4f(__handle_shader, "handleColour", color.x, color.y, color.z, color.w);
        glBindVertexArray(__handleVisualVaoId);
        glBindBuffer(GL_ARRAY_BUFFER, __handleVisualVboId);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)4 * vertexCount * 3, vb, GL_DYNAMIC_DRAW);
        if (__drawMode.At(i) == PrimitiveDrawMode::Triangles)
        {
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }
        else if (__drawMode.At(i) == PrimitiveDrawMode::Lines)
        {
            glDrawArrays(GL_LINES, 0, vertexCount);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    __colors.ResetCount();
    __drawMode.ResetCount();
    __verticesCounts.ResetCount();
    __vb.ResetCount();
}

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color)
{
    __colors.PushBack(color);
    __drawMode.PushBack(PrimitiveDrawMode::Triangles);
    __verticesCounts.PushBack(12 * 3);

    vec3 tangent = Normalize(Cross(normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, normal));
    quat q = quat(GM_DEG2RAD * 30.f, normal);
    for(int i = 0; i < 12; ++i)
    {
        vec3 a = center;
        vec3 b = center + tangent * radius;
        tangent = RotateVector(tangent, q);
        vec3 c = center + tangent * radius;

        __vb.PushBack(a.x);
        __vb.PushBack(a.y);
        __vb.PushBack(a.z);
        __vb.PushBack(b.x);
        __vb.PushBack(b.y);
        __vb.PushBack(b.z);
        __vb.PushBack(c.x);
        __vb.PushBack(c.y);
        __vb.PushBack(c.z);
    }
}

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius)
{
    PrimitiveDrawSolidDisc(center, normal, radius, vec4(RGB255TO1(248, 230, 60), 1.f));
}

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color)
{
    __colors.PushBack(color);
    __drawMode.PushBack(PrimitiveDrawMode::Lines);
    __verticesCounts.PushBack(2);
    __vb.PushBack(p1.x);
    __vb.PushBack(p1.y);
    __vb.PushBack(p1.z);
    __vb.PushBack(p2.x);
    __vb.PushBack(p2.y);
    __vb.PushBack(p2.z);
}

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color, float thickness, vec3 cameraPosition)
{
    __colors.PushBack(color);
    __drawMode.PushBack(PrimitiveDrawMode::Triangles);
    __verticesCounts.PushBack(6);

    vec3 mid = (p2 - p1) * 0.5f + p1;
    vec3 toCam = Normalize(cameraPosition - mid);
    vec3 lineTangent = Normalize(Cross(p2 - p1, toCam));
    vec3 pa = p1 + lineTangent * thickness * -0.5f;
    vec3 pb = p1 + lineTangent * thickness * 0.5f;
    vec3 pc = p2 + lineTangent * thickness * -0.5f;
    vec3 pd = p2 + lineTangent * thickness * 0.5f;
    __vb.PushBack(pa.x);
    __vb.PushBack(pa.y);
    __vb.PushBack(pa.z);
    __vb.PushBack(pb.x);
    __vb.PushBack(pb.y);
    __vb.PushBack(pb.z);
    __vb.PushBack(pc.x);
    __vb.PushBack(pc.y);
    __vb.PushBack(pc.z);
    __vb.PushBack(pd.x);
    __vb.PushBack(pd.y);
    __vb.PushBack(pd.z);
    __vb.PushBack(pc.x);
    __vb.PushBack(pc.y);
    __vb.PushBack(pc.z);
    __vb.PushBack(pb.x);
    __vb.PushBack(pb.y);
    __vb.PushBack(pb.z);
}

bool FindIntersectionBetweenPlaneAndLine(vec3 pointOnPlane, vec3 normalOfPlane, vec3 pointOnLine, vec3 directionOfLine, vec3 *intersectionPoint)
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









