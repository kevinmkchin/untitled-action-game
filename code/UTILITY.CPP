

vec3 ScreenPointToWorldRay(ivec2 screenspaceCoords);
vec3 ScreenPointToWorldPoint(ivec2 screenspaceCoords, float z_NDC);
vec3 WorldPointToScreenPoint(vec3 worldPosition);


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



#pragma region PRIMITIVES_AND_HANDLES_DRAWING

static u32 PRIMITIVES_VAO = 0;
static u32 PRIMITIVES_VBO = 0;
static GPUShader PRIMITIVES_SHADER;
static const char* PRIMITIVES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "void main() {\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "}\n";
static const char* PRIMITIVES_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform vec4 primitivesColor;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "void main() {\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(primitivesColor.xyz, 0.5);\n" // TODO let opacity be defined in immediate mode
    "    } else {\n"
    "        colour = primitivesColor;\n"
    "    }\n"
    "}\n";
enum class PrimitiveDrawMode
{
    Triangles,
    Lines
};
static NiceArray<vec4, 128> PRIMITIVES_COLORS;
static NiceArray<PrimitiveDrawMode, 128> PRIMITIVES_DRAWMODES;
static NiceArray<u16, 128> PRIMITIVES_VERTEXCOUNTS; // vertex buffer sizes in number of floats
static NiceArray<float, 256 * 128> PRIMITIVES_VB;

static GPUFrameBuffer mousePickingRenderTarget;
static u32 HANDLES_VAO = 0;
static u32 HANDLES_VBO = 0;
static GPUShader HANDLES_SHADER;
static const char* HANDLES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec3 idRGB;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out vec3 handlesIdRGB;\n"
    "void main() {\n"
    "    handlesIdRGB = idRGB;\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "}\n";
static const char* HANDLES_SHADER_FS =
    "#version 330 core\n"
    "in vec3 handlesIdRGB;\n"
    "out vec4 colour;\n"
    "void main() {\n"
    "    colour = vec4(handlesIdRGB, 1.0);\n"
    "}\n";
static NiceArray<float, 256 * 128> HANDLES_VB;

void InitPrimitivesAndHandlesSystems()
{
    glGenVertexArrays(1, &PRIMITIVES_VAO);
    glBindVertexArray(PRIMITIVES_VAO);
    glGenBuffers(1, &PRIMITIVES_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, PRIMITIVES_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(PRIMITIVES_SHADER, PRIMITIVES_SHADER_VS, PRIMITIVES_SHADER_FS);

    glGenVertexArrays(1, &HANDLES_VAO);
    glBindVertexArray(HANDLES_VAO);
    glGenBuffers(1, &HANDLES_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, HANDLES_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(HANDLES_SHADER, HANDLES_SHADER_VS, HANDLES_SHADER_FS);

    mousePickingRenderTarget.width = 32;
    mousePickingRenderTarget.height = 32;
    CreateGPUFrameBuffer(&mousePickingRenderTarget);
}

void PrimitiveDrawAll(const mat4 *projectionMatrix, const mat4 *viewMatrix, GLuint sceneDepthTextureId, vec2 framebufferSize)
{
    if (PRIMITIVES_COLORS.count == 0) return;

    UseShader(PRIMITIVES_SHADER);
    GLBindMatrix4fv(PRIMITIVES_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
    GLBindMatrix4fv(PRIMITIVES_SHADER, "viewMatrix", 1, viewMatrix->ptr());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
    GLBind1i(PRIMITIVES_SHADER, "sceneDepthTexture", 0);

    GLBind2f(PRIMITIVES_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

    u32 __vbIndexer = 0;
    for (int i = 0; i < PRIMITIVES_COLORS.count; ++i)
    {
        vec4 color = PRIMITIVES_COLORS.At(i);
        u16 vertexCount = PRIMITIVES_VERTEXCOUNTS.At(i);
        float* vb = PRIMITIVES_VB.data + __vbIndexer;
        __vbIndexer += vertexCount * 3;

        GLBind4f(PRIMITIVES_SHADER, "primitivesColor", color.x, color.y, color.z, color.w);
        glBindVertexArray(PRIMITIVES_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIMITIVES_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)4 * vertexCount * 3, vb, GL_DYNAMIC_DRAW);
        if (PRIMITIVES_DRAWMODES.At(i) == PrimitiveDrawMode::Triangles)
        {
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }
        else if (PRIMITIVES_DRAWMODES.At(i) == PrimitiveDrawMode::Lines)
        {
            glDrawArrays(GL_LINES, 0, vertexCount);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    PRIMITIVES_COLORS.ResetCount();
    PRIMITIVES_DRAWMODES.ResetCount();
    PRIMITIVES_VERTEXCOUNTS.ResetCount();
    PRIMITIVES_VB.ResetCount();
}

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color)
{
    PRIMITIVES_COLORS.PushBack(color);
    PRIMITIVES_DRAWMODES.PushBack(PrimitiveDrawMode::Triangles);
    PRIMITIVES_VERTEXCOUNTS.PushBack(12 * 3);

    vec3 tangent = Normalize(Cross(normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, normal));
    quat q = quat(GM_DEG2RAD * 30.f, normal);
    for(int i = 0; i < 12; ++i)
    {
        vec3 a = center;
        vec3 b = center + tangent * radius;
        tangent = RotateVector(tangent, q);
        vec3 c = center + tangent * radius;

        PRIMITIVES_VB.PushBack(a.x);
        PRIMITIVES_VB.PushBack(a.y);
        PRIMITIVES_VB.PushBack(a.z);
        PRIMITIVES_VB.PushBack(b.x);
        PRIMITIVES_VB.PushBack(b.y);
        PRIMITIVES_VB.PushBack(b.z);
        PRIMITIVES_VB.PushBack(c.x);
        PRIMITIVES_VB.PushBack(c.y);
        PRIMITIVES_VB.PushBack(c.z);
    }
}

void PrimitiveDrawSolidDisc(vec3 center, vec3 normal, float radius)
{
    PrimitiveDrawSolidDisc(center, normal, radius, vec4(RGB255TO1(248, 230, 60), 1.f));
}

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color)
{
    PRIMITIVES_COLORS.PushBack(color);
    PRIMITIVES_DRAWMODES.PushBack(PrimitiveDrawMode::Lines);
    PRIMITIVES_VERTEXCOUNTS.PushBack(2);
    PRIMITIVES_VB.PushBack(p1.x);
    PRIMITIVES_VB.PushBack(p1.y);
    PRIMITIVES_VB.PushBack(p1.z);
    PRIMITIVES_VB.PushBack(p2.x);
    PRIMITIVES_VB.PushBack(p2.y);
    PRIMITIVES_VB.PushBack(p2.z);
}

void PrimitiveDrawLine(vec3 p1, vec3 p2, vec4 color, float thickness, vec3 cameraPosition)
{
    PRIMITIVES_COLORS.PushBack(color);
    PRIMITIVES_DRAWMODES.PushBack(PrimitiveDrawMode::Triangles);
    PRIMITIVES_VERTEXCOUNTS.PushBack(6);

    vec3 mid = (p2 - p1) * 0.5f + p1;
    vec3 toCam = Normalize(cameraPosition - mid);
    vec3 lineTangent = Normalize(Cross(p2 - p1, toCam));
    vec3 pa = p1 + lineTangent * thickness * -0.5f;
    vec3 pb = p1 + lineTangent * thickness * 0.5f;
    vec3 pc = p2 + lineTangent * thickness * -0.5f;
    vec3 pd = p2 + lineTangent * thickness * 0.5f;
    PRIMITIVES_VB.PushBack(pa.x);
    PRIMITIVES_VB.PushBack(pa.y);
    PRIMITIVES_VB.PushBack(pa.z);
    PRIMITIVES_VB.PushBack(pb.x);
    PRIMITIVES_VB.PushBack(pb.y);
    PRIMITIVES_VB.PushBack(pb.z);
    PRIMITIVES_VB.PushBack(pc.x);
    PRIMITIVES_VB.PushBack(pc.y);
    PRIMITIVES_VB.PushBack(pc.z);
    PRIMITIVES_VB.PushBack(pd.x);
    PRIMITIVES_VB.PushBack(pd.y);
    PRIMITIVES_VB.PushBack(pd.z);
    PRIMITIVES_VB.PushBack(pc.x);
    PRIMITIVES_VB.PushBack(pc.y);
    PRIMITIVES_VB.PushBack(pc.z);
    PRIMITIVES_VB.PushBack(pb.x);
    PRIMITIVES_VB.PushBack(pb.y);
    PRIMITIVES_VB.PushBack(pb.z);
}


vec3 HandleIdToRGB(u32 id)
{
    float r = (float) ((id & 0x000000FF) >> 0);
    float g = (float) ((id & 0x0000FF00) >> 8);
    float b = (float) ((id & 0x00FF0000) >> 16);
    return vec3(r,g,b)/255.f;
}

void DoDiscHandle(u32 id, vec3 worldpos, vec3 normal, float radius)
{
    vec3 idrgb = HandleIdToRGB(id);

    vec3 tangent = Normalize(Cross(normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, normal));
    quat q = quat(GM_DEG2RAD * 30.f, normal);
    for(int i = 0; i < 12; ++i)
    {
        vec3 a = worldpos;
        vec3 b = worldpos + tangent * radius;
        tangent = RotateVector(tangent, q);
        vec3 c = worldpos + tangent * radius;

        HANDLES_VB.PushBack(a.x);
        HANDLES_VB.PushBack(a.y);
        HANDLES_VB.PushBack(a.z);
        HANDLES_VB.PushBack(idrgb.x);
        HANDLES_VB.PushBack(idrgb.y);
        HANDLES_VB.PushBack(idrgb.z);

        HANDLES_VB.PushBack(b.x);
        HANDLES_VB.PushBack(b.y);
        HANDLES_VB.PushBack(b.z);
        HANDLES_VB.PushBack(idrgb.x);
        HANDLES_VB.PushBack(idrgb.y);
        HANDLES_VB.PushBack(idrgb.z);

        HANDLES_VB.PushBack(c.x);
        HANDLES_VB.PushBack(c.y);
        HANDLES_VB.PushBack(c.z);
        HANDLES_VB.PushBack(idrgb.x);
        HANDLES_VB.PushBack(idrgb.y);
        HANDLES_VB.PushBack(idrgb.z);
    }
}

// draw all handles, return id of clicked handle
u32 FlushHandles(ivec2 clickat, mat4 viewMatrix, mat4 perspectiveMatrix)
{
    if (HANDLES_VB.count == 0) return 0;

//    vec3 mouseClickWorldPos = ScreenPointToWorldPoint(clickat, 0.f);
//    vec3 mouseCameraDirection = ScreenPointToWorldRay(clickat);
//    vec3 mouseCameraRight = Normalize(Cross(mouseCameraDirection, GM_UP_VECTOR));
//    vec3 mouseCameraUp = Normalize(Cross(mouseCameraRight, mouseCameraDirection));
//    mat4 handlesViewMatrix = ViewMatrixLookAt(mouseClickWorldPos, mouseClickWorldPos+mouseCameraDirection, mouseCameraUp);
//    mat4 handlesProjMatrix = ProjectionMatrixOrthographic(-2, 2,
//                                                          -2, 2,
//                                                          -32000.f, 32000.f);
//    mat4 handlesProjMatrix = ProjectionMatrixPerspective(1.f, 1.f, 0.1f, 32000.f);
    mat4 proj = perspectiveMatrix;
    proj = ScaleMatrix(1920.f/4.f, 1080.f/4.f, 1.f) * proj; // scales down frustrum
    const float offsetX = (clickat.x - 1920.f*0.5f) / (4.f * 0.5f);
    const float offsetY = (clickat.y - 1080.f*0.5f) / (4.f * 0.5f);
    proj[2][0] += offsetX;
    proj[2][1] -= offsetY; // then the offset moves it around like light coming through a moving hole on the near plane

    glBindFramebuffer(GL_FRAMEBUFFER, mousePickingRenderTarget.fbo);
    glViewport(0,0,mousePickingRenderTarget.width,mousePickingRenderTarget.height);
    glClearColor(0.f,0.f,0.f,1.f); // clear with rgb(id 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    UseShader(HANDLES_SHADER);

    GLBindMatrix4fv(HANDLES_SHADER, "projectionMatrix", 1, proj.ptr());
    GLBindMatrix4fv(HANDLES_SHADER, "viewMatrix", 1, viewMatrix.ptr());

    glBindVertexArray(HANDLES_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, HANDLES_VBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float)*HANDLES_VB.count, HANDLES_VB.data, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, HANDLES_VB.count / 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    HANDLES_VB.ResetCount();

    glFlush();
    glFinish();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    u8 pickedRGBData[4];
    glReadPixels(mousePickingRenderTarget.width/2, mousePickingRenderTarget.height/2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pickedRGBData);
    u32 pickedHandleId = pickedRGBData[0] + pickedRGBData[1] * 256 + pickedRGBData[2] * 256 * 256;
    return pickedHandleId;
}

#pragma endregion



#pragma region MATH_HELPERS

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

#pragma endregion








