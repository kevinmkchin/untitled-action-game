

static u32 PRIM_VERTEX_POS_AND_COLOR_VAO;
static u32 PRIM_VERTEX_POS_AND_COLOR_VBO;

static NiceArray<float, 256000> PRIMITIVE_TRIS_VB;
static GPUShader PRIMITIVES_TRIS_SHADER;
static const char* PRIMITIVES_TRIS_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec4 col;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out vec4 baseColour;\n"
    "void main() {\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "    baseColour = col;\n"
    "}\n";
static const char* PRIMITIVES_TRIS_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in vec4 baseColour;\n"
    "void main() {\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(baseColour.xyz, baseColour.w * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(baseColour.xyz, baseColour.w);\n"
    "    }\n"
    "}\n";

static u32 PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO;
static u32 PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO;

static NiceArray<float, 256000> PRIMITIVE_FATLINES_VB;
static GPUShader FATLINES_SHADER;
static const char* FATLINES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec4 col;\n"
    "layout (location = 2) in float width;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out float gs_distance_from_eye;\n"
    "out vec4 gs_baseColour;\n"
    "out float gs_lineWidth;\n"
    "void main() {\n"
    "    vec4 viewspace_pos = viewMatrix * vec4(pos, 1.0);\n"
    "    gl_Position = projectionMatrix * viewspace_pos;\n"
    "    gs_distance_from_eye = -viewspace_pos.z;\n"
    "    gs_baseColour = col;\n"
    "    gs_lineWidth = width;\n"
    "}\n";
static const char* FATLINES_SHADER_GS = 
    "#version 330 core\n"
    "layout(lines) in;\n"
    "layout(triangle_strip, max_vertices = 4) out;\n"
    "\n"
    "in float gs_distance_from_eye[];\n"
    "in vec4 gs_baseColour[];\n"
    "in float gs_lineWidth[];\n"
    "out float distance_from_eye;\n"
    "out vec4 baseColour;\n"
    "void main() {\n"
    "    vec4 p0 = gl_in[0].gl_Position;"
    "    vec4 p1 = gl_in[1].gl_Position;"
    "\n"
    "    vec2 direction = normalize(p1.xy - p0.xy);\n"
    "    vec2 offset = vec2(-direction.y, direction.x) * gs_lineWidth[0] * 0.5;\n"
    "\n"
    "    vec2 offset0 = offset * p0.w;  // Apply perspective scaling to offset for p0\n"
    "    vec2 offset1 = offset * p1.w;  // Apply perspective scaling to offset for p1\n"
    "    // Emit four vertices for the quad\n"
    "    distance_from_eye = gs_distance_from_eye[0];\n"
    "    baseColour = gs_baseColour[0];\n"
    "    gl_Position = p0 + vec4(offset0, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "    gl_Position = p0 - vec4(offset0, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    distance_from_eye = gs_distance_from_eye[1];\n"
    "    baseColour = gs_baseColour[1];\n"
    "    gl_Position = p1 + vec4(offset1, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "    gl_Position = p1 - vec4(offset1, 0.0, 0.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    EndPrimitive();\n"
    "}";

static const char* FATLINES_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in float distance_from_eye;\n"
    "in vec4 baseColour;\n"
    "void main() {\n"
    "    float fadeMult = clamp(exp(-pow((distance_from_eye * 0.0001), 5)), 0.0, 1.0);\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult);\n"
    "    }\n"
    "}\n";

static NiceArray<float, 256000> PRIMITIVE_LINES_VB;
static GPUShader LINES_SHADER;
static const char* LINES_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec4 col;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out float distance_from_eye;\n"
    "out vec4 baseColour;\n"
    "void main() {\n"
    "    vec4 viewspace_pos = viewMatrix * vec4(pos, 1.0);\n"
    "    gl_Position = projectionMatrix * viewspace_pos;\n"
    "    distance_from_eye = -viewspace_pos.z;\n"
    "    baseColour = col;\n"
    "}\n";
static const char* LINES_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in float distance_from_eye;\n"
    "in vec4 baseColour;\n"
    "void main() {\n"
    "    float fadeMult = clamp(exp(-pow((distance_from_eye * 0.0001), 5)), 0.0, 1.0);\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(baseColour.xyz, baseColour.w * fadeMult);\n"
    "    }\n"
    "}\n";
static bool DrawAxisLines = false;

static GPUFrameBuffer mousePickingRenderTarget;
static u32 HANDLES_VAO = 0;
static u32 HANDLES_VBO = 0;
static NiceArray<float, 256 * 128> HANDLES_VB;

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

static GPUMesh PICKABLE_BILLBOARDS_MESH;
static dynamic_array<float> PICKABLE_BILLBOARDS_VB; 
static GPUShader PICKABLE_BILLBOARDS_SHADER;
static const char *PICKABLE_BILLBOARDS_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "layout (location = 1) in vec2 uv;\n"
    "layout (location = 2) in vec3 idRGB;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "out vec2 billboardUV;\n"
    "out vec3 handlesIdRGB;\n"
    "void main() {\n"
    "    billboardUV = uv;\n"
    "    handlesIdRGB = idRGB;\n"
    "    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);\n"
    "}\n";
static const char *PICKABLE_BILLBOARDS_SHADER_FS =
    "#version 330 core\n"
    "in vec2 billboardUV;\n"
    "in vec3 handlesIdRGB;\n"
    "uniform sampler2D BillboardTexture;\n"
    "uniform int UseColorIds;\n"
    "out vec4 colour;\n"
    "void main() {\n"
    "    vec4 TexColor = texture(BillboardTexture, billboardUV);\n"
    "    if (UseColorIds == 0) { \n"
    "        colour = TexColor; \n"
    "    } else { \n"
    "        colour = vec4(handlesIdRGB, ceil(TexColor.a)); \n"
    "    }\n"
    "}\n";

static u32 GRID_MESH_VAO;
static u32 GRID_MESH_VBO;
static GPUShader GRID_MESH_SHADER;
static const char* GRID_MESH_SHADER_VS =
    "#version 330 core\n"
    "layout (location = 0) in vec3 pos;\n"
    "uniform mat4 projectionMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "uniform mat4 transformMatrix;\n"
    "out float distance_from_eye;\n"
    "void main() {\n"
    "    vec4 viewspace_pos = viewMatrix * transformMatrix * vec4(pos, 1.0);\n"
    "    gl_Position = projectionMatrix * viewspace_pos;\n"
    "    distance_from_eye = -viewspace_pos.z;\n"
    "}\n";
static const char* GRID_MESH_SHADER_FS =
    "#version 330 core\n"
    "out vec4 colour;\n"
    "uniform vec2 framebufferSize;\n"
    "uniform sampler2D sceneDepthTexture;\n"
    "uniform float occludedOpacity;\n"
    "in float distance_from_eye;\n"
    "void main() {\n"
    "    float fadeMult = clamp(exp(-pow((distance_from_eye * 0.0001), 5)), 0.0, 1.0);\n"
    "    vec2 depthUV = gl_FragCoord.xy / framebufferSize;\n"
    "    float sceneDepth = texture(sceneDepthTexture, depthUV).r;\n"
    "    if (sceneDepth < gl_FragCoord.z) {  \n"
    "        colour = vec4(0.6,0.6,0.6, 0.25 * fadeMult * occludedOpacity);\n"
    "    } else {\n"
    "        colour = vec4(0.6,0.6,0.6, 0.25 * fadeMult);\n"
    "    }\n"
    "}\n";

support_renderer_t SupportRenderer;

void support_renderer_t::Initialize()
{
    glGenVertexArrays(1, &PRIM_VERTEX_POS_AND_COLOR_VAO);
    glBindVertexArray(PRIM_VERTEX_POS_AND_COLOR_VAO);
    glGenBuffers(1, &PRIM_VERTEX_POS_AND_COLOR_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_AND_COLOR_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*7, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float)*7, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(PRIMITIVES_TRIS_SHADER, PRIMITIVES_TRIS_SHADER_VS, PRIMITIVES_TRIS_SHADER_FS);
    GLCreateShaderProgram(LINES_SHADER, LINES_SHADER_VS, LINES_SHADER_FS);

    glGenVertexArrays(1, &PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO);
    glBindVertexArray(PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO);
    glGenBuffers(1, &PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*8, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(sizeof(float)*7));
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(FATLINES_SHADER, FATLINES_SHADER_VS, FATLINES_SHADER_GS, FATLINES_SHADER_FS);

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

    CreateGPUMesh(&PICKABLE_BILLBOARDS_MESH, 3, 2, 3, GL_DYNAMIC_DRAW);
    PICKABLE_BILLBOARDS_VB.setcap(384);
    GLCreateShaderProgram(PICKABLE_BILLBOARDS_SHADER, PICKABLE_BILLBOARDS_SHADER_VS, PICKABLE_BILLBOARDS_SHADER_FS);

    vec3 gridmeshdata[4001*4];
    for (int i = -2000; i <= 2000; ++i)
    {
        gridmeshdata[(i+2000)*4+0] = vec3((float)i, 0.f, -2000.f);
        gridmeshdata[(i+2000)*4+1] = vec3((float)i, 0.f, 2000.f);
        gridmeshdata[(i+2000)*4+2] = vec3(-2000.f, 0.f, (float)i);
        gridmeshdata[(i+2000)*4+3] = vec3(2000.f, 0.f, (float)i);
    }
    glGenVertexArrays(1, &GRID_MESH_VAO);
    glBindVertexArray(GRID_MESH_VAO);
    glGenBuffers(1, &GRID_MESH_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, GRID_MESH_VBO);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(4001*4*sizeof(vec3)), (void*)gridmeshdata, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    GLCreateShaderProgram(GRID_MESH_SHADER, GRID_MESH_SHADER_VS, GRID_MESH_SHADER_FS);

    mousePickingRenderTarget.width = 4;
    mousePickingRenderTarget.height = 4;
    CreateGPUFrameBuffer(&mousePickingRenderTarget);
}

void support_renderer_t::DrawGrid(float scale, mat3 rotation, vec3 translation, const mat4 *projectionMatrix, const mat4 *viewMatrix, GLuint sceneDepthTextureId, vec2 framebufferSize)
{
    mat4 transformMatrix = TranslationMatrix(translation) * mat4(rotation) * ScaleMatrix(scale, scale, scale);

    UseShader(GRID_MESH_SHADER);
    GLBindMatrix4fv(GRID_MESH_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
    GLBindMatrix4fv(GRID_MESH_SHADER, "viewMatrix", 1, viewMatrix->ptr());
    GLBindMatrix4fv(GRID_MESH_SHADER, "transformMatrix", 1, transformMatrix.ptr());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
    GLBind1i(GRID_MESH_SHADER, "sceneDepthTexture", 0);
    GLBind2f(GRID_MESH_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

    GLBind1f(GRID_MESH_SHADER, "occludedOpacity", 0.0f);
    glBindVertexArray(GRID_MESH_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, GRID_MESH_VBO);
    glDrawArrays(GL_LINES, 0, 4001*4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void support_renderer_t::FlushPrimitives(const mat4 *projectionMatrix, const mat4 *viewMatrix, GLuint sceneDepthTextureId, vec2 framebufferSize)
{
    if (PRIMITIVE_LINES_VB.count > 0)
    {
        UseShader(LINES_SHADER);
        GLBindMatrix4fv(LINES_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
        GLBindMatrix4fv(LINES_SHADER, "viewMatrix", 1, viewMatrix->ptr());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
        GLBind1i(LINES_SHADER, "sceneDepthTexture", 0);
        GLBind2f(LINES_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

        GLBind1f(LINES_SHADER, "occludedOpacity", 0.0f);
        glBindVertexArray(PRIM_VERTEX_POS_AND_COLOR_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_AND_COLOR_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * PRIMITIVE_LINES_VB.count, PRIMITIVE_LINES_VB.data,
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, PRIMITIVE_LINES_VB.count / 7);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        PRIMITIVE_LINES_VB.ResetCount();
    }

    if (PRIMITIVE_FATLINES_VB.count > 0)
    {
        UseShader(FATLINES_SHADER);
        GLBindMatrix4fv(FATLINES_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
        GLBindMatrix4fv(FATLINES_SHADER, "viewMatrix", 1, viewMatrix->ptr());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
        GLBind1i(FATLINES_SHADER, "sceneDepthTexture", 0);
        GLBind2f(FATLINES_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

        GLBind1f(FATLINES_SHADER, "occludedOpacity", 0.33f);
        glBindVertexArray(PRIM_VERTEX_POS_COLOR_LINEWIDTH_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_COLOR_LINEWIDTH_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * PRIMITIVE_FATLINES_VB.count, PRIMITIVE_FATLINES_VB.data,
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, PRIMITIVE_FATLINES_VB.count / 8);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        PRIMITIVE_FATLINES_VB.ResetCount();
    }

    if (PRIMITIVE_TRIS_VB.count > 0)
    {
        UseShader(PRIMITIVES_TRIS_SHADER);
        GLBindMatrix4fv(PRIMITIVES_TRIS_SHADER, "projectionMatrix", 1, projectionMatrix->ptr());
        GLBindMatrix4fv(PRIMITIVES_TRIS_SHADER, "viewMatrix", 1, viewMatrix->ptr());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTextureId);
        GLBind1i(PRIMITIVES_TRIS_SHADER, "sceneDepthTexture", 0);
        GLBind2f(PRIMITIVES_TRIS_SHADER, "framebufferSize", framebufferSize.x, framebufferSize.y);

        GLBind1f(PRIMITIVES_TRIS_SHADER, "occludedOpacity", 0.5f);
        glBindVertexArray(PRIM_VERTEX_POS_AND_COLOR_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, PRIM_VERTEX_POS_AND_COLOR_VBO);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * PRIMITIVE_TRIS_VB.count, PRIMITIVE_TRIS_VB.data,
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, PRIMITIVE_TRIS_VB.count / 7);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        PRIMITIVE_TRIS_VB.ResetCount();
    }
}

void support_renderer_t::DrawSolidDisc(vec3 center, vec3 normal, float radius, vec4 color)
{
    vec3 tangent = Normalize(Cross(normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, normal));
    quat q = quat(GM_DEG2RAD * 30.f, normal);
    for(int i = 0; i < 12; ++i)
    {
        vec3 a = center;
        vec3 b = center + tangent * radius;
        tangent = RotateVector(tangent, q);
        vec3 c = center + tangent * radius;

        PRIMITIVE_TRIS_VB.PushBack(a.x);
        PRIMITIVE_TRIS_VB.PushBack(a.y);
        PRIMITIVE_TRIS_VB.PushBack(a.z);
        PRIMITIVE_TRIS_VB.PushBack(color.x);
        PRIMITIVE_TRIS_VB.PushBack(color.y);
        PRIMITIVE_TRIS_VB.PushBack(color.z);
        PRIMITIVE_TRIS_VB.PushBack(color.w);

        PRIMITIVE_TRIS_VB.PushBack(b.x);
        PRIMITIVE_TRIS_VB.PushBack(b.y);
        PRIMITIVE_TRIS_VB.PushBack(b.z);
        PRIMITIVE_TRIS_VB.PushBack(color.x);
        PRIMITIVE_TRIS_VB.PushBack(color.y);
        PRIMITIVE_TRIS_VB.PushBack(color.z);
        PRIMITIVE_TRIS_VB.PushBack(color.w);

        PRIMITIVE_TRIS_VB.PushBack(c.x);
        PRIMITIVE_TRIS_VB.PushBack(c.y);
        PRIMITIVE_TRIS_VB.PushBack(c.z);
        PRIMITIVE_TRIS_VB.PushBack(color.x);
        PRIMITIVE_TRIS_VB.PushBack(color.y);
        PRIMITIVE_TRIS_VB.PushBack(color.z);
        PRIMITIVE_TRIS_VB.PushBack(color.w);
    }
}

void support_renderer_t::DrawSolidDisc(vec3 center, vec3 normal, float radius)
{
    DrawSolidDisc(center, normal, radius, vec4(RGB255TO1(248, 230, 60), 1.f));
}

void support_renderer_t::DrawLine(vec3 p1, vec3 p2, vec4 color)
{
    PRIMITIVE_LINES_VB.PushBack(p1.x);
    PRIMITIVE_LINES_VB.PushBack(p1.y);
    PRIMITIVE_LINES_VB.PushBack(p1.z);
    PRIMITIVE_LINES_VB.PushBack(color.x);
    PRIMITIVE_LINES_VB.PushBack(color.y);
    PRIMITIVE_LINES_VB.PushBack(color.z);
    PRIMITIVE_LINES_VB.PushBack(color.w);

    PRIMITIVE_LINES_VB.PushBack(p2.x);
    PRIMITIVE_LINES_VB.PushBack(p2.y);
    PRIMITIVE_LINES_VB.PushBack(p2.z);
    PRIMITIVE_LINES_VB.PushBack(color.x);
    PRIMITIVE_LINES_VB.PushBack(color.y);
    PRIMITIVE_LINES_VB.PushBack(color.z);
    PRIMITIVE_LINES_VB.PushBack(color.w);
}

void support_renderer_t::DrawLine(vec3 p1, vec3 p2, vec4 color, float thickness)
{
    PRIMITIVE_FATLINES_VB.PushBack(p1.x);
    PRIMITIVE_FATLINES_VB.PushBack(p1.y);
    PRIMITIVE_FATLINES_VB.PushBack(p1.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.x);
    PRIMITIVE_FATLINES_VB.PushBack(color.y);
    PRIMITIVE_FATLINES_VB.PushBack(color.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.w);
    PRIMITIVE_FATLINES_VB.PushBack(thickness*0.0025f);

    PRIMITIVE_FATLINES_VB.PushBack(p2.x);
    PRIMITIVE_FATLINES_VB.PushBack(p2.y);
    PRIMITIVE_FATLINES_VB.PushBack(p2.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.x);
    PRIMITIVE_FATLINES_VB.PushBack(color.y);
    PRIMITIVE_FATLINES_VB.PushBack(color.z);
    PRIMITIVE_FATLINES_VB.PushBack(color.w);
    PRIMITIVE_FATLINES_VB.PushBack(thickness*0.0025f);
}


vec3 support_renderer_t::HandleIdToRGB(u32 id)
{
    float r = (float) ((id & 0x000000FF) >> 0);
    float g = (float) ((id & 0x0000FF00) >> 8);
    float b = (float) ((id & 0x00FF0000) >> 16);
    return vec3(r,g,b)/255.f;
}

void support_renderer_t::DoDiscHandle(u32 id, vec3 worldpos, vec3 normal, float radius)
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

void support_renderer_t::DoPickableBillboard(u32 Id, vec3 WorldPos, vec3 Normal, billboard_t Billboard)
{
    // TODO batch/atlas this shit

    vec3 idrgb = HandleIdToRGB(Id);
    vec3 RightTangent = Normalize(Cross(Normal == GM_UP_VECTOR ? GM_RIGHT_VECTOR : GM_UP_VECTOR, Normal));
    vec3 UpTangent = Normalize(Cross(Normal, RightTangent));

    RightTangent *= Billboard.Sz;
    UpTangent *= Billboard.Sz;

    vec3 BL = WorldPos - UpTangent - RightTangent;
    vec3 BR = WorldPos - UpTangent + RightTangent;
    vec3 TL = WorldPos + UpTangent - RightTangent;
    vec3 TR = WorldPos + UpTangent + RightTangent;

    // BL BR TL
    // TL BR TR
    float BillboardVertsTemp[48] = {
        BL.x, BL.y, BL.z, 0.f, 0.f, idrgb.x, idrgb.y, idrgb.z, 
        BR.x, BR.y, BR.z, 1.f, 0.f, idrgb.x, idrgb.y, idrgb.z,
        TL.x, TL.y, TL.z, 0.f, 1.f, idrgb.x, idrgb.y, idrgb.z,
        TL.x, TL.y, TL.z, 0.f, 1.f, idrgb.x, idrgb.y, idrgb.z,
        BR.x, BR.y, BR.z, 1.f, 0.f, idrgb.x, idrgb.y, idrgb.z,
        TR.x, TR.y, TR.z, 1.f, 1.f, idrgb.x, idrgb.y, idrgb.z,
    };

    float *BillboardVerts = PICKABLE_BILLBOARDS_VB.addnptr(48);

    memmove(BillboardVerts, BillboardVertsTemp, 48*sizeof(float));
}

void support_renderer_t::AddTrianglesToPickableHandles(float *vertices, int count)
{
    if (HANDLES_VB.count + count > HANDLES_VB.capacity)
    {
        LogError("AddTrianglesToPickableHandles exceeds allocated HANDLES_VB buffer.");
        return;
    }

    memcpy(HANDLES_VB.data + HANDLES_VB.count, vertices, count * sizeof(float));
    HANDLES_VB.count += count;
}

void support_renderer_t::DrawHandlesVertexArray_GL(float *vertexArrayData, u32 vertexArrayDataCount, 
    float *projectionMat, float *viewMat)
{
    UseShader(HANDLES_SHADER);

    GLBindMatrix4fv(HANDLES_SHADER, "projectionMatrix", 1, projectionMat);
    GLBindMatrix4fv(HANDLES_SHADER, "viewMatrix", 1, viewMat);

    glBindVertexArray(HANDLES_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, HANDLES_VBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float)*vertexArrayDataCount, vertexArrayData, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, vertexArrayDataCount / 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void support_renderer_t::DrawPickableBillboards_GL(float *ProjectionMat, float *ViewMat, bool UseColorIds)
{
    // pickable billboards can have transparent area and non-transparent area.
    // so if the used texture gives alpha of 0 at a point
    // ok so just draw the same billboards, but use a shader that overwrites texture
    // sample with handle id

    // because these can be transparent, I need to sort them
    // then assemble the vertex buffer here instead

    UseShader(PICKABLE_BILLBOARDS_SHADER);

    GLBindMatrix4fv(PICKABLE_BILLBOARDS_SHADER, "projectionMatrix", 1, ProjectionMat);
    GLBindMatrix4fv(PICKABLE_BILLBOARDS_SHADER, "viewMatrix", 1, ViewMat);
    GLBind1i(PICKABLE_BILLBOARDS_SHADER, "UseColorIds", UseColorIds);

    RebindGPUMesh(
        &PICKABLE_BILLBOARDS_MESH,
        PICKABLE_BILLBOARDS_VB.lenu() * sizeof(float),
        PICKABLE_BILLBOARDS_VB.data);
    RenderGPUMesh(
        PICKABLE_BILLBOARDS_MESH.idVAO,
        PICKABLE_BILLBOARDS_MESH.idVBO,
        PICKABLE_BILLBOARDS_MESH.vertexCount,
        &Assets.PickableBillboardsAtlas);
}

void support_renderer_t::ClearPickableBillboards()
{
    PICKABLE_BILLBOARDS_VB.setlen(0);
}

u32 support_renderer_t::FlushHandles(ivec2 clickat, const GPUFrameBuffer activeSceneTarget,
                 const mat4& activeViewMatrix, const mat4& activeProjectionMatrix, bool orthographic)
{
    if (HANDLES_VB.count == 0 && PICKABLE_BILLBOARDS_VB.lenu() == 0) return 0;

    const float sceneResolutionW = (float)activeSceneTarget.width;
    const float sceneResolutionH = (float)activeSceneTarget.height;
    const float mousePickTargetW = (float)mousePickingRenderTarget.width;
    const float mousePickTargetH = (float)mousePickingRenderTarget.width;

    mat4 scaledDownFrustum = ScaleMatrix(sceneResolutionW/mousePickTargetW, sceneResolutionH/mousePickTargetH, 1.f) * activeProjectionMatrix;
    const float offsetX = ((float)clickat.x - sceneResolutionW*0.5f) / (mousePickTargetW * 0.5f);
    const float offsetY = ((float)clickat.y - sceneResolutionH*0.5f) / (mousePickTargetH * 0.5f);
    if (orthographic)
    {
        // NOTE(Kevin): untested! haven't tried orthographic picking yet! prob needs some debugging
        scaledDownFrustum[3][0] += -offsetX;
        scaledDownFrustum[3][1] -= -offsetY;
    }
    else
    {
        // then the offset moves it around like light coming through a moving hole on the near plane
        scaledDownFrustum[2][0] += offsetX;
        scaledDownFrustum[2][1] -= offsetY;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, mousePickingRenderTarget.fbo);
    glViewport(0,0,(int)mousePickTargetW,(int)mousePickTargetH);
    glClearColor(0.f,0.f,0.f,1.f); // clear with rgb(id 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    if (HANDLES_VB.count > 0)
        DrawHandlesVertexArray_GL(HANDLES_VB.data, HANDLES_VB.count,
            scaledDownFrustum.ptr(), activeViewMatrix.ptr());
    if (PICKABLE_BILLBOARDS_VB.lenu() > 0)
        DrawPickableBillboards_GL(scaledDownFrustum.ptr(), activeViewMatrix.ptr(), true);

    HANDLES_VB.ResetCount();
    ClearPickableBillboards();

    glFlush();
    glFinish();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    u8 pickedRGBData[4];
    glReadPixels((int)mousePickTargetW/2, (int)mousePickTargetH/2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pickedRGBData);
    u32 pickedHandleId = pickedRGBData[0] + pickedRGBData[1] * 256 + pickedRGBData[2] * 256 * 256;
    return pickedHandleId;
}
