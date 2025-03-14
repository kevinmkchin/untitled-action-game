
const char* __editor_scene_shader_vs =
    "#version 330\n"
    "\n"
    "layout (location = 0) in vec3 vertex_pos;\n"
    "layout (location = 1) in vec2 vertex_uv;\n"
    "layout (location = 2) in vec3 vertex_normal;\n"
    "\n"
    "out vec2 uv;\n"
    "out vec3 LIGHT;\n"
    "\n"
    "uniform mat4 modelMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "uniform mat4 projMatrix;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    uv = vertex_uv;\n"
    "\n"
    "    vec4 vertInClipSpace = projMatrix * viewMatrix * modelMatrix * vec4(vertex_pos, 1.0);\n"
    "    gl_Position = vertInClipSpace;\n"
    "\n"
    "    const vec4 sunlightWCS = vec4(0.5, -0.7, -0.8, 0.0);\n"
    "    vec3 normalWCS = transpose(inverse(mat3(modelMatrix))) * vertex_normal;\n"
    "    \n"
    "    float ambientIntensity = 0.6;\n"
    "    float diffuseIntensity = max(0.0, dot(normalize(normalWCS), normalize(-vec3(sunlightWCS))));\n"
    "    vec3 lightAMB = vec3(1.0) * ambientIntensity;\n"
    "    vec3 lightDIFF = vec3(0.4) * diffuseIntensity;\n"
    "    LIGHT = min(lightAMB + lightDIFF, vec3(1.0));\n"
    "}";

const char* __editor_scene_shader_fs = 
    "#version 330\n"
    "\n"
    "in vec2 uv;\n"
    "in vec3 LIGHT;\n"
    "\n"
    "layout(location = 0) out vec4 out_colour;\n"
    "\n"
    "uniform sampler2D texture0;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    vec3 COLOR = texture(texture0, uv).xyz;\n"
    "    vec3 TOTAL = COLOR * LIGHT;\n"
    "    out_colour = vec4(TOTAL, 1.0);\n"
    "}";

const char* __editor_scene_wireframe_shader_vs =
    "#version 330\n"
    "\n"
    "layout (location = 0) in vec3 vertex_pos;\n"
    "layout (location = 1) in vec2 vertex_uv;\n"
    "layout (location = 2) in vec3 vertex_normal;\n"
    "\n"
    "uniform mat4 modelMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "uniform mat4 projMatrix;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    vec4 vertInClipSpace = projMatrix * viewMatrix * modelMatrix * vec4(vertex_pos, 1.0);\n"
    "    gl_Position = vertInClipSpace;\n"
    "}";

const char* __editor_scene_wireframe_shader_fs = 
    "#version 330\n"
    "\n"
    "layout(location = 0) out vec4 out_colour;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    out_colour = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "}";

const char* __editor_shader_face_selected_vs =
    "#version 330\n"
    "\n"
    "layout (location = 0) in vec3 vertex_pos;\n"
    "layout (location = 1) in vec2 vertex_uv;\n"
    "layout (location = 2) in vec3 vertex_normal;\n"
    "\n"
    "out vec2 uv;\n"
    "\n"
    "uniform mat4 modelMatrix;\n"
    "uniform mat4 viewMatrix;\n"
    "uniform mat4 projMatrix;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    uv = vertex_uv;\n"
    "    vec4 vertInClipSpace = projMatrix * viewMatrix * modelMatrix * vec4(vertex_pos, 1.0);\n"
    "    gl_Position = vertInClipSpace;\n"
    "}";

const char* __editor_shader_face_selected_fs = 
    "#version 330\n"
    "\n"
    "in vec2 uv;\n"
    "\n"
    "layout(location = 0) out vec4 out_colour;\n"
    "\n"
    "uniform sampler2D texture0;\n"
    "uniform vec3 tint;"
    "\n"
    "void main()\n"
    "{\n"
    "    vec3 COLOR = texture(texture0, uv).xyz;\n"
    "    vec3 TOTAL = COLOR * tint;\n"
    "    out_colour = vec4(TOTAL, 1.0);\n"
    "}";


const char* __finalpass_shader_vs =
    "#version 330\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec2 uv;\n"
    "out vec2 texcoord;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(pos, 0, 1.0);\n"
    "    texcoord = uv;\n"
    "}\n";

const char* __finalpass_shader_fs =
    "#version 330\n"
    "uniform sampler2D screen_texture;\n"
    "in vec2 texcoord;\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "    vec4 in_color = texture(screen_texture, texcoord);\n"
    "    if(in_color.w < 0.001)\n"
    "    {\n"
    "        discard;\n"
    "    }\n"
    "    color = in_color;\n"
    "}\n";

bool GLHasErrors()
{
#if INTERNAL_BUILD
    GLenum error = glGetError();

    if (error == GL_NO_ERROR) return false;

    while (error != GL_NO_ERROR)
    {
        const char* error_str = "";
        switch (error)
        {
            case GL_INVALID_OPERATION:
                error_str = "GL error: INVALID_OPERATION\n";
                break;
            case GL_INVALID_ENUM:
                error_str = "GL error: INVALID_ENUM\n";
                break;
            case GL_INVALID_VALUE:
                error_str = "GL error: INVALID_VALUE\n";
                break;
            case GL_OUT_OF_MEMORY:
                error_str = "GL error: OUT_OF_MEMORY\n";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error_str = "GL error: INVALID_FRAMEBUFFER_OPERATION\n";
                break;
        }

        fprintf(stderr, error_str);
        error = glGetError();
        ASSERT(false);
    }

    return true;
#else
    return false;
#endif
}


void GLCreateShaderProgram(GPUShader& shader, const char* vertexShaderStr, const char* fragmentShaderStr);
void GLCreateShaderProgram(GPUShader& shader, const char* vertexShaderStr, const char* geometryShaderStr, const char* fragmentShaderStr);


i32 GetCachedUniformLocation(const GPUShader& shader, const char* uniformName)
{
    auto location_iter = shader.uniformLocationsMap.find(uniformName);
    if (location_iter != shader.uniformLocationsMap.end())
    {
        return location_iter->second;
    }
    return -1;
}

void warningUniformNotFound(const GPUShader& shader, const char* uniformName)
{
    if (shader.bPrintWarnings)
    {
        printf("Warning: Uniform '%s' doesn't exist or isn't active on shader %d.\n", uniformName, shader.idShaderProgram);
    }
}

void cacheUniformLocation(GPUShader& shader, const char* uniformName)
{
    i32 location = glGetUniformLocation(shader.idShaderProgram, uniformName);
    if (location != 0xffffffff)
    {
        shader.uniformLocationsMap[std::string(uniformName)] = location;
    }
    else
    {
        printf("Warning! Unable to get the location of uniform '%s' for shader id %d...\n", uniformName, shader.idShaderProgram);
    }
}

void cacheUniformLocations(GPUShader& shader)
{
    shader.uniformLocationsMap.clear();

    GLint longest_uniform_name_length;
    GLint number_of_uniforms;
    glGetProgramiv(shader.idShaderProgram, GL_ACTIVE_UNIFORM_MAX_LENGTH, &longest_uniform_name_length);
    glGetProgramiv(shader.idShaderProgram, GL_ACTIVE_UNIFORMS, &number_of_uniforms);
    //printf("number of active uniforms for shader %d:  %d\n", shader.id_shader_program, number_of_uniforms);

    GLint readlength;
    GLint size;
    GLenum type;
    GLchar uniformName[128]; ASSERT(longest_uniform_name_length <= 128);

    /**If one or more elements of an array are active, the name of the array is returned in name, the type is returned
        * in type, and the size parameter returns the highest array element index used, plus one, as determined by the compiler
        * and/or linker. Only one active uniform variable will be reported for a uniform array.
    */
    for (GLint i = 0; i < number_of_uniforms; ++i)
    {
        glGetActiveUniform(shader.idShaderProgram, i, longest_uniform_name_length, &readlength, &size, &type, uniformName);
        cacheUniformLocation(shader, uniformName);
    }
}

void GLCompileShader(u32 program_id, const char* shader_code, GLenum shaderType)
{
    GLuint id_shader = glCreateShader(shaderType);             // Create an empty shader of given type and get id
    GLint code_length = (GLint)strlen(shader_code);
    glShaderSource(id_shader, 1, &shader_code, &code_length);   // Fill the empty shader with the shader code
    glCompileShader(id_shader);                                 // Compile the shader source

    GLint result = 0;
    GLchar eLog[1024] = {};
    glGetShaderiv(id_shader, GL_COMPILE_STATUS, &result);       // Make sure the shader compiled correctly
    if (!result)
    {
        glGetProgramInfoLog(id_shader, sizeof(eLog), nullptr, eLog);
        fprintf(stderr, "Error compiling the %d shader: '%s' \n", shaderType, eLog);
        return;
    }

    glAttachShader(program_id, id_shader);
}

bool GLCheckErrorAndValidate(GLuint program_id)
{
#if INTERNAL_BUILD
    GLint result = 0;
    GLchar eLog[1024] = {};
    glGetProgramiv(program_id, GL_LINK_STATUS, &result); // Make sure the program was created
    if (!result)
    {
        glGetProgramInfoLog(program_id, sizeof(eLog), nullptr, eLog);
        printf("Error linking program: '%s'! Aborting.\n", eLog);
        return true;
    }

    // Validate the program will work
    glValidateProgram(program_id);
    glGetProgramiv(program_id, GL_VALIDATE_STATUS, &result);
    if (!result)
    {
        glGetProgramInfoLog(program_id, sizeof(eLog), nullptr, eLog);
        printf("Error validating program %u: %s", program_id, eLog);
    }
#endif
    return false;
}


void GLLoadShaderProgramFromFile(GPUShader& shader, const char* vertexPath, const char* fragmentPath)
{
    std::string v = ReadFileString(vertexPath);
    std::string f = ReadFileString(fragmentPath);
    GLCreateShaderProgram(shader, v.c_str(), f.c_str());
}

void GLLoadShaderProgramFromFile(GPUShader& shader, const char* vertexPath, const char* geometryPath, const char* fragmentPath)
{
    std::string v = ReadFileString(vertexPath);
    std::string g = ReadFileString(geometryPath);
    std::string f = ReadFileString(fragmentPath);
    GLCreateShaderProgram(shader, v.c_str(), g.c_str(), f.c_str());
}

void GLCreateShaderProgram(GPUShader& shader, const char* vertexShaderStr, const char* fragmentShaderStr)
{
    shader.idShaderProgram = glCreateProgram(); // Create an empty shader program and get the id
    if (!shader.idShaderProgram)
    {
        fprintf(stderr, "Failed to create shader program! Aborting.\n");
        return;
    }
    GLCompileShader(shader.idShaderProgram, vertexShaderStr, GL_VERTEX_SHADER); // Compile and attach the shaders
    GLHasErrors();
    GLCompileShader(shader.idShaderProgram, fragmentShaderStr, GL_FRAGMENT_SHADER);
    GLHasErrors();
    glLinkProgram(shader.idShaderProgram); // Actually create the exectuable shader program on the graphics card
    GLHasErrors();

#if INTERNAL_BUILD
    if (GLCheckErrorAndValidate(shader.idShaderProgram))
    {
        return;
    }
#endif

    cacheUniformLocations(shader);
}

void GLCreateShaderProgram(GPUShader& shader, const char* vertexShaderStr, const char* geometryShaderStr, const char* fragmentShaderStr)
{

    shader.idShaderProgram = glCreateProgram(); // Create an empty shader program and get the id
    if (!shader.idShaderProgram)
    {
        fprintf(stderr, "Failed to create shader program! Aborting.\n");
        return;
    }
    GLCompileShader(shader.idShaderProgram, vertexShaderStr, GL_VERTEX_SHADER); // Compile and attach the shaders
    GLCompileShader(shader.idShaderProgram, geometryShaderStr, GL_GEOMETRY_SHADER);
    GLCompileShader(shader.idShaderProgram, fragmentShaderStr, GL_FRAGMENT_SHADER);
    glLinkProgram(shader.idShaderProgram); // Actually create the exectuable shader program on the graphics card

#if INTERNAL_BUILD
    if (GLCheckErrorAndValidate(shader.idShaderProgram))
    {
        return;
    }
#endif

    cacheUniformLocations(shader);
}

#ifdef GL_VERSION_4_3_OR_HIGHER
void GLCreateComputeShaderProgram(GPUShader& shader, const char* computeShaderStr)
{
    shader.idShaderProgram = glCreateProgram();
    if (!shader.idShaderProgram)
    {
        fprintf(stderr, "Failed to create shader program! Aborting.\n");
        return;
    }
    GLCompileShader(shader.idShaderProgram, computeShaderStr, GL_COMPUTE_SHADER);
    glLinkProgram(shader.idShaderProgram);

#if INTERNAL_BUILD
    if (GLCheckErrorAndValidate(shader.idShaderProgram))
    {
        return;
    }
#endif

    cacheUniformLocations(shader);
}

void GLLoadComputeShaderProgramFromFile(GPUShader& shader, const char* computePath)
{
    std::string c = ReadFileString(computePath);
    GLCreateComputeShaderProgram(shader, c.c_str());
}
#endif

/** Delete the shader program off GPU memory */
void GLDeleteShader(GPUShader& shader)
{
    if (shader.idShaderProgram == 0)
    {
        printf("WARNING: Passed an unloaded shader program to GLDeleteShader! Aborting.\n");
        return;
    }
    glDeleteProgram(shader.idShaderProgram);
}

/** Telling opengl to start using this shader program */
void UseShader(const GPUShader& shader)
{
    if (shader.idShaderProgram == 0)
    {
        printf("WARNING: Passed an unloaded shader program to UseShader! Aborting.\n");
        return;
    }
    glUseProgram(shader.idShaderProgram);
}

void GLBind1i(const GPUShader& shader, const char* uniformName, GLint v0)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform1i(location, v0);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind2i(const GPUShader& shader, const char* uniformName, GLint v0, GLint v1)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform2i(location, v0, v1);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind3i(const GPUShader& shader, const char* uniformName, GLint v0, GLint v1, GLint v2)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform3i(location, v0, v1, v2);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind4i(const GPUShader& shader, const char* uniformName, GLint v0, GLint v1, GLint v2, GLint v3)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform4i(location, v0, v1, v2, v3);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind1f(const GPUShader& shader, const char* uniformName, GLfloat v0)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform1f(location, v0);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind2f(const GPUShader& shader, const char* uniformName, GLfloat v0, GLfloat v1)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform2f(location, v0, v1);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind3f(const GPUShader& shader, const char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform3f(location, v0, v1, v2);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBind4f(const GPUShader& shader, const char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform4f(location, v0, v1, v2, v3);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBindMatrix3fv(const GPUShader& shader, const char* uniformName, GLsizei count, const GLfloat* value)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniformMatrix3fv(location, count, GL_FALSE, value);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}

void GLBindMatrix4fv(const GPUShader& shader, const char* uniformName, GLsizei count, const GLfloat* value)
{
    i32 location = GetCachedUniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniformMatrix4fv(location, count, GL_FALSE, value);
    }
    else
    {
        warningUniformNotFound(shader, uniformName);
    }
}
