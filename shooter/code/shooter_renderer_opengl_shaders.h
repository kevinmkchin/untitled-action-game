#pragma once

// TODO(Kevin): Get rid of the printfs and fprintfs
// TODO(Kevin): Get rid of std::unordered_map
// TODO(Kevin): Get rid of std::string

#include <unordered_map>
#include <string>

global char* FinalCompositeVS =
    "#version 330\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec2 uv;\n"
    "out vec2 texcoord;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(pos, 0, 1.0);\n"
    "    texcoord = uv;\n"
    "}\n";

global char* FinalCompositeFS =
    "#version 330\n"
    "uniform sampler2D screen_texture;\n"
    "in vec2 texcoord;\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "    vec4 in_color = texture(screen_texture, texcoord);\n"
    "    color = in_color;\n"
    "}\n";

global float FinalCompositeQuadVertices[16] =
{
    //  x   y    u    v
    -1.0f, -1.0f, 0.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f
};

global uint32 FinalCompositeQuadIndices[6] =
{
    0, 1, 3,
    0, 3, 2
};

internal open_gl_mesh FinalCompositeQuadMesh()
{
    GLsizei StrideInBytes = sizeof(float)*4;
    GLsizeiptr VertexBufferSzInBytes = (GLsizeiptr)sizeof(float)*ArrayCount(FinalCompositeQuadVertices);
    GLsizeiptr IndexBufferSzInBytes = (GLsizeiptr)sizeof(uint32)*ArrayCount(FinalCompositeQuadIndices);

    open_gl_mesh Mesh = {};

    glGenVertexArrays(1, &Mesh.VAO);
    glBindVertexArray(Mesh.VAO);
    glGenBuffers(1, &Mesh.VBO);
    glBindBuffer(GL_ARRAY_BUFFER, Mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, VertexBufferSzInBytes, FinalCompositeQuadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, StrideInBytes, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, StrideInBytes, (void*)(sizeof(float)*2));

    glGenBuffers(1, &Mesh.IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Mesh.IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSzInBytes, FinalCompositeQuadIndices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Mesh.IndicesCount = ArrayCount(FinalCompositeQuadIndices);

    return Mesh;
}

struct open_gl_shader
{
    GLuint ProgramId;
    bool32 PrintWarnings;
    std::unordered_map<std::string, int32> UniformLocations;
};

internal void
GLCreateShaderProgram(open_gl_shader& shader, char* vertexShaderStr, char* fragmentShaderStr);
internal void
GLCreateShaderProgram(open_gl_shader& shader, char* vertexShaderStr, char* geometryShaderStr, char* fragmentShaderStr);
#if 0
internal void
GLLoadComputeShaderProgramFromFile(open_gl_shader& shader, char* computePath);
internal void
GLCreateComputeShaderProgram(open_gl_shader& shader, char* computeShaderStr);
#endif
internal void
GLDeleteShader(open_gl_shader& shader);
internal int32
UniformLocation(open_gl_shader& shader, char* uniformName);
internal void
GLBind1i(open_gl_shader& shader, char* uniformName, GLint v0);
internal void
GLBind2i(open_gl_shader& shader, char* uniformName, GLint v0, GLint v1);
internal void
GLBind3i(open_gl_shader& shader, char* uniformName, GLint v0, GLint v1, GLint v2);
internal void
GLBind4i(open_gl_shader& shader, char* uniformName, GLint v0, GLint v1, GLint v2, GLint v3);
internal void
GLBind1f(open_gl_shader& shader, char* uniformName, GLfloat v0);
internal void
GLBind2f(open_gl_shader& shader, char* uniformName, GLfloat v0, GLfloat v1);
internal void
GLBind3f(open_gl_shader& shader, char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2);
internal void
GLBind4f(open_gl_shader& shader, char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
internal void
GLBindMatrix3fv(open_gl_shader& shader, char* uniformName, GLsizei count, GLfloat* value);
internal void
GLBindMatrix4fv(open_gl_shader& shader, char* uniformName, GLsizei count, GLfloat* value);
internal bool
GLHasErrors();

internal bool
GLHasErrors()
{
#if HANDMADE_INTERNAL
    GLenum error = glGetError();

    if (error == GL_NO_ERROR) return false;

    while (error != GL_NO_ERROR)
    {
        char* error_str = "";
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
        Assert(false);
    }

    return true;
#else
    return false;
#endif
}

internal void
WarningUniformNotFound(open_gl_shader &shader, char *uniformName)
{
    if (shader.PrintWarnings)
    {
        printf("Warning: Uniform '%s' doesn't exist or isn't active on shader %d.\n", uniformName, shader.ProgramId);
    }
}

internal int32
UniformLocation(open_gl_shader& shader, char* uniformName)
{
    auto location_iter = shader.UniformLocations.find(uniformName);
    if (location_iter != shader.UniformLocations.end())
    {
        return location_iter->second;
    }
    WarningUniformNotFound(shader, uniformName);
    return -1;
}

internal void
cacheUniformLocation(open_gl_shader& shader, char* uniformName)
{
    int32 location = glGetUniformLocation(shader.ProgramId, uniformName);
    if (location != 0xffffffff)
    {
        shader.UniformLocations[std::string(uniformName)] = location;
    }
    else
    {
        printf("Warning! Unable to get the location of uniform '%s' for shader id %d...\n", uniformName, shader.ProgramId);
    }
}

internal void
cacheUniformLocations(open_gl_shader& shader)
{
    shader.UniformLocations.clear();

    GLint longest_uniform_name_length;
    GLint number_of_uniforms;
    glGetProgramiv(shader.ProgramId, GL_ACTIVE_UNIFORM_MAX_LENGTH, &longest_uniform_name_length);
    glGetProgramiv(shader.ProgramId, GL_ACTIVE_UNIFORMS, &number_of_uniforms);
    //printf("number of active uniforms for shader %d:  %d\n", shader.id_shader_program, number_of_uniforms);

    GLint readlength;
    GLint size;
    GLenum type;
    GLchar uniformName[128]; Assert(longest_uniform_name_length <= 128);

    /**If one or more elements of an array are active, the name of the array is returned in name, the type is returned
        * in type, and the size parameter returns the highest array element index used, plus one, as determined by the compiler
        * and/or linker. Only one active uniform variable will be reported for a uniform array.
    */
    for (GLint i = 0; i < number_of_uniforms; ++i)
    {
        glGetActiveUniform(shader.ProgramId, i, longest_uniform_name_length, &readlength, &size, &type, uniformName);
        cacheUniformLocation(shader, uniformName);
    }
}

internal void
GLCompileShader(uint32 program_id, char* shader_code, GLenum shaderType)
{
    GLuint id_shader = glCreateShader(shaderType);              // Create an empty shader of given type and get id
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
#if HANDMADE_INTERNAL
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

internal void
GLCreateShaderProgram(open_gl_shader& shader, char* vertexShaderStr, char* fragmentShaderStr)
{
    shader.ProgramId = glCreateProgram(); // Create an empty shader program and get the id
    if (!shader.ProgramId)
    {
        fprintf(stderr, "Failed to create shader program! Aborting.\n");
        return;
    }
    GLCompileShader(shader.ProgramId, vertexShaderStr, GL_VERTEX_SHADER); // Compile and attach the shaders
    GLHasErrors();
    GLCompileShader(shader.ProgramId, fragmentShaderStr, GL_FRAGMENT_SHADER);
    GLHasErrors();
    glLinkProgram(shader.ProgramId); // Actually create the exectuable shader program on the graphics card
    GLHasErrors();

#if HANDMADE_INTERNAL
    if (GLCheckErrorAndValidate(shader.ProgramId))
    {
        return;
    }
#endif

    cacheUniformLocations(shader);
}

internal void
GLCreateShaderProgram(open_gl_shader& shader, char* vertexShaderStr, char* geometryShaderStr, char* fragmentShaderStr)
{

    shader.ProgramId = glCreateProgram(); // Create an empty shader program and get the id
    if (!shader.ProgramId)
    {
        fprintf(stderr, "Failed to create shader program! Aborting.\n");
        return;
    }
    GLCompileShader(shader.ProgramId, vertexShaderStr, GL_VERTEX_SHADER); // Compile and attach the shaders
    GLCompileShader(shader.ProgramId, geometryShaderStr, GL_GEOMETRY_SHADER);
    GLCompileShader(shader.ProgramId, fragmentShaderStr, GL_FRAGMENT_SHADER);
    glLinkProgram(shader.ProgramId); // Actually create the exectuable shader program on the graphics card

#if HANDMADE_INTERNAL
    if (GLCheckErrorAndValidate(shader.ProgramId))
    {
        return;
    }
#endif

    cacheUniformLocations(shader);
}

#if 0
internal void
GLCreateComputeShaderProgram(open_gl_shader& shader, char* computeShaderStr)
{
    shader.ProgramId = glCreateProgram();
    if (!shader.ProgramId)
    {
        fprintf(stderr, "Failed to create shader program! Aborting.\n");
        return;
    }
    GLCompileShader(shader.ProgramId, computeShaderStr, GL_COMPUTE_SHADER);
    glLinkProgram(shader.ProgramId);

#if HANDMADE_INTERNAL
    if (GLCheckErrorAndValidate(shader.ProgramId))
    {
        return;
    }
#endif

    cacheUniformLocations(shader);
}

internal void
GLLoadComputeShaderProgramFromFile(open_gl_shader& shader, char* computePath)
{
    std::string c = ReadFileString(computePath);
    GLCreateComputeShaderProgram(shader, c.c_str());
}
#endif

internal void
GLDeleteShader(open_gl_shader& shader)
{
    if (shader.ProgramId == 0)
    {
        printf("WARNING: Passed an unloaded shader program to GLDeleteShader! Aborting.\n");
        return;
    }
    glDeleteProgram(shader.ProgramId);
}

internal void
GLBind1i(open_gl_shader& shader, char* uniformName, GLint v0)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform1i(location, v0);
    }
}

internal void
GLBind2i(open_gl_shader& shader, char* uniformName, GLint v0, GLint v1)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform2i(location, v0, v1);
    }
}

internal void
GLBind3i(open_gl_shader& shader, char* uniformName, GLint v0, GLint v1, GLint v2)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform3i(location, v0, v1, v2);
    }
    else
    {
        WarningUniformNotFound(shader, uniformName);
    }
}

internal void
GLBind4i(open_gl_shader& shader, char* uniformName, GLint v0, GLint v1, GLint v2, GLint v3)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform4i(location, v0, v1, v2, v3);
    }
}

internal void
GLBind1f(open_gl_shader& shader, char* uniformName, GLfloat v0)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform1f(location, v0);
    }
}

internal void
GLBind2f(open_gl_shader& shader, char* uniformName, GLfloat v0, GLfloat v1)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform2f(location, v0, v1);
    }
}

internal void
GLBind3f(open_gl_shader& shader, char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform3f(location, v0, v1, v2);
    }
}

internal void
GLBind4f(open_gl_shader& shader, char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniform4f(location, v0, v1, v2, v3);
    }
}

internal void
GLBindMatrix3fv(open_gl_shader& shader, char* uniformName, GLsizei count, GLfloat* value)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniformMatrix3fv(location, count, GL_FALSE, value);
    }
}

internal void
GLBindMatrix4fv(open_gl_shader& shader, char* uniformName, GLsizei count, GLfloat* value)
{
    int32 location = UniformLocation(shader, uniformName);
    if (location >= 0)
    {
        glUniformMatrix4fv(location, count, GL_FALSE, value);
    }
}

