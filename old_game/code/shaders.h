#pragma once

#include "common.h"

struct GPUShader
{
    GLuint idShaderProgram = 0; // id of this shader program in GPU memory

    bool bPrintWarnings = true;
    std::unordered_map<std::string, i32> uniformLocationsMap;
};

void GLLoadShaderProgramFromFile(GPUShader& shader, const char* vertexPath, const char* fragmentPath);
void GLLoadShaderProgramFromFile(GPUShader& shader, const char* vertexPath, const char* geometryPath, const char* fragmentPath);
void GLCreateShaderProgram(GPUShader& shader, const char* vertexShaderStr, const char* fragmentShaderStr);
void GLCreateShaderProgram(GPUShader& shader, const char* vertexShaderStr, const char* geometryShaderStr, const char* fragmentShaderStr);
#ifdef GL_VERSION_4_3_OR_HIGHER
void GLLoadComputeShaderProgramFromFile(GPUShader& shader, const char* computePath);
void GLCreateComputeShaderProgram(GPUShader& shader, const char* computeShaderStr);
#endif
void GLDeleteShader(GPUShader& shader);

void UseShader(const GPUShader& shader);

i32 UniformLocation(const GPUShader& shader, const char* uniformName);

void GLBind1i(const GPUShader& shader, const char* uniformName, GLint v0);
void GLBind2i(const GPUShader& shader, const char* uniformName, GLint v0, GLint v1);
void GLBind3i(const GPUShader& shader, const char* uniformName, GLint v0, GLint v1, GLint v2);
void GLBind4i(const GPUShader& shader, const char* uniformName, GLint v0, GLint v1, GLint v2, GLint v3);
void GLBind1f(const GPUShader& shader, const char* uniformName, GLfloat v0);
void GLBind2f(const GPUShader& shader, const char* uniformName, GLfloat v0, GLfloat v1);
void GLBind3f(const GPUShader& shader, const char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2);
void GLBind4f(const GPUShader& shader, const char* uniformName, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
void GLBindMatrix3fv(const GPUShader& shader, const char* uniformName, GLsizei count, const GLfloat* value);
void GLBindMatrix4fv(const GPUShader& shader, const char* uniformName, GLsizei count, const GLfloat* value);

bool GLHasErrors();
