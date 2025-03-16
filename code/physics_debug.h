#pragma once

#ifdef JPH_DEBUG_RENDERER
#include <Jolt/Core/Color.h>
#include <Jolt/Core/StringTools.h>
#include <Jolt/Renderer/DebugRendererSimple.h>

void JoltDebugDrawCharacterState(class jph_debug_draw_gl3_t *mDebugRenderer, const JPH::CharacterBase *inCharacter, 
    JPH::RMat44Arg inCharacterTransform, JPH::Vec3Arg inCharacterVelocity);

class jph_debug_draw_gl3_t : public JPH::DebugRendererSimple
{
public:
    JPH_OVERRIDE_NEW_DELETE
    // jph_debug_draw_gl3_t();

    void Init();
    void Destroy();

    void Ready();
    void Flush(float *ViewProjectionMatrix);

public:
    virtual void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) override;
    virtual void DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow) override;
    virtual void DrawText3D(JPH::RVec3Arg inPosition, const std::string_view &inString, JPH::ColorArg inColor, float inHeight) override;

private:
    GPUMesh DebugDrawMesh;
    dynamic_array<float> LinesVertices;
    dynamic_array<float> TrisVertices;

    GPUShader JPH_DEBUG_GL3_SHADER;
    const char* JPH_DEBUG_GL3_VS =
    "#version 330\n"
    "layout (location = 0) in vec3 VertexPos;\n"
    "layout (location = 1) in vec4 VertexColor;\n"
    "out vec4 Color;\n"
    "uniform mat4 ViewProjectionMatrix;\n"
    "void main()\n"
    "{\n"
    "    Color = VertexColor;\n"
    "    gl_Position = ViewProjectionMatrix * vec4(VertexPos, 1.0);\n"
    "}";

    const char* JPH_DEBUG_GL3_FS = 
    "#version 330\n"
    "\n"
    "in vec4 Color;\n"
    "\n"
    "layout(location = 0) out vec4 OutColor;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    OutColor = Color;\n"
    "}";
};

extern jph_debug_draw_gl3_t *JoltDebugDrawer;

#endif // JPH_DEBUG_RENDERER

