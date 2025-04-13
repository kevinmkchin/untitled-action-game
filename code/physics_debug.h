#pragma once

#ifdef JPH_DEBUG_RENDERER
#include "common.h"
#include "physics.h"
#include "primitives.h"
#include <Jolt/Core/Color.h>
#include <Jolt/Core/StringTools.h>
#include <Jolt/Renderer/DebugRendererSimple.h>

void JoltDebugDrawCharacterState(class jph_debug_draw_gl3_t *mDebugRenderer, const JPH::CharacterBase *inCharacter, 
    JPH::RMat44Arg inCharacterTransform, JPH::Vec3Arg inCharacterVelocity);

class jph_debug_draw_gl3_t : public JPH::DebugRendererSimple
{
public:
    JPH_OVERRIDE_NEW_DELETE

    support_renderer_t *SupportRenderer = nullptr;

public:
    virtual void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) override;
    virtual void DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow) override;
    virtual void DrawText3D(JPH::RVec3Arg inPosition, const std::string_view &inString, JPH::ColorArg inColor, float inHeight) override;

};

#endif // JPH_DEBUG_RENDERER

