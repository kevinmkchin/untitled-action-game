
#ifdef JPH_DEBUG_RENDERER

jph_debug_draw_gl3_t *JoltDebugDrawer;

void JoltDebugDrawCharacterState(jph_debug_draw_gl3_t *mDebugRenderer, const JPH::CharacterBase *inCharacter, 
    JPH::RMat44Arg inCharacterTransform, JPH::Vec3Arg inCharacterVelocity)
{
    // Draw current location
    // Drawing prior to update since the physics system state is also that prior to the simulation step (so that all detected collisions etc. make sense)
    mDebugRenderer->DrawCoordinateSystem(inCharacterTransform, 0.1f);

    // Draw the state of the ground contact
    JPH::CharacterBase::EGroundState ground_state = inCharacter->GetGroundState();
    if (ground_state != JPH::CharacterBase::EGroundState::InAir)
    {
        JPH::RVec3 ground_position = inCharacter->GetGroundPosition();
        JPH::Vec3 ground_normal = inCharacter->GetGroundNormal();
        JPH::Vec3 ground_velocity = inCharacter->GetGroundVelocity();

        // Draw ground position
        mDebugRenderer->DrawMarker(ground_position, JPH::Color::sRed, 0.1f);
        mDebugRenderer->DrawArrow(ground_position, ground_position + 2.0f * ground_normal, JPH::Color::sGreen, 0.1f);

        // Draw ground velocity
        if (!ground_velocity.IsNearZero())
            mDebugRenderer->DrawArrow(ground_position, ground_position + ground_velocity, JPH::Color::sBlue, 0.1f);
    }

    // Draw provided character velocity
    if (!inCharacterVelocity.IsNearZero())
        mDebugRenderer->DrawArrow(inCharacterTransform.GetTranslation(), inCharacterTransform.GetTranslation() + inCharacterVelocity, JPH::Color::sYellow, 0.1f);

    // Draw text info
    const JPH::PhysicsMaterial *ground_material = inCharacter->GetGroundMaterial();
    JPH::Vec3 horizontal_velocity = inCharacterVelocity;
    horizontal_velocity.SetY(0);
    mDebugRenderer->DrawText3D(inCharacterTransform.GetTranslation(), JPH::StringFormat("State: %s\nMat: %s\nHorizontal Vel: %.1f m/s\nVertical Vel: %.1f m/s", JPH::CharacterBase::sToString(ground_state), ground_material->GetDebugName(), (double)horizontal_velocity.Length(), (double)inCharacterVelocity.GetY()), JPH::Color::sWhite, 0.25f);
}

void jph_debug_draw_gl3_t::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor)
{
    SupportRenderer.DrawLine(FromJoltVector(inFrom), FromJoltVector(inTo), 
        FromJoltVectorNoConvert(inColor.ToVec4()));
}

void jph_debug_draw_gl3_t::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow)
{
    SupportRenderer.DrawTri(
        FromJoltVector(inV1),
        FromJoltVector(inV2),
        FromJoltVector(inV3),
        FromJoltVectorNoConvert(inColor.ToVec4()));
}

void jph_debug_draw_gl3_t::DrawText3D(JPH::RVec3Arg inPosition, const std::string_view &inString, JPH::ColorArg inColor, float inHeight)
{
    // todo
}

#endif // JPH_DEBUG_RENDERER
