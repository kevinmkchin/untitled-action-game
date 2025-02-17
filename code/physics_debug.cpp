
#ifdef JPH_DEBUG_RENDERER

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

// jph_debug_draw_gl3_t::jph_debug_draw_gl3_t()
// {
//     Initialize();
// }

void jph_debug_draw_gl3_t::Init()
{
    GLCreateShaderProgram(JPH_DEBUG_GL3_SHADER, JPH_DEBUG_GL3_VS, JPH_DEBUG_GL3_FS);

    // pos x y z, color r g b a
    CreateGPUMesh(&DebugDrawMesh, 3, 4, 0, GL_DYNAMIC_DRAW);
}

void jph_debug_draw_gl3_t::Destroy()
{
    if (JPH_DEBUG_GL3_SHADER.idShaderProgram)
        GLDeleteShader(JPH_DEBUG_GL3_SHADER);
    if (DebugDrawMesh.idVAO)
        DeleteGPUMesh(DebugDrawMesh.idVAO, DebugDrawMesh.idVBO);
}

void jph_debug_draw_gl3_t::Ready()
{
    LinesVertices.setlen(0);
    TrisVertices.setlen(0);
}

void jph_debug_draw_gl3_t::Flush(float *ViewProjectionMatrix)
{
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    UseShader(JPH_DEBUG_GL3_SHADER);
    GLBindMatrix4fv(JPH_DEBUG_GL3_SHADER, "ViewProjectionMatrix", 1, ViewProjectionMatrix);
    //GLHasErrors();

    if (LinesVertices.lenu() > 0)
    {
        RebindGPUMesh(&DebugDrawMesh, sizeof(float)*LinesVertices.lenu(), LinesVertices.data);
        int VertexCount = (int)LinesVertices.lenu() / 7;
        GLHasErrors();

        glBindVertexArray(DebugDrawMesh.idVAO);
        glBindBuffer(GL_ARRAY_BUFFER, DebugDrawMesh.idVBO);
        glDrawArrays(GL_LINES, 0, VertexCount);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        GLHasErrors();
    
        LinesVertices.setlen(0);
    }

    if (TrisVertices.lenu() > 0)
    {
        RebindGPUMesh(&DebugDrawMesh, sizeof(float)*TrisVertices.lenu(), TrisVertices.data);
        int VertexCount = (int)TrisVertices.lenu() / 7;
        GLHasErrors();

        glBindVertexArray(DebugDrawMesh.idVAO);
        glBindBuffer(GL_ARRAY_BUFFER, DebugDrawMesh.idVBO);
        glDrawArrays(GL_TRIANGLES, 0, VertexCount);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        GLHasErrors();

        TrisVertices.setlen(0);
    }
}

void jph_debug_draw_gl3_t::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor)
{
    JPH::Vec4 Colorf = inColor.ToVec4();
    LinesVertices.put(inFrom.GetX());
    LinesVertices.put(inFrom.GetY());
    LinesVertices.put(inFrom.GetZ());
    LinesVertices.put(Colorf.GetX());
    LinesVertices.put(Colorf.GetY());
    LinesVertices.put(Colorf.GetZ());
    LinesVertices.put(Colorf.GetW());

    LinesVertices.put(inTo.GetX());
    LinesVertices.put(inTo.GetY());
    LinesVertices.put(inTo.GetZ());
    LinesVertices.put(Colorf.GetX());
    LinesVertices.put(Colorf.GetY());
    LinesVertices.put(Colorf.GetZ());
    LinesVertices.put(Colorf.GetW());
}

void jph_debug_draw_gl3_t::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow)
{
    JPH::Vec4 Colorf = inColor.ToVec4();
    TrisVertices.put(inV1.GetX());
    TrisVertices.put(inV1.GetY());
    TrisVertices.put(inV1.GetZ());
    TrisVertices.put(Colorf.GetX());
    TrisVertices.put(Colorf.GetY());
    TrisVertices.put(Colorf.GetZ());
    TrisVertices.put(Colorf.GetW());

    TrisVertices.put(inV2.GetX());
    TrisVertices.put(inV2.GetY());
    TrisVertices.put(inV2.GetZ());
    TrisVertices.put(Colorf.GetX());
    TrisVertices.put(Colorf.GetY());
    TrisVertices.put(Colorf.GetZ());
    TrisVertices.put(Colorf.GetW());

    TrisVertices.put(inV3.GetX());
    TrisVertices.put(inV3.GetY());
    TrisVertices.put(inV3.GetZ());
    TrisVertices.put(Colorf.GetX());
    TrisVertices.put(Colorf.GetY());
    TrisVertices.put(Colorf.GetZ());
    TrisVertices.put(Colorf.GetW());
}

void jph_debug_draw_gl3_t::DrawText3D(JPH::RVec3Arg inPosition, const std::string_view &inString, JPH::ColorArg inColor, float inHeight)
{
    // todo
}

#endif // JPH_DEBUG_RENDERER
