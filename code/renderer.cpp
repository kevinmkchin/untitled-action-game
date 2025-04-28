#include "renderer.h"
#include "primitives.h"
#include "shaders.h"
#include "game.h"
#include "enemy.h"
#include "game_assets.h"

static GPUShader Sha_GameLevel;
static GPUShader Sha_ParticlesDefault;
static GPUShader Sha_ModelTexturedLit;
static GPUShader Sha_ModelSkinnedLit;

void AcquireRenderingResources()
{
    GLLoadShaderProgramFromFile(Sha_GameLevel, 
        shader_path("__game_level.vert").c_str(), 
        shader_path("__game_level.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ParticlesDefault, 
        shader_path("particles.vert").c_str(), 
        shader_path("particles.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ModelTexturedLit, 
        shader_path("model_textured.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    GLLoadShaderProgramFromFile(Sha_ModelSkinnedLit, 
        shader_path("model_skinned.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
    InstanceDrawing_AcquireGPUResources();
}

void ReleaseRenderingResources()
{
    GLDeleteShader(Sha_GameLevel);
    GLDeleteShader(Sha_ParticlesDefault);
    GLDeleteShader(Sha_ModelTexturedLit);
    GLDeleteShader(Sha_ModelSkinnedLit);
    InstancedDrawing_ReleaseGPUResources();
}

void RenderTexturedLitMeshes(
    fixed_array<textured_lit_drawinfo> &TexLitDrawInfos,
    struct game_state *GameState,
    const mat4 &ProjFromView, 
    const mat4 &ViewFromWorld)
{
    UseShader(Sha_ModelTexturedLit);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(Sha_ModelTexturedLit, "MuzzleFlash", 
        GameState->Player.Weapon.MuzzleFlash.x, 
        GameState->Player.Weapon.MuzzleFlash.y, 
        GameState->Player.Weapon.MuzzleFlash.z, 
        GameState->Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_ModelTexturedLit, "ProjFromView", 1, ProjFromView.ptr());
    GLBindMatrix4fv(Sha_ModelTexturedLit, "ViewFromWorld", 1, ViewFromWorld.ptr());

    for (u32 i = 0; i < TexLitDrawInfos.lenu(); ++i)
    {
        textured_lit_drawinfo &DrawInfo = TexLitDrawInfos[i];

        i32 loc0 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.AmbientCube[0]");
        i32 loc1 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.DoSunLight");
        i32 loc2 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.DirectionToSun");
        i32 loc3 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.PointLightsCount");
        i32 loc4 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.PointLightsPos[0]");
        i32 loc5 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.PointLightsAttLin[0]");
        i32 loc6 = UniformLocation(Sha_ModelTexturedLit, "ModelLighting.PointLightsAttQuad[0]");
        glUniform1fv(loc0, 6, DrawInfo.RenderingInfo.AmbientCube);
        glUniform1i(loc1, DrawInfo.RenderingInfo.DoSunLight);
        glUniform3fv(loc2, 1, (float*)&GameState->DirectionToSun);
        glUniform1i(loc3, DrawInfo.RenderingInfo.PointLightsCount);
        glUniform4fv(loc4, 4, (float*)DrawInfo.RenderingInfo.PointLightsPos);
        glUniform1fv(loc5, 4, (float*)DrawInfo.RenderingInfo.PointLightsAttLin);
        glUniform1fv(loc6, 4, (float*)DrawInfo.RenderingInfo.PointLightsAttQuad);

        GLBindMatrix4fv(Sha_ModelTexturedLit, "WorldFromModel", 1, 
            DrawInfo.RenderingInfo.WorldFromModel.ptr());

        glActiveTexture(GL_TEXTURE0);
        if (DrawInfo.T.id > 0)
            glBindTexture(GL_TEXTURE_2D, DrawInfo.T.id);
        else
            glBindTexture(GL_TEXTURE_2D, Assets.DefaultMissingTexture.id);

        RenderGPUMeshIndexed(DrawInfo.M);
    }
}

void FillSkinnedModelDrawInfo(
    sm_drawinfo *DrawInfo,
    struct game_state *GameState,
    vec3 ModelCentroid,
    vec3 RenderPosition,
    quat RenderRotation,
    animator_t *Animator,
    skinned_model_t *Model)
{
    FillModelInstanceData(GameState,
        &DrawInfo->RenderingInfo,
        ModelCentroid,
        RenderPosition,
        RenderRotation,
        nullptr);
    DrawInfo->Animator = Animator;
    DrawInfo->Model = Model;
}

void RenderSkinnedModels(
    fixed_array<sm_drawinfo> &SMDrawInfos,
    struct game_state *GameState,
    const mat4 &ProjFromView, 
    const mat4 &ViewFromWorld)
{
    UseShader(Sha_ModelSkinnedLit);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(Sha_ModelSkinnedLit, "MuzzleFlash", 
        GameState->Player.Weapon.MuzzleFlash.x, 
        GameState->Player.Weapon.MuzzleFlash.y, 
        GameState->Player.Weapon.MuzzleFlash.z, 
        GameState->Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_ModelSkinnedLit, "Projection", 1, ProjFromView.ptr());
    GLBindMatrix4fv(Sha_ModelSkinnedLit, "View", 1, ViewFromWorld.ptr());

    for (u32 i = 0; i < SMDrawInfos.lenu(); ++i)
    {
        sm_drawinfo &SMInfo = SMDrawInfos[i];

        i32 loc0 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.AmbientCube[0]");
        i32 loc1 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.DoSunLight");
        i32 loc2 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.DirectionToSun");
        i32 loc3 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.PointLightsCount");
        i32 loc4 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.PointLightsPos[0]");
        i32 loc5 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.PointLightsAttLin[0]");
        i32 loc6 = UniformLocation(Sha_ModelSkinnedLit, "ModelLighting.PointLightsAttQuad[0]");
        glUniform1fv(loc0, 6, SMInfo.RenderingInfo.AmbientCube);
        glUniform1i(loc1, SMInfo.RenderingInfo.DoSunLight);
        glUniform3fv(loc2, 1, (float*)&GameState->DirectionToSun);
        glUniform1i(loc3, SMInfo.RenderingInfo.PointLightsCount);
        glUniform4fv(loc4, 4, (float*)SMInfo.RenderingInfo.PointLightsPos);
        glUniform1fv(loc5, 4, (float*)SMInfo.RenderingInfo.PointLightsAttLin);
        glUniform1fv(loc6, 4, (float*)SMInfo.RenderingInfo.PointLightsAttQuad);

        mat4 &ModelMatrix = SMInfo.RenderingInfo.WorldFromModel;
        GLBindMatrix4fv(Sha_ModelSkinnedLit, "Model", 1, ModelMatrix.ptr());
        GLBindMatrix4fv(Sha_ModelSkinnedLit, "FinalBonesMatrices[0]", MAX_BONES, 
            SMInfo.Animator->SkinningMatrixPalette[0].ptr());
        for (size_t j = 0; j < SMInfo.Model->Meshes.length; ++j)
        {
            skinned_mesh_t M = SMInfo.Model->Meshes[j];
            GPUTexture T = SMInfo.Model->Textures[j];

            glActiveTexture(GL_TEXTURE0);
            if (T.id > 0)
                glBindTexture(GL_TEXTURE_2D, T.id);
            else
                glBindTexture(GL_TEXTURE_2D, Assets.DefaultMissingTexture.id);

            glBindVertexArray(M.VAO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, M.IBO);
            glDrawElements(GL_TRIANGLES, M.IndicesCount, GL_UNSIGNED_INT, nullptr);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
    }
}

void RenderGameState(game_state *GameState)
{
    app_state *AppState = GameState->AppState;

    glBindFramebuffer(GL_FRAMEBUFFER, AppState->RenderTargetGame->fbo);
    glViewport(0, 0, AppState->RenderTargetGame->width, AppState->RenderTargetGame->height);
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    UseShader(Sha_GameLevel);
    glEnable(GL_CULL_FACE);
    GLBind4f(Sha_GameLevel, "MuzzleFlash",
        GameState->Player.Weapon.MuzzleFlash.x, 
        GameState->Player.Weapon.MuzzleFlash.y, 
        GameState->Player.Weapon.MuzzleFlash.z, 
        GameState->Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_GameLevel, "projMatrix", 1, GameState->ClipFromView.ptr());
    GLBindMatrix4fv(Sha_GameLevel, "viewMatrix", 1, GameState->ViewFromWorld.ptr());
    for (size_t i = 0; i < GameState->GameLevelFaceBatches.size(); ++i)
    {
        face_batch_t fb = GameState->GameLevelFaceBatches.at(i);
        RenderFaceBatch(&Sha_GameLevel, &fb);
    }

    RenderTexturedLitMeshes(GameState->TexturedLitRenderData, GameState,
        GameState->ClipFromView, GameState->ViewFromWorld);

    RenderSkinnedModels(GameState->SMRenderData, GameState,
        GameState->ClipFromView, GameState->ViewFromWorld);

    SortAndDrawInstancedModels(GameState, 
        GameState->StaticInstances,
        GameState->DynamicInstances, 
        GameState->ClipFromView,
        GameState->ViewFromWorld);

    UseShader(Sha_ParticlesDefault);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE); // Particles should depth test but not write to depth buffer
    GLBindMatrix4fv(Sha_ParticlesDefault, "ClipFromWorld", 1, GameState->ClipFromWorld.ptr());
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, Assets.DefaultMissingTexture.id);
    GameState->BloodParticlesVB.Draw(GameState->PQuadBuf.data, GameState->PQuadBuf.lenu());
    GLHasErrors();
    glDepthMask(GL_TRUE);

    // PRIMITIVES
    glEnable(GL_BLEND);
    // Blanket disable depth test might be problematic but whatever its debug drawing
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    AppState->PrimitivesRenderer->FlushPrimitives(
        &GameState->ClipFromView, 
        &GameState->ViewFromWorld,
        AppState->RenderTargetGame->depthTexId, 
        vec2((float)AppState->RenderTargetGame->width, 
            (float)AppState->RenderTargetGame->height));
}

