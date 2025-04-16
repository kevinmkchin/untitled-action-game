#include "renderer.h"
#include "game.h"
#include "enemy.h"
#include "game_assets.h"
#include "shaders.h"
#include "shader_helpers.h"

static GPUShader Sha_ModelSkinnedLit;

void AcquireResources()
{
    GLLoadShaderProgramFromFile(Sha_ModelSkinnedLit, 
        shader_path("model_skinned.vert").c_str(), 
        shader_path("model_textured_skinned.frag").c_str());
}

// these should be generic "draw skinned model here, draw lit model there" functions

void FillSkinnedModelDrawInfo(
    sm_draw_info *DrawInfo,
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
    fixed_array<sm_draw_info> &SMDrawInfos,
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
        sm_draw_info &SMInfo = SMDrawInfos[i];

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
            skinned_mesh_t m = SMInfo.Model->Meshes[j];
            GPUTexture t = SMInfo.Model->Textures[j];

            glActiveTexture(GL_TEXTURE0);
            //glBindTexture(GL_TEXTURE_2D, t.id);
            glBindTexture(GL_TEXTURE_2D, Assets.DefaultEditorTexture.gputex.id);

            glBindVertexArray(m.VAO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.IBO);
            glDrawElements(GL_TRIANGLES, m.IndicesCount, GL_UNSIGNED_INT, nullptr);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
    }
}

