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

void RenderEnemies(
    fixed_array<enemy_t> &Enemies,
    game_state *GameState,
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

    for (u32 i = 0; i < Enemies.lenu(); ++i)
    {
        enemy_t& Enemy = Enemies[i];
        if (!(Enemy.Flags & EnemyFlag_Active))
            continue;

        // TODO(Kevin): should use centroid instead of root
        BindUniformsForModelLighting(Sha_ModelSkinnedLit, GameState, Enemy.Position);

        mat4 ModelMatrix = TranslationMatrix(Enemy.Position) * 
            RotationMatrix(Enemy.Orientation) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);

        GLBindMatrix4fv(Sha_ModelSkinnedLit, "Model", 1, ModelMatrix.ptr());

        GLBindMatrix4fv(Sha_ModelSkinnedLit, "FinalBonesMatrices[0]", MAX_BONES, 
            Enemy.Animator->SkinningMatrixPalette[0].ptr());

        for (size_t i = 0; i < Assets.Model_Attacker->Meshes.length; ++i)
        {
            skinned_mesh_t m = Assets.Model_Attacker->Meshes[i];
            GPUTexture t = Assets.Model_Attacker->Textures[i];

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

