#include "corpses.h"


// TODO(Kevin): Instanced drawing for corpses

void SortAndDrawCorpses(map_info_t &RuntimeMap, fixed_array<corpse_t> &Corpses,
    const mat4 &ProjFromView, const mat4 &ViewFromWorld)
{
    // NOTE(Kevin): We only ever insert into this array so it should remain sorted
    //              with only the new entries from this frame requiring sorting.
    std::sort(Corpses.begin(), Corpses.end());

    // u16 LastSeenID = MT_INVALID;
    // u16 NumModelTypes = 0;
    // fixed_array<fixed_array<corpse_t>>
    // for (u32 i = 0; i < Corpses.lenu(); ++i)
    // {
    //     if (!Corpses[i].CorpseModel)
    //         break;
    //     if (Corpses[i].CorpseModel.MT_ID != LastSeenID)
    //     {
    //         ++NumModelTypes;
    //         LastSeedID = Corpses[i].CorpseModel.MT_ID;
    //     }
    //     else
    // }

    // ok we see first model, populate instance matrices and lighting data
    // when the model changes, flush then repeat with new model

    //fixed_array<mat4> InstanceModelMatrices

    UseShader(GameModelTexturedShader);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(GameModelTexturedShader, "MuzzleFlash", 
        Player.Weapon.MuzzleFlash.x, 
        Player.Weapon.MuzzleFlash.y, 
        Player.Weapon.MuzzleFlash.z, 
        Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(GameModelTexturedShader, "ProjFromView", 1, ProjFromView.ptr());
    GLBindMatrix4fv(GameModelTexturedShader, "ViewFromWorld", 1, ViewFromWorld.ptr());

    for (size_t i = 0; i < Corpses.length; ++i)
    {
        // CENTROID instead of root
        BindUniformsForModelLighting(GameModelTexturedShader, RuntimeMapInfo, Corpses[i].Pos);
        mat4 ModelMatrix = TranslationMatrix(Corpses[i].Pos) * 
            RotationMatrix(Corpses[i].Rot) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);
        GLBindMatrix4fv(GameModelTexturedShader, "WorldFromModel", 1, ModelMatrix.ptr());
        RenderModelGLTF(*Corpses[i].CorpseModel);
    }
}


