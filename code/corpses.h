#pragma once

// corpses are simply instanced static models that are textured and lit with no collision

struct corpse_t
{
    vec3 Pos;
    quat Rot;

    size_t LightCacheIndex;

    ModelGLTF *CorpseModel;

    bool operator<(const corpse_t &Other) const
    {
        if (!CorpseModel)
            return false;
        if (!Other.CorpseModel)
            return true;
        return CorpseModel->MT_ID < Other.CorpseModel->MT_ID;
    }
};

void SortAndDrawCorpses(map_info_t &RuntimeMap, fixed_array<corpse_t> &Corpses, 
    const mat4 &ProjFromView, const mat4 &ViewFromWorld);

