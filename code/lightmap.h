#pragma once

struct lm_face_t
{
    vec3  *pos = NULL;
    vec3  *norm = NULL;
    vec3  *tangent = NULL;
    // vec3  *patches_id = NULL;
    float *light = NULL;
    float *light_indirect = NULL;
    i32 w = -1;
    i32 h = -1;
};

struct game_map_build_data_t
{
    ByteBuffer Output;

    int TotalFaceCount = 0;
    std::unordered_map<u32, std::vector<float>> VertexBuffers;
    std::vector<vec3> ColliderWorldPoints;
    std::vector<u32> ColliderSpans;
};

void BakeStaticLighting(game_map_build_data_t& BuildData);

