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
