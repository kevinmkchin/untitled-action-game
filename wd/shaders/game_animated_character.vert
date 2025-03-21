#version 330 core

layout(location = 0) in vec3 Pos;
layout(location = 1) in vec2 Tex;
layout(location = 2) in vec3 Norm;
layout(location = 3) in ivec4 BoneIds; 
layout(location = 4) in vec4 Weights;

uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;

const int MAX_BONES = 64;
const int MAX_BONE_INFLUENCE = 4;
uniform mat4 FinalBonesMatrices[MAX_BONES];

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 WorldNormal;

void main()
{
    vec4 TotalPosition = vec4(0.0f);
    for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    {
        if(BoneIds[i] == -1) 
            continue;
        if(BoneIds[i] >= MAX_BONES) 
        {
            TotalPosition = vec4(Pos,1.0f);
            break;
        }
        vec4 LocalPosition = FinalBonesMatrices[BoneIds[i]] * vec4(Pos,1.0f);
        TotalPosition += LocalPosition * Weights[i];
        vec3 LocalNormal = mat3(FinalBonesMatrices[BoneIds[i]]) * Norm;
    }

    WorldPos = (Model * TotalPosition).xyz;
    WorldNormal = transpose(inverse(mat3(Model))) * Norm;

    mat4 ViewModel = View * Model;
    gl_Position =  Projection * ViewModel * TotalPosition;
    TexCoords = Tex;
}
