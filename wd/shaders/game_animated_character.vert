#version 330 core

layout(location = 0) in vec3 Pos;
layout(location = 1) in vec2 Tex;
layout(location = 2) in vec3 Norm;
layout(location = 5) in ivec4 BoneIds; 
layout(location = 6) in vec4 Weights;
    
uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;
    
const int MAX_BONES = 100;
const int MAX_BONE_INFLUENCE = 4;
uniform mat4 FinalBonesMatrices[MAX_BONES];
    
out vec2 TexCoords;
    
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
        
    mat4 ViewModel = View * Model;
    gl_Position =  Projection * ViewModel * TotalPosition;
    TexCoords = Tex;
}
