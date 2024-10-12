#version 330

in vec2 uv1;
in vec2 uv2;

layout(location = 0) out vec4 FragmentColorAndDepth;

uniform sampler2D ColorTexture;
uniform sampler2D LightMap;


void main()
{
    // using ColorTexture so OpenGL doesn't discard the uniform
    FragmentColorAndDepth.rgb = vec3(texture(LightMap, uv2).r, texture(ColorTexture, uv1).g, texture(ColorTexture, uv1).b);

    if (gl_FrontFacing)
    {
        FragmentColorAndDepth.a = 0.0;
    }
    else
    {
        FragmentColorAndDepth.a = 0.69;
    }
}
