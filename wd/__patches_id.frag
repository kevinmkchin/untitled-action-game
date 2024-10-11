#version 330

in vec2 uv;
in vec3 FragWorldPos;

layout(location = 0) out vec4 FragmentColorAndDepth;

uniform sampler2D RadianceMap;


void main()
{
    FragmentColorAndDepth.rgb = vec3(texture(RadianceMap, uv).r);

    if (gl_FrontFacing)
    {
        FragmentColorAndDepth.a = 0.0;
    }
    else
    {
        FragmentColorAndDepth.a = 0.69;
    }
}
