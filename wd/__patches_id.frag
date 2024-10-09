#version 330

in vec2 uv;
in vec3 FragWorldPos;

layout(location = 0) out vec4 FragmentWorldPosAndRadiance;

uniform sampler2D RadianceMap;


void main()
{
    // FragmentWorldPosAndRadiance.rgb = vec3(texture(RadianceMap, uv).r);
    // FragmentWorldPosAndRadiance.a = 1.f;
    FragmentWorldPosAndRadiance.rgb = FragWorldPos;
    if (gl_FrontFacing)
    {
        FragmentWorldPosAndRadiance.a = texture(RadianceMap, uv).r;
    }
    else
    {
        FragmentWorldPosAndRadiance = vec4(0.326, 0.789, 0.982, 0.802);
    }
}
