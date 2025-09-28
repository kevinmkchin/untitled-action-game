#version 330

in vec2 uv1;
in vec2 uv2;
in vec3 WorldPos;
in vec3 WorldNormal;

layout(location = 0) out vec4 FragColor;

uniform sampler2D ColorTexture;
uniform sampler2D LightMap;

void main()
{
    vec3 ColorAlbedo = texture(ColorTexture, uv1).rgb;
    ColorAlbedo = vec3(1.0);
    // ColorAlbedo = vec3(0.7);
    float LinearLight = texture(LightMap, uv2).r;
    LinearLight = LinearLight * 1.0;
    vec3 FinalColor = ColorAlbedo * LinearLight;

    FragColor = vec4(FinalColor, 1.0);
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.6)); // gamma correction
}
