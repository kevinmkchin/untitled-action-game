#version 330

in vec2 uv1;
in vec2 uv2;

layout(location = 0) out vec4 FragColor;

uniform sampler2D ColorTexture;
uniform sampler2D LightMap; // I need to mark this as texture slot 1 but not sure how

void main()
{
    vec3 ColorAlbedo = texture(ColorTexture, uv1).rgb;
    float LinearLight = texture(LightMap, uv2).r;
    //vec3 FinalColor = vec3(1.0,1.0,1.0) * LinearLight;
    vec3 FinalColor = ColorAlbedo * LinearLight;
    FragColor = vec4(FinalColor, 1.0);
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.2)); // gamma correction
}
