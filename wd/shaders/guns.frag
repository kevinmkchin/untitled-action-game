#version 330

layout(location = 0) out vec4 FragColor;

uniform sampler2D ColorTexture;

in vec3 Light;

void main()
{
    vec3 Color = vec3(0.2, 0.2, 0.2);
    FragColor = vec4(Color * Light, 1.0);
    // vec3 ColorAlbedo = texture(ColorTexture, uv1).rgb;
    // float LinearLight = texture(LightMap, uv2).r;
    //vec3 FinalColor = vec3(1.0,1.0,1.0) * LinearLight;
    // vec3 FinalColor = ColorAlbedo * LinearLight;
    // FragColor = vec4(FinalColor, 1.0);
    // FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.6)); // gamma correction
}
