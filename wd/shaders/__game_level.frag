#version 330

in vec2 uv1;
in vec2 uv2;
in vec3 WorldPos;
in vec3 WorldNormal;

layout(location = 0) out vec4 FragColor;

uniform sampler2D ColorTexture;
uniform sampler2D LightMap;
uniform vec4 MuzzleFlash;

void main()
{
    vec3 ColorAlbedo = texture(ColorTexture, uv1).rgb;
    float LinearLight = texture(LightMap, uv2).r;
    vec3 FinalColor = ColorAlbedo * LinearLight;

    if (MuzzleFlash.w > 0.0)
    {
        vec3 ToLight = MuzzleFlash.xyz - WorldPos;
        float DistToLight = length(ToLight);
        float CosTheta = dot(normalize(ToLight), WorldNormal);
        if (CosTheta > 0.f)
        {
            float MuzzleIntensity = CosTheta * (2.0 /
                (1.0 + 0.004 * DistToLight + 0.0001 * DistToLight * DistToLight));
            FinalColor = FinalColor * vec3(1.0 + MuzzleIntensity * 2.0, 
                1.0 + MuzzleIntensity, 1.0 + MuzzleIntensity * 0.3);
        }
    }

    FragColor = vec4(FinalColor, 1.0);
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.6)); // gamma correction
}
