#version 330 core

layout(location = 0) out vec4 FragColor;

in vec2 TexCoords;
in vec3 WorldPos;
in vec3 WorldNormal;

uniform sampler2D ColorTexture;
uniform vec4 MuzzleFlash;

struct model_lighting_data_t
{
    float AmbientCube[6];
    int DoSunLight;
    vec3 DirectionToSun;
    int PointLightsCount;
    vec3 PointLightsPos[4];
    float PointLightsAttLin[4];
    float PointLightsAttQuad[4];
};

uniform model_lighting_data_t ModelLighting;

// world normal -> directional ambient contribution
float AmbientLight(vec3 WorldNormal)
{
    vec3 NSquared = WorldNormal * WorldNormal;
    ivec3 IsNegative = ivec3(WorldNormal.x < 0.0, WorldNormal.y < 0.0, WorldNormal.z < 0.0);
    float LinearColor = NSquared.x * ModelLighting.AmbientCube[IsNegative.x] 
                      + NSquared.y * ModelLighting.AmbientCube[IsNegative.y+2] 
                      + NSquared.z * ModelLighting.AmbientCube[IsNegative.z+4];
    return LinearColor;
}

// world pos normal -> direct light contribution
float DiffuseLight(vec3 WorldPos, vec3 WorldNormal)
{
    /* Half-Lambertian Diffuse Shading Model */

    float DirectLight = 0.f;

    if (ModelLighting.DoSunLight != 0)
    {
        // TODO(Kevin): Intensity multiplier should be configured in map editor
        float SunIntensity = 1.f;
        float Lambertian = dot(ModelLighting.DirectionToSun, WorldNormal);
        float ModifiedLambertian = (Lambertian * 0.5f + 0.5f);
        Lambertian = ModifiedLambertian * ModifiedLambertian;
        if (Lambertian > 0.f)
        {
            DirectLight += Lambertian * SunIntensity;
        }
    }

    for (int i = 0; i < ModelLighting.PointLightsCount; ++i)
    {
        vec3 PointLightPos = ModelLighting.PointLightsPos[i];
        vec3 ToPointLight = PointLightPos - WorldPos;
        float Lambertian = dot(normalize(ToPointLight), WorldNormal);
        float ModifiedLambertian = (Lambertian * 0.5f + 0.5f);
        Lambertian = ModifiedLambertian * ModifiedLambertian;
        if (Lambertian > 0.f)
        {
            float DistToLight = length(ToPointLight);
            float AttLin = ModelLighting.PointLightsAttLin[i];
            float AttQuad = ModelLighting.PointLightsAttQuad[i];
            float Attenuation = 1.f / (1.f + AttLin * DistToLight + AttQuad * DistToLight * DistToLight);
            DirectLight += Lambertian * Attenuation;
        }
    }

    return DirectLight;
}

// albedo texture color -> shaded color
vec3 ShadeDynamicModel(vec3 Color, vec3 WorldPos, vec3 WorldNormal)
{
    WorldNormal = normalize(WorldNormal);
    float AmbientTerm = AmbientLight(WorldNormal);
    float DiffuseTerm = DiffuseLight(WorldPos, WorldNormal);
    // TODO(Kevin): heuristically boost ambient term contribution as a function of direct lights (diffuse term)
    //              so direct light doesn't swamp the ambient and cause harsh appearance
    float Light = AmbientTerm + DiffuseTerm;
    return Color * Light;
}

void main()
{
    vec3 FinalColor = texture(ColorTexture, TexCoords).xyz;
    // FinalColor = vec3(1.0);

    FinalColor = ShadeDynamicModel(FinalColor, WorldPos, normalize(WorldNormal));

    if (MuzzleFlash.w > 0.0)
    {
        vec3 ToLight = MuzzleFlash.xyz - WorldPos;
        float DistToLight = length(ToLight);
        float CosTheta = dot(normalize(ToLight), normalize(WorldNormal));
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
