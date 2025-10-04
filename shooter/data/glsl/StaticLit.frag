#version 430

// TODO(Kevin): Can we make this shader branchless? e.g. Lambertian > 0.f could be a min clamp to 0 perhaps

layout(location = 0) out vec4 FragColor;

struct model_instance_data_t
{
    mat4 WorldFromModel;
    vec4 PointLightsPos[4];
    float AmbientCube[6];
    int DoSunLight;
    int PointLightsCount;
    float PointLightsAttLin[4];
    float PointLightsAttQuad[4];
    float _padding_[4];
};

layout(std430, binding = 0) buffer InstanceDataSSBO 
{
    model_instance_data_t Instances[];
};

in flat int InstanceIndex;
in vec2 UV0;
in vec2 UV1;
in vec3 WorldPos;
in vec3 WorldNormal;

uniform int BaseInstanceIndex;
uniform sampler2D ColorTexture;
uniform sampler2D LightmapTexture;
uniform vec4 MuzzleFlash;
uniform vec3 DirectionToSun;

// world normal -> directional ambient contribution
float AmbientLight(vec3 WorldNormal)
{
    /*

        Ambient Cube

    */

    vec3 NSquared = WorldNormal * WorldNormal;
    ivec3 IsNegative = ivec3(WorldNormal.x < 0.0, WorldNormal.y < 0.0, WorldNormal.z < 0.0);
    float LinearColor = NSquared.x * Instances[BaseInstanceIndex + InstanceIndex].AmbientCube[IsNegative.x] 
                      + NSquared.y * Instances[BaseInstanceIndex + InstanceIndex].AmbientCube[IsNegative.y+2] 
                      + NSquared.z * Instances[BaseInstanceIndex + InstanceIndex].AmbientCube[IsNegative.z+4];
    return LinearColor;
}

// world pos normal -> direct light contribution
float DiffuseLight(vec3 WorldPos, vec3 WorldNormal)
{
    /*

        Half-Lambertian Diffuse

        Softens diffuse contribution from light sources to prevent backsides of characters
        and props from looking too flat. Purely non-physical but pleasing to the eye.

    */

    float DirectLight = 0.f;

    if (Instances[BaseInstanceIndex + InstanceIndex].DoSunLight != 0)
    {
        // TODO(Kevin): Intensity multiplier should be configured in map editor
        float SunIntensity = 1.f;
        float Lambertian = dot(DirectionToSun, WorldNormal);
        float ModifiedLambertian = (Lambertian * 0.5f + 0.5f);
        Lambertian = ModifiedLambertian * ModifiedLambertian;
        if (Lambertian > 0.f)
        {
            DirectLight += Lambertian * SunIntensity;
        }
    }

    for (int i = 0; i < Instances[BaseInstanceIndex + InstanceIndex].PointLightsCount; ++i)
    {
        vec3 PointLightPos = Instances[BaseInstanceIndex + InstanceIndex].PointLightsPos[i].xyz;
        vec3 ToPointLight = PointLightPos - WorldPos;
        float Lambertian = dot(normalize(ToPointLight), WorldNormal);
        float ModifiedLambertian = (Lambertian * 0.5f + 0.5f);
        Lambertian = ModifiedLambertian * ModifiedLambertian;
        if (Lambertian > 0.f)
        {
            float DistToLight = length(ToPointLight);
            float AttLin = Instances[BaseInstanceIndex + InstanceIndex].PointLightsAttLin[i];
            float AttQuad = Instances[BaseInstanceIndex + InstanceIndex].PointLightsAttQuad[i];
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
    vec3 FinalColor = texture(ColorTexture, UV0).xyz;
    float LightmapValue = texture(LightmapTexture, UV1).r;

    FinalColor = ShadeDynamicModel(FinalColor, WorldPos, normalize(WorldNormal));
    FinalColor *= LightmapValue;

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
