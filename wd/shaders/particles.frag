#version 450

layout(location = 0) out vec4 FragColor;

in vec4 Color;
in vec2 UV;

uniform sampler2D ParticleSpriteAtlas;

void main()
{
    vec4 SpriteColor = texture(ParticleSpriteAtlas, UV);
    vec4 FinalColor = SpriteColor * Color;
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.6));
    FragColor = FinalColor;
}
