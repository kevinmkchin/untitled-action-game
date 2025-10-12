#version 330 core

in vec2 FragUV;
in vec4 FragColorMultiplier;

out vec4 FragColor;

uniform sampler2D TextureSampler0;

void main()
{
    vec4 TextureColor = texture(TextureSampler0, FragUV);
    FragColor = TextureColor * FragColorMultiplier;
}
