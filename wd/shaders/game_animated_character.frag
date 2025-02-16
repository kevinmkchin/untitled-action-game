#version 330 core

in vec2 TexCoords;

uniform sampler2D ColorTexture;

out vec4 FragColor;

void main()
{    
    FragColor = texture(ColorTexture, TexCoords);
}
