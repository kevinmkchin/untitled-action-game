#version 330

in vec2 uv;

layout(location = 0) out vec3 out_colour;

uniform sampler2D texture0;

void main()
{
    out_colour = texture(texture0, uv).rgb;
}
