#version 330

in vec2 uv;

layout(location = 0) out vec4 out_colour;

uniform sampler2D texture0;

void main()
{
    vec3 COLOR = texture(texture0, uv).rgb;
    out_colour = vec4(COLOR, 1.0);
}
