#version 330

in vec2 uv;

layout(location = 0) out vec4 out_colour;

uniform sampler2D texture0;

void main()
{
    float r = texture(texture0, uv).r;
    vec3 COLOR = vec3(r,r,r);
    out_colour = vec4(COLOR, 1.0);
}
