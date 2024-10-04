#version 330

in vec2 uv;

layout(location = 0) out vec3 FragColor;

uniform sampler2D texture0;

void main()
{
    if (gl_FrontFacing)
    {
        FragColor = texture(texture0, uv).rgb;
    }
    else
    {
        FragColor = vec3(1.0); // TAGGING BACKFACE AS WHITE TO BE ABLE TO DETECT IT WHEN READ BACK
    }
}
