#version 330

in vec2 uv;

layout(location = 0) out vec4 FragColor;

uniform sampler2D texture0;

void main()
{
    float LinearLight = texture(texture0, uv).r;
    vec3 FinalColor = vec3(1.0,1.0,1.0) * LinearLight;
    FragColor = vec4(FinalColor, 1.0);
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/2.2)); // gamma correction
}
