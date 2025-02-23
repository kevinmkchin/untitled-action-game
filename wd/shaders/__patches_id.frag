#version 330

in vec2 uv1;
in vec2 uv2;

layout(location = 0) out vec4 FragmentColorAndDepth;

uniform sampler2D ColorTexture;
uniform sampler2D LightMap;


void main()
{
    // using ColorTexture so OpenGL doesn't discard the uniform
    FragmentColorAndDepth.rgb = vec3(texture(LightMap, uv2).r, texture(ColorTexture, uv1).g, texture(ColorTexture, uv1).b);

    // I won't need gl_FrontFacing if I use gl_FragCoord.z

    if (gl_FrontFacing) // https://docs.vulkan.org/glsl/latest/chapters/builtins.html
    {
        float near = 1.0; // Near clip plane
        float far = 32000.0; // Far clip plane
        float linearDepth = (2.0 * near * far) / (far + near - gl_FragCoord.z * (far - near));

        FragmentColorAndDepth.g = linearDepth;
        FragmentColorAndDepth.a = 0.0;
    }
    else
    {
        // 2025-01-27
        // GL_BLEND must be disabled since I'm using the alpha channel as a tag
        FragmentColorAndDepth.g = 0.0;
        FragmentColorAndDepth.a = 0.69;
    }
}
