#version 450 core
layout (location = 1) in vec3 position;
layout (location = 2) in vec2 texCoord;
uniform mat4 u_projection;
uniform mat4 u_modelview;

out vec2 TexCoord;

void main()
{
    gl_Position = u_projection * u_modelview *  vec4(position, 1.0);
    vec2 tex = texCoord;
    TexCoord = tex;
}