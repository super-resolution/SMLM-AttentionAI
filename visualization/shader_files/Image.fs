#version 450 core
in vec2 TexCoord;

layout(location = 0) out vec4 color;

uniform sampler2D image;
uniform sampler1D cmap;


void main()
{
    float c = texture2D(image, TexCoord).r;
    if (c>.99){
        c=.99;
    }
    color = texture(cmap, c);
}