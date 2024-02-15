#version 330
in float Emission;
in float u_maxEmission;
//out vec4 Color;
layout(location = 0) out vec4 Color;



void main()
{
    vec2 coord = gl_PointCoord-vec2(0.5);
    float l = length(coord);
    if (l > .5)
        discard;
    float pos = pow(2.71, (-pow(l/0.5,2.0)));
    //sigma is in primitive coords -> does not change only point size changes
    //todo: strength is not used atm and has to be implemented into the network
    float strength = sqrt(Emission/u_maxEmission);
    if (strength>1.0)
        strength = 1.0;
    Color = vec4(pos)*vec4(vec3(strength), 1.0);
}