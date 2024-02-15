#version 330
layout (location = 1) in vec2 vPosition;
layout (location = 2) in vec2 vSize;
//todo: add other attributed and filter enuemrator
layout (location = 3) in float frame;
layout (location = 4) in float prob;

uniform mat4 u_projection;
uniform mat4 u_modelview;
uniform float maxEmission;
uniform vec2 precision_filter;
uniform ivec2 frame_filter;
uniform vec2 probability_filter;


out float Emission;
out float u_maxEmission;
const float radius = 20.0;

void main()
{
    //todo: out of view position for filtered vertices
    if ((vSize.x<precision_filter.x)||(vSize.x>precision_filter.y)||(vSize.y<precision_filter.x)||(vSize.y>precision_filter.y)
    ||frame_filter.x>frame || frame_filter.y<frame ||probability_filter.x>prob || probability_filter.y<prob)
        {
        gl_Position = vec4(9999.0);
        gl_PointSize = 0.0;
        }
    else{
        gl_Position = u_projection * u_modelview * vec4(vPosition.xy, 0.0, 1.0);
        Emission = 1.0;
        u_maxEmission = 100.0;
        gl_PointSize = vSize.x*vSize.y * u_projection[1][1] * radius / gl_Position.w;
        }
}