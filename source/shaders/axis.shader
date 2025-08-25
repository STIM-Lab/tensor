R"(

# shader vertex
# version 330 core

layout(location = 0) in vec3 v;

uniform mat4 MVP;
uniform vec3 loc;
uniform int axis;
out vec4 FragColor;

void main()
{
    gl_Position = MVP * vec4(v.x, v.y, v.z, 1.0f);
    if (axis == 0)
        FragColor = vec4(1.2 - loc.x, 0.0, 0.0, 1.0);
    if (axis == 1)
        FragColor = vec4(0.0, 1.2 - loc.y, 0.0, 1.0);
    if (axis == 2)
        FragColor = vec4(0.0, 0.0, 1.2 -loc.z, 1.0);
};



# shader fragment
# version 330 core

in vec4 FragColor;
out vec4 color;


void main()
{
    color = FragColor;
};
)"