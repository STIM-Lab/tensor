# shader vertex
# version 330 core


layout(location = 0) in vec4 aPos;
layout(location = 2) in vec3 texcoords;

uniform mat4 MVP;
uniform float slider;
//uniform int axis;


out vec3 vertex_tex;

void main()
{
	gl_Position = MVP * aPos;
	int axis = 2;
	// based on eye, choose the vertex position
	if (axis == 2) {
		vertex_tex = vec3(texcoords.x, texcoords.y, slider);
	}
	/*else if (axis == 1)
	{
		vertex_tex = vec3(texcoords.x, slider, texcoords.y);
	}
	else
	{
		vertex_tex = vec3(slider, texcoords.x, texcoords.y);
	}*/
};




# shader fragment
# version 330 core

in vec3 vertex_tex;
uniform float opacity;
out vec4 colors;

uniform sampler3D volumeTexture;

void main()
{
	colors = vec4(1.0f, 1.0f, 1.0f, opacity) * texture(volumeTexture, vertex_tex);
};