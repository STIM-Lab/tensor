R"(

# shader vertex
# version 330 core

layout(location = 0) in vec3 v;
layout(location = 1) in vec3 n;
layout(location = 2) in vec2 t;

out vec2 vertex_texcoord;
uniform mat4 Mview;
uniform mat4 Mobj;

void main() {

	gl_Position = Mview * Mobj * vec4(v.x, v.y, v.z, 1.0f);
	vertex_texcoord = t;
};

# shader fragment
# version 330 core

layout(location = 0) out vec4 color;

in vec2 vertex_texcoord;

uniform float maxval;
uniform float minval;
uniform sampler2D mapped_image;

void main() {
	//color = vec4(1.0, 0.0, 0.0, 1.0);
	//color = (texture(scalar, vertex_texcoord) - minval) / (maxval - minval);
	color = texture(mapped_image, vertex_texcoord);
};
)"