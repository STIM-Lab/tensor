R"(

# shader vertex
# version 330 core

layout(location = 0) in vec3 v;
layout(location = 1) in vec3 n;
layout(location = 2) in vec2 t;

out vec4 texcoord;
uniform mat4 Mview;
uniform mat4 Mobj;
uniform mat4 Mtex;

void main() {

	gl_Position = Mview * Mobj * vec4(v.x, v.y, v.z, 1.0f);
	texcoord = Mtex * vec4(t.x, t.y, 0.0f, 1.0f);
};

# shader fragment
# version 330 core

layout(location = 0) out vec4 color;

in vec4 texcoord;
uniform sampler3D mapped_volume;
uniform float alpha;

void main() {
	color = vec4(1.0f, 1.0f, 1.0f, alpha) * texelFetch(mapped_volume, ivec3(texcoord.x, texcoord.y, texcoord.z), 0);
};
)"