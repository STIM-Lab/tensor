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
uniform sampler3D opacity;
uniform float alpha;
uniform vec3 crop_pos;
uniform vec3 crop_wid;

void main() {
	vec3 limit_min = vec3(crop_pos.x - crop_wid.x / 2, crop_pos.y - crop_wid.y / 2, crop_pos.z - crop_wid.z / 2);
	vec3 limit_max = vec3(crop_pos.x + crop_wid.x / 2, crop_pos.y + crop_wid.y / 2, crop_pos.z + crop_wid.z / 2);

	if(texcoord.x < 0 || texcoord.y < 0 || texcoord.z < 0 || texcoord.x > 1 || texcoord.y > 1 || texcoord.z > 1)
		color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	else if (texcoord.x < limit_min.x || texcoord.y < limit_min.y || texcoord.z < limit_min.z || texcoord.x > limit_max.x || texcoord.y > limit_max.y || texcoord.z > limit_max.z) {
		color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	}
	else {
		color = vec4(1.0f, 1.0f, 1.0f, 1.0f) * texture(mapped_volume, vec3(texcoord.x, texcoord.y, texcoord.z));
		color.a = alpha * texture(opacity, vec3(texcoord.x, texcoord.y, texcoord.z)).r;
	}
};
)"