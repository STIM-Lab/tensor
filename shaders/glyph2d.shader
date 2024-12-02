R"(

# shader vertex
# version 330 core

layout(location = 0) in vec3 v;
layout(location = 1) in vec3 n;
layout(location = 2) in vec2 t;

out vec4 vertex_color;

uniform mat4 Mview;
uniform mat4 Mobj;
uniform float scale;
uniform float norm;
uniform sampler2D lambda;
uniform sampler2D evecs;

vec4 rainbow_cycle_cmap(float low, float high, float v){
	float x = mix(0.0f, 6.0f, (v - low) / (high - low));
	vec4 r = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	vec4 g = vec4(0.0f, 1.0f, 0.0f, 1.0f);
	vec4 b = vec4(0.0f, 0.0f, 1.0f, 1.0f);

	if(x <= 1.0f) return mix(r, g, x);
	if(x <= 2.0f) return mix(g, b, x - 1.0f);
	if(x <= 3.0f) return mix(b, r, x - 2.0f);
	if(x <= 4.0f) return mix(r, g, x - 3.0f);
	if(x <= 5.0f) return mix(g, b, x - 4.0f);
	return mix(b, r, x - 5.0f);
}

float eccentricity(float l0, float l1){
	float ratio = pow(l0, 2) / pow(l1, 2);
	if(ratio > 1.0f) return 0.0f;
	if(ratio < 0.0f) return 1.0f;
	return sqrt(1.0f - ratio);	// otherwise calculate the elliptical eccentricity [0, 1]
}

float superquadric(float ecc, float theta, float l0, float l1){
	float gamma = ecc;
	float beta = pow(1.0f - ecc, gamma);
	if(beta < 0.1) beta = 0.1;
	float n = 2.0f / beta;
	float a = 1.0f;
	float b = l0/l1;
	

	float cos_theta = cos(theta);
	float sin_theta = sin(theta);
	float cos_theta_n = pow(abs(cos_theta)/a, n);
	float sin_theta_n = pow(abs(sin_theta)/b, n);
	float r = 0.5 * scale * pow(cos_theta_n + sin_theta_n, -1.0f/n);
	return r;
}

void main() {

	// fetch the eigenvalues
	ivec2 tensor_coord = ivec2(v.x, v.y);
	vec4 l = texelFetch(lambda, tensor_coord, 0);

	//fetch the eigenvectors
	vec4 ev = texelFetch(evecs, tensor_coord, 0);

	// calculate the tensor eccentricity
	float ecc = eccentricity(l.x, l.y);

	// calculate the position on the superquadric
	float sq = superquadric(ecc, t.y - ev.y, l.x, l.y);

	if(norm != 0){
	    sq = sq * l.y / norm;
	}

	// set the vertex position
	vec3 p = v + vec3(sq * cos(t.y), sq * sin(t.y), 0.0);

	// transform
	gl_Position = Mview * Mobj * vec4(p.x, p.y, p.z, 1.0);
	//gl_Position = Mview * Mobj * vec4(v.x, v.y, v.z, 1.0);

	// get the cartesian coordinates of the largest eigenvector
	vec2 ev1 = vec2(cos(ev.y), sin(ev.y));
	vec4 cmap = rainbow_cycle_cmap(-3.14159, 3.14159, ev.y);
	vec4 white = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	vertex_color = mix(white, cmap, ecc);
};

# shader fragment
# version 330 core

layout(location = 0) out vec4 color;

in vec4 vertex_color;


void main() {
	color = vertex_color;
	//color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
};

)"