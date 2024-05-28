R"(

# shader vertex
# version 330 core

layout(location = 0) in vec3 v;

out vec4 vertex_color;

uniform float tx;
uniform float ty;
uniform sampler2D tensorfield;
uniform mat4 MVP;
uniform float maxnorm;
uniform float scale = 0.3;
float epsilon = 0.1;

/*
	eccentricity = sqrt(a^2 - b^2) for ellipse use this for n
	axis of rotation is z
*/

// returns r for a given theta and eigenvector
mat4 rot(float theta){

	return mat4(cos(theta), sin(theta), 0, 0,
		        -sin(theta), cos(theta),  0, 0,
		        0         , 0         ,  1, 0,
		        0         , 0         ,  0, 1);

}

// Calculate the i-th eigenvalue of a symmetric 2D tensor T
// T is the tensor represented as a vec4 instead of a mat2, i is the flag to return the lambda you want
float EigenvalueSym2D(vec4 T, int i) {
    float d = T.x;
    float e = T.y;
    float f = e;
    float g = T.w;

    float dpg = d + g;
    float disc = sqrt((4 * e * f) + pow(d - g, 2));
    float a = (dpg + disc) / 2.0f;
    float b = (dpg - disc) / 2.0f;

    if (i == 0) return min(a, b);
    else return max(a, b);
}

// Calculate the eigenvector of a symmetric 2D tensor T given its eigenvalue ev
vec2 EigenvectorSym2D(vec4 T, float lambda) {

    float a = T.x;
    float b = T.y;
    //float c = b;
    float d = T.w;
   
    if(b != 0)
        return normalize(vec2(lambda - d, b));
    else if (lambda == 0)
        return vec2(1.0f, 0.0f);
    else
        return vec2(0.0f, 1.0f);
}

void main() {


	vec4 T = texture(tensorfield, vec2(tx, ty));			// get the tensor encoded in the texture
	float lambda0 = EigenvalueSym2D(T, 0);					// get the largest eigenvalue
	float lambda1 = EigenvalueSym2D(T, 1);					// get the smallest eigenvalue
	vec2 e1 = EigenvectorSym2D(T, lambda1);					// get the largest eigenvector

	float norm = maxnorm;									// scale the eigenvalues for display
	if(maxnorm == 0) norm = lambda1;
	float l0, l1;
	if(norm == 0){
		l0 = 0;
		l1 = 0;
	}
	else{
		l0 = lambda0 / norm;
		l1 = lambda1 / norm;
	}
	if(l0 < epsilon) l0 = epsilon;

	float x, y;									// create coordinates to store the new (x,y) value of the vertex
	if (v.x == 0.0f && v.y == 0.0f) {			// keep the center vertex at the center (prevent dividing by 0)
		x = 0.0f;
		y = 0.0f;
	}
	else {

		float theta = atan(v.y, v.x);			// calculate the theta value for the vertex (in polar coords)

		float cos_theta = cos(theta);			// calculate the sine and cosine to for glyph shape
		float sin_theta = sin(theta);

												// eccentricity is 0 for an ellipse, 1 for a line
		float ecc;
		float ratio = pow(l0, 2)/ pow(l1, 2);
		ecc = sqrt(1.0f - ratio);	// otherwise calculate the elliptical eccentricity [0, 1]

		
		// here we use superquadrics to calculate the glyph shape based on its eccentricity
		///////////////////////////////////////////////////////////////////////////////////
		float n = 1 - pow(ecc, 3);


		float cos_theta_n = pow(abs(cos_theta), n) * sign(cos_theta);
		float sin_theta_n = pow(abs(sin_theta), n) * sign(sin_theta);
		x = cos_theta_n * l1;
		//if(l1 < epsilon)
		//	l1 = epsilon;
		y = sin_theta_n * l0;
		///////////////////////////////////////////////////////////////////////////////////
		// now we have the new (x, y) coordinates for the vertex

	}
	
	gl_Position = MVP * rot(atan(e1.y, e1.x)) * vec4(x * scale, y * scale, v.z, 1.0f);
	vertex_color = vec4(abs(e1.x), abs(e1.y), 0.0f, 1.0f);

};

# shader fragment
# version 330 core

layout(location = 0) out vec4 color;

in vec4 vertex_color;


void main() {
	color = vertex_color;
};

)"