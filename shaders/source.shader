# shader vertex
# version 330 core

layout(location = 0) in vec3 V;

const float PI = 3.14159265358979323846;
const float epsilon = 0.0000001;

uniform mat4 MV;
uniform mat4 Mtran;

uniform int ColorComponent;
uniform int size;
uniform float gamma;
uniform int anisotropy;
uniform uvec3 position;
uniform float filter;
uniform float thresh;

uniform sampler3D Diagonal;
uniform sampler3D Upper_trian;

out vec4 vertexColor;
out vec3 vertexNorm;

float signpow(float x, float exponent) {
	if (x < 0) return -pow(abs(x), exponent);
	else return pow(abs(x), exponent);
}

vec3 sq_vertex(float alpha, float beta, float theta, float phi) {

	float cos_phi = cos(phi);
	float sin_theta = sin(theta);
	float sin_phi = sin(phi);
	float cos_theta = cos(theta);

	float x = signpow(cos_phi, beta);
	float y = -signpow(sin_theta, alpha) * signpow(sin_phi, beta);
	float z = signpow(cos_theta, alpha) * signpow(sin_phi, beta);
	return vec3(x, y, z);
}

vec3 ComputeEigenvalues(mat3 A) {
    vec3 l;
	float a, b, c, d, e, f, g, h, i;
	a = A[0][0]; b = A[0][1]; c = A[0][2];
	d = A[1][0]; e = A[1][1]; f = A[1][2];
	g = A[2][0]; h = A[2][1]; i = A[2][2];


    // Case: matrix is diagonal
	float p1 = b * b + c * c + f * f;
    if (p1 == 0.0) {
        l[0] = a;
        l[1] = e;
        l[2] = i;
		return l;
    } 

	float q = (a + e + i) / 3.0;
	float p2 = (a - q) * (a - q) + (e - q) * (e - q) + (i - q) * (i - q) + 2.0 * p1;
	float p = sqrt(p2 / 6.0);
	mat3 B = (1.0 / p) * mat3(vec3(a - q, d, g), vec3(b, e - q, h), vec3(c, f, i - q));
	float r = determinant(B) / 2.0;

    // In exact arithmetic for a symmetric matrix - 1 <= r <= 1
    // but computation error can leave it slightly outside this range.

	float phi = 0.0;
    if (r <= -1.0)
        phi = PI / 3.0;
    else if (r > 1.0)
        phi = 0.0;
    else
        phi = acos(r) / 3.0;

    // The eigenvalues satisfy l[0] <= l[1] <= l[2]
    l[2] = q + 2.0 * p * cos(phi);
    l[0] = q + 2.0 * p * cos(phi + (2.0 * PI / 3.0));
    l[1] = 3.0 * q - l[2] - l[0];  // since trace(A) = eig1 + eig2 + eig3

    return l;
}

vec3 ComputeEigenvector(mat3 matrix, float eigvalue) {
	vec3 eigvec;
	float a, b, c, d, e, f, g, h, i;
	a = matrix[0][0]; b = matrix[0][1]; c = matrix[0][2];
	d = matrix[1][0]; e = matrix[1][1]; f = matrix[1][2];
	g = matrix[2][0]; h = matrix[2][1]; i = matrix[2][2];

	// rows of (A - lambda*I)
	// all the rows multiplied by the eigenvector yield zero vector => eigenvector is prependicular to at least two of the rows
	vec3 row0 = vec3(a - eigvalue, b, c);
	vec3 row1 = vec3(d, e - eigvalue, f);
	vec3 row2 = vec3(g, h, i - eigvalue);

	// calculate the cross-product of each two rows
	// v is parallel to the cross product of two of these rows
	vec3 r0xr1 = vec3(row0[1] * row1[2] - row0[2] * row1[1],
		row0[2] * row1[0] - row0[0] * row1[2],
		row0[0] * row1[1] - row0[1] * row1[0]);

	vec3 r0xr2 = vec3(row0[1] * row2[2] - row0[2] * row2[1],
		row0[2] * row2[0] - row0[0] * row2[2],
		row0[0] * row2[1] - row0[1] * row2[0]);

	vec3 r1xr2 = vec3(row1[1] * row2[2] - row1[2] * row2[1],
		row1[2] * row2[0] - row1[0] * row2[2],
		row1[0] * row2[1] - row1[1] * row2[0]);

	// dot product - to find out which cross-product has the largest length
	float d0 = r0xr1[0] * r0xr1[0] + r0xr1[1] * r0xr1[1] + r0xr1[2] * r0xr1[2];
	float d1 = r0xr2[0] * r0xr2[0] + r0xr2[1] * r0xr2[1] + r0xr2[2] * r0xr2[2];
	float d2 = r1xr2[0] * r1xr2[0] + r1xr2[1] * r1xr2[1] + r1xr2[2] * r1xr2[2];
	int imax = 0;
	float dmax = d0;

	if (d1 > dmax) {
		dmax = d1;
		imax = 1;
	}
	if (d2 > dmax)
		imax = 2;

	if (imax == 0.0) {
		eigvec = vec3(r0xr1[0] / sqrt(d0), r0xr1[1] / sqrt(d0), r0xr1[2] / sqrt(d0));
	}
	else if (imax == 1.0) {
		eigvec = vec3(r0xr2[0] / sqrt(d1), r0xr2[1] / sqrt(d1), r0xr2[2] / sqrt(d1));
	}
	else {
		eigvec = vec3(r1xr2[0] / sqrt(d2), r1xr2[1] / sqrt(d2), r1xr2[2] / sqrt(d2));
	}

	return eigvec;
};

void main() {
	
	// Find the texture coordinate
	vec3 Coords = vec3(position) / vec3(textureSize(Diagonal, 0));
	vec3 DiagValue = texture(Diagonal, Coords).rgb;
	vec3 OffDiag = texture(Upper_trian, Coords).rgb;

	// Take the tensor field values from Diagonal and Upper traingular texture maps
	mat3 tensor;
	tensor[0][0] = DiagValue[0]; 	tensor[0][1] = OffDiag[0]; 		tensor[0][2] = OffDiag[1];
	tensor[1][0] = tensor[0][1]; 	tensor[1][1] = DiagValue[1]; 	tensor[1][2] = OffDiag[2];
	tensor[2][0] = tensor[0][2];	tensor[2][1] = tensor[1][2];	tensor[2][2] = DiagValue[2];

	// Compute eigenvalues and eigenvectors using the 3x3 matrix of tensor field
	vec3 eigvals = ComputeEigenvalues(tensor);
	mat3 eigvecs;

	float a, b, c, d, e, f, g, h, i;
	a = tensor[0][0]; b = tensor[0][1]; c = tensor[0][2];
	d = tensor[1][0]; e = tensor[1][1]; f = tensor[1][2];
	g = tensor[2][0]; h = tensor[2][1]; i = tensor[2][2];

	// Case 1: input matrix is diagonal
	if (b * b + c * c + f * f == 0.0) {
		eigvecs[0] = vec3(1.0, 0.0, 0.0);
		eigvecs[1] = vec3(0.0, 1.0, 0.0);
		eigvecs[2] = vec3(0.0, 0.0, 1.0);
	}

	// Case 2: two identical eigenvalues
	else if (abs(eigvals[0] - eigvals[1]) < epsilon) {
		eigvecs[2] = ComputeEigenvector(tensor, eigvals[2]);
		float invLength;
		float w0, w1, w2;
		w0 = eigvecs[2][0];
		w1 = eigvecs[2][1];
		w2 = eigvecs[2][2];
		if (abs(w0) > abs(w1)) {
			invLength = 1.0 / (sqrt(w0 * w0 + w2 * w2));
			eigvecs[0] = vec3(-w2 * invLength, 0.0, w0 * invLength);
		}
		else {
			invLength = 1.0 / (sqrt(w1 * w1 + w2 * w2));
			eigvecs[0] = vec3(0.0, w2 * invLength, -w1 * invLength);
		}
		float u0, u1, u2;
		u0 = eigvecs[0][0];
		u1 = eigvecs[0][1];
		u2 = eigvecs[0][2];
		eigvecs[1] = vec3(w1 * u2 - w2 * u1, w2 * u0 - w0 * u2, w0 * u1 - w1 * u0);
	}
	else {
		eigvecs[0] = ComputeEigenvector(tensor, eigvals[0]);
		eigvecs[1] = ComputeEigenvector(tensor, eigvals[1]);
		eigvecs[2] = ComputeEigenvector(tensor, eigvals[2]);
	}
	
	// Sort the eigenvalues and eigenvectors from largest to smallest (L0 = largest, L2 = smallest)
	float temp0 = eigvals[0];
	vec3 tempv0 = eigvecs[0];
	if (eigvals[0] < eigvals[1]) {
		eigvals[0] = eigvals[1];
		eigvals[1] = temp0;
		eigvecs[0] = eigvecs[1];
		eigvecs[1] = tempv0;
	}
	float temp2 = eigvals[2];
	vec3 tempv2 = eigvecs[2];
	if (eigvals[0] < eigvals[2]) {
		eigvals[2] = eigvals[0];
		eigvals[0] = temp2;
		eigvecs[2] = eigvecs[0];
		eigvecs[0] = tempv2;
	}
	float temp1 = eigvals[1];
	vec3 tempv1 = eigvecs[1];
	if (eigvals[1] < eigvals[2]) {
		eigvals[1] = eigvals[2];
		eigvals[2] = temp1;
		eigvecs[1] = eigvecs[2];
		eigvecs[2] = tempv1;
	}

	// Create Model matrix from translation and rotation matrices
	mat4 Mrot = mat4(eigvecs);
	Mrot[3][3] = 1.0;

	mat4 ModelMat = Mtran * Mrot;

	vec3 evals = normalize(eigvals) * 0.5 * size;
	float l0 = evals[0];
	float l1 = evals[1];
	float l2 = evals[2];

	float suml = l0 + l1 + l2;
	float Cl = (l0 - l1) / suml;
	float Cp = 2 * (l1 - l2) / suml;
	float Cs = 3 * l2 / suml;
		
	float x = V.x;
	float y = V.y;
	float z = V.z;

	float theta = atan(y, x);
	float phi = atan(sqrt(x * x + y * y), z);

	vec3 sq_v, sq_n;

	if (Cl >= Cp) {
		float alpha = pow(1 - Cp, gamma);
		float beta = pow(1 - Cl, gamma);
		sq_v = sq_vertex(alpha, beta, theta, phi);
		sq_n = sq_vertex(2.0f - alpha, 2.0f - beta, theta, phi);
	}
	else {
		float alpha = pow(1 - Cl, gamma);
		float beta = pow(1 - Cp, gamma);
		sq_v = sq_vertex(alpha, beta, theta, phi);
		sq_v = sq_v.zyx;
		sq_v.y = -sq_v.y;
		sq_n = sq_vertex(2.0f - alpha, 2.0f - beta, theta, phi);
		sq_n = sq_n.zyx;
		sq_n.y = -sq_n.y;					
	}

	sq_v = vec3(l0 * sq_v.x, l1 * sq_v.y, l2 * sq_v.z);

	float sx = 1.0f / l0;
	float sy = 1.0f / l1;
	float sz = 1.0f / l2;

	sq_n = normalize(vec3(sx * sq_n.x, sy * sq_n.y, sz * sq_n.z));
		
	mat3 NormMat = transpose(inverse(mat3(Mtran * Mrot)));
	vertexNorm = NormMat * sq_n;

	/////////////// DR.MAYERICH'S CODE ENDED /////////////////////////
	//////////////////////////////////////////////////////////////////

	// Calculating and separating by anisotropy

	float anisotropy_value;
	if (anisotropy == 1) anisotropy_value = (l0 - l1) / (l0 + l1 + l2);						// linear
	if (anisotropy == 2) anisotropy_value = 2 * (l1 - l2) / (l0 + l1 + l2);					// Planar
	if (anisotropy == 3) anisotropy_value = 3 * l2 / (l0 + l1 + l2);						// Spherical

	// 1st condition: only draw tensor with anisotropy value above a filter value
	// 2nd condition: if the option to show all tensors is selected
	// 3rd condition: only draw tensor with their largest eigenvalue above a certain threshold

	if (anisotropy_value > filter || anisotropy == 0)
		gl_Position = MV * ModelMat * vec4(sq_v, 1.0);
	else
		gl_Position = vec4(0.0, 0.0, 0.0, 0.0);

	if(eigvals[0] < thresh)
		gl_Position = vec4(0.0, 0.0, 0.0, 0.0);

	//vertexColor = vec4(1.0, 0.0, 0.0, 1.0);
	vec3 dirColor;
	vec3 glyphColor;
	if(ColorComponent == 0){
		dirColor = vec3(abs(ModelMat[0][0]), abs(ModelMat[0][1]), abs(ModelMat[0][2]));
		glyphColor = Cp + Cs + (1.0f - Cp - Cs) * dirColor;
	}
	if(ColorComponent == 2){
		dirColor = vec3(abs(ModelMat[2][0]), abs(ModelMat[2][1]), abs(ModelMat[2][2]));
		glyphColor = Cl + Cs + (1.0f - Cl - Cs) * dirColor;
	}
	vertexColor = vec4(glyphColor, 1.0f);
};


# shader fragment
#version 330 core

uniform vec4 light0;
uniform vec4 light1;
uniform float ambient;
out vec4 FragColor;
in vec4 vertexColor;
in vec3 vertexNorm;

void main() {
	float l0 = max(0, dot(vertexNorm, normalize(vec3(light0)))) * light0.a;
	float l1 = max(0, dot(vertexNorm, normalize(vec3(light1)))) * light1.a;
	float l = min(l0 + l1 + ambient, 1.0);
	FragColor = vertexColor * (l0 + l1 + ambient);
	//FragColor = vec4(abs(vertexNorm) * l, 1.0);
};