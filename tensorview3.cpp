#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <numbers>

#include "tira/graphics_gl.h"
#include "tira/volume.h"
#include "gui.h"
#include <glm/gtc/quaternion.hpp>
#include <boost/program_options.hpp>
#include <complex>
#include <Eigen/Eigenvalues>
#define PI 3.14159265358979323846

GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
tira::camera camera;

bool perspective = false;
float zoom = 1.0f;
float right = 0.0f;
float up = 0.0f;
bool ctrl = false;
bool dragging = false;
double xprev, yprev;

tira::volume<glm::mat3> T;
tira::volume<float> lambda;
tira::volume<glm::mat3> eigenvectors;
float epsilon = 0.001;

int zi = 0;

// input variables for arguments
std::string in_filename;
float in_gamma;
int in_cmap;

void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && window_focused)
		dragging = true;
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		dragging = false;
	glfwGetCursorPos(window, &xprev, &yprev);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
	double ANGLE_SCALE = 0.005;
	if (dragging) {
		double dx = xpos - xprev;
		double dy = ypos - yprev;
		camera.OrbitFocus(-dx * ANGLE_SCALE, dy * ANGLE_SCALE);
		xprev = xpos;
		yprev = ypos;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	zi += yoffset;
	if (zi < 0) zi = 0;
	if (zi >= T.Z()) zi = T.Z() - 1;
	std::cout << zi << std::endl;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		perspective = !perspective;
	}
	if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS) {
		ctrl = true;
		std::cout << "true" << std::endl;
	}
	if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE) {
		ctrl = false;
		std::cout << "false" << std::endl;
	}
	if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS && ctrl) {
		zoom -= 0.1;
		if (zoom < 0.1) zoom = 0.1;
		std::cout << "zoom=" << zoom << std::endl;
	}
	if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS && ctrl) {
		zoom += 0.1;
		if (zoom > 2) zoom = 2;
		std::cout << "zoom=" << zoom << std::endl;
	}
	if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
		up += 5;
		if (up > 100) up = 100;
		std::cout << "up=" << up << std::endl;
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		up -= 5;
		if (up < -100) up = -100;
		std::cout << "up=" << up << std::endl;
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		right += 5;
		if (right > 100) right = 100;
		std::cout << "right=" << right << std::endl;
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		right -= 5;
		if (right < -100) right = -100;
		std::cout << "right=" << right << std::endl;
	}
}

void resetPlane(float frame) {
	camera.setPosition(frame / 2.0f, frame / 2.0f, frame);
	camera.LookAt(frame / 2.0f, frame / 2.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	right = 0.0f;
	up = 0.0f;
	zoom = 1.0f;
	zi = 0;
}

std::string vertex_shader_source = R"(
	#version 330 core
	layout(location = 0) in vec3 V;
	layout(location = 1) in vec3 N;
	uniform mat4 ModelMat;
	uniform mat4 ViewMat;
	uniform mat4 ProjMat;
	uniform vec3 lambda;
	uniform float gamma;
	uniform int ColorComponent;
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


	void main() {

		float l0 = lambda[0];
		float l1 = lambda[1];
		float l2 = lambda[2];

		// calculate the linear and planar anisotropy
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
		
		//mat4 VM = ViewMat * ModelMat;
		mat3 NormMat = transpose(inverse(mat3(ModelMat)));
		vertexNorm = NormMat * sq_n;

		gl_Position = ProjMat * ViewMat * ModelMat * vec4(sq_v, 1.0);
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
)";

std::string fragment_shader_source = R"(
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
)";

GLFWwindow* InitGLFW() {
	GLFWwindow* window;

	// Setup window
	glfwSetErrorCallback(glfw_error_callback);


	if (!glfwInit())
		return NULL;

	// GL 3.0 + GLSL 130
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	// Create window with graphics context
	window = glfwCreateWindow(1600, 1200, "GLFW+OpenGL3 Hello World Program", NULL, NULL);
	if (window == NULL)
		return NULL;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// input callback functions
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetKeyCallback(window, key_callback);
	return window;
}

void InitGLEW() {
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(1);
	}
}

float determinant(glm::mat3 T, int n) {
	float det = 0.0f;
	glm::mat3 submatrix(0.0f);
	if (n == 2)
		return ((T[0][0] * T[1][1]) - (T[1][0] * T[0][1]));
	else {
		for (int x = 0; x < n; x++) {
			int subi = 0;
			for (int i = 1; i < n; i++) {
				int subj = 0;
				for (int j = 0; j < n; j++) {
					if (j == x)
						continue;
					submatrix[subi][subj] = T[i][j];
					subj++;
				}
				subi++;
			}
			det = det + (pow(-1, x) * T[0][x] * determinant(submatrix, n - 1));
		}
	}
	return det;
}

std::vector<std::vector<float>> ComputeEigenvectors(glm::mat3 matrix, std::vector<float> eigvals) {
	std::vector<std::vector<float>> vecs;

	for (const float& v : eigvals) {
		float a, b, c, d, e, f, g, h, i;
		a = matrix[0][0]; b = matrix[0][1]; c = matrix[0][2];
		d = matrix[1][0]; e = matrix[1][1]; f = matrix[1][2];
		g = matrix[2][0]; h = matrix[2][1]; i = matrix[2][2];

		std::vector<float> row0 = { a - v, b, c };
		std::vector<float> row1 = { d, e - v, f };
		std::vector<float> row2 = { g, h, i - v };

		std::vector<float> r0xr1 = {
			row0[1] * row1[2] - row0[2] * row1[1],
			row0[2] * row1[0] - row0[0] * row1[2],
			row0[0] * row1[1] - row0[1] * row1[0]
		};

		std::vector<float> r0xr2 = {
			row0[1] * row2[2] - row0[2] * row2[1],
			row0[2] * row2[0] - row0[0] * row2[2],
			row0[0] * row2[1] - row0[1] * row2[0]
		};

		std::vector<float> r1xr2 = {
			row1[1] * row2[2] - row1[2] * row2[1],
			row1[2] * row2[0] - row1[0] * row2[2],
			row1[0] * row2[1] - row1[1] * row2[0]
		};

		float d0 = r0xr1[0] * r0xr1[0] + r0xr1[1] * r0xr1[1] + r0xr1[2] * r0xr1[2];
		float d1 = r0xr2[0] * r0xr2[0] + r0xr2[1] * r0xr2[1] + r0xr2[2] * r0xr2[2];
		float d2 = r1xr2[0] * r1xr2[0] + r1xr2[1] * r1xr2[1] + r1xr2[2] * r1xr2[2];
		int imax = 0;
		float dmax = d0;

		if (d1 > dmax) {
			dmax = d1;
			imax = 1;
		}
		if (d2 > dmax) {
			imax = 2;
		}
		
		if (d0 == 0.0f) d0 = epsilon;
		if (d1 == 0.0f) d1 = epsilon;
		if (d2 == 0.0f) d2 = epsilon;

		if (imax == 0) {
			vecs.push_back({ r0xr1[0] / std::sqrt(d0), r0xr1[1] / std::sqrt(d0), r0xr1[2] / std::sqrt(d0) });
		}
		if (imax == 1) {
			vecs.push_back({ r0xr2[0] / std::sqrt(d1), r0xr2[1] / std::sqrt(d1), r0xr2[2] / std::sqrt(d1) });
		}
		if (imax == 2) {
			vecs.push_back({ r1xr2[0] / std::sqrt(d2), r1xr2[1] / std::sqrt(d2), r1xr2[2] / std::sqrt(d2) });
		}
	}
	return vecs;
}

std::vector<float> cubic(float a, float b, float c, float d) {

	std::vector<float> roots(3);

	//float q = (c / 3.0f) - ((b * b) / 9.0f);
	//float r = ((c * b - 3.0f * d) / 6.0f) - (b * b * b / 27.0f);
	//
	//float discrim = (r * r) + (q * q * q);
	//// Case 1: only one real solution
	//if (discrim > 0) {
	//	float A = std::cbrtf(abs(r) + std::sqrt(discrim));
	//	float term1 = 0.0f;
	//	if (r >= 0) term1 = A - (q / A);
	//	else term1 = (q / A) - A;
	// 
	//	roots[0] = term1 - (b / 3.0f);
	//	roots[1] = -(term1 / 2.0f) - (b / 3.0f);
	//	roots[2] = roots[1];
	//	//std::cout << "case 1 happened" << std::endl;
	//}
	//// Case 2: All roots are real
	//else if (discrim <= 0.0f) {
	//	float theta = 0.0f;
	//	if (q == 0.0f) theta = 0.0f;
	//	else theta = std::acosf(r / std::sqrt(-q * -q * -q));
	//	roots[0] = 2 * std::sqrt(-q) * std::cosf(theta / 3.0f) - b / 3.0f;
	//	roots[1] = 2 * std::sqrt(-q) * std::cosf((theta / 3.0f) + (120.0f * (PI / 180.0f))) - b / 3.0f;
	//	roots[2] = 2 * std::sqrt(-q) * std::cosf((theta / 3.0f) + (240 * (PI / 180.0f))) - b / 3.0f;
	//}
	

	float p = b * b - (3.0f * a * c);
	float q = (9.0f * a * b * c) - (2 * b * b * b) - 27.0f * a * a * d;
	
	if (p == 0.0f) {
		roots = { -b / (3.0f * a), -b / (3.0f * a) , -b / (3.0f * a) };
		std::cout << "p is zero" << std::endl;
	}
	else {
		float discrim = 27.0f * (p * p * p) / (q * q);
		float theta = 0.0f;
		if (q / (2.0f * p * std::sqrt(p)) > 1.0f) theta = std::acosf(1.0f);// std::cout << "bigger than 1 " << p << " " << q << std::endl;
		else
			theta = std::acosf(q / (2.0f * p * std::sqrt(p)));
		
		roots[0] = (-b + 2.0f * std::cosf(theta / 3.0f) * std::sqrt(p)) / (3.0f * a);
		roots[1] = (-b + 2.0f * std::cosf((theta / 3.0f) + (120.0f * (PI / 180.0f))) * std::sqrt(p)) / (3.0f * a);
		roots[2] = (-b + 2.0f * std::cosf((theta / 3.0f) + (240.0f * (PI / 180.0f))) * std::sqrt(p)) / (3.0f * a);
	}

	return roots;
}

void CalculateEigendecomposition(tira::volume< glm::mat3 > T) {
	lambda = tira::volume<float>(T.X(), T.Y(), T.Z(), 3);
	eigenvectors = tira::volume<glm::mat3>(T.X(), T.Y(), T.Z());

	for (size_t zi = 0; zi < T.Z(); zi++) {
		for (size_t yi = 0; yi < T.Y(); yi++) {
			for (size_t xi = 0; xi < T.X(); xi++) {

				glm::mat3 A = T(xi, yi, zi);
				
				std::vector<float> lambdas(3);
				std::vector<std::vector<float>> eigvecs(3, std::vector<float>(3));

				float a, b, c, d, e, f, g, h, i;
				a = A[0][0]; b = A[0][1]; c = A[0][2];
				d = A[1][0]; e = A[1][1]; f = A[1][2];
				g = A[2][0]; h = A[2][1]; i = A[2][2];

				if ((b * b + c * c + f * f) == 0.0f)
					lambdas = { a, e, i };
				else {
					float trace = a + e + i;
					float trace2 = (a * a + b * d + c * g) + (d * b + e * e + f * h) + (g * c + h * f + i * i);
					float det = determinant(A, 3); //(a * e * i - a * f * h) + (b * f * g - b * d * i) + (c * d * h - c * e * g);
					lambdas = cubic(1.0f, -trace, 0.5f * (trace * trace - trace2), -det);
					//if (xi == 10 && yi == 11)
					//	std::cout << trace << " " << -0.5f * (trace * trace - trace2) << " " << det << std::endl;
				}
					

				// calculating eigenvectors
				eigvecs = ComputeEigenvectors(A, lambdas);

				// sort the eignevalues
				if (lambdas[0] < lambdas[1]) {
					std::swap(lambdas[0], lambdas[1]);
					std::swap(eigvecs[0], eigvecs[1]);
				}
				if (lambdas[0] < lambdas[2]) {
					std::swap(lambdas[0], lambdas[2]);
					std::swap(eigvecs[0], eigvecs[2]);
				}
				if (lambdas[1] < lambdas[2]) {
					std::swap(lambdas[1], lambdas[2]);
					std::swap(eigvecs[1], eigvecs[2]);
				}
					

				lambda(xi, yi, zi, 0) = lambdas[0];
				eigenvectors(xi, yi, zi)[0][0] = eigvecs[0][0];
				eigenvectors(xi, yi, zi)[0][1] = eigvecs[0][1];
				eigenvectors(xi, yi, zi)[0][2] = eigvecs[0][2];

				lambda(xi, yi, zi, 1) = lambdas[1];
				eigenvectors(xi, yi, zi)[1][0] = eigvecs[1][0];
				eigenvectors(xi, yi, zi)[1][1] = eigvecs[1][1];
				eigenvectors(xi, yi, zi)[1][2] = eigvecs[1][2];

				lambda(xi, yi, zi, 2) = lambdas[2];
				eigenvectors(xi, yi, zi)[2][0] = eigvecs[2][0];
				eigenvectors(xi, yi, zi)[2][1] = eigvecs[2][1];
				eigenvectors(xi, yi, zi)[2][2] = eigvecs[2][2];
			}
		}
	}
}

void CalculateEigendecomposition_old(tira::volume< glm::mat3 > T) {

	/// I'm looking for better methods to do this, and eventually have to put it on the GPU
	/// Unstable (roots method): https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
	/// Quaternion method: https://stackoverflow.com/questions/4372224/fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition

	lambda = tira::volume<float>(T.X(), T.Y(), T.Z(), 3);
	eigenvectors = tira::volume<glm::mat3>(T.X(), T.Y(), T.Z());

	float lambda0, lambda1, lambda2;
	for (size_t zi = 0; zi < T.Z(); zi++) {
		for (size_t yi = 0; yi < T.Y(); yi++) {
			for (size_t xi = 0; xi < T.X(); xi++) {
				glm::mat3 t = T(xi, yi, zi);

				Eigen::Matrix3f A;
				A(0, 0) = t[0][0];
				A(0, 1) = t[0][1];
				A(0, 2) = t[0][2];
				A(1, 0) = t[1][0];
				A(1, 1) = t[1][1];
				A(1, 2) = t[1][2];
				A(2, 0) = t[2][0];
				A(2, 1) = t[2][1];
				A(2, 2) = t[2][2];

				Eigen::EigenSolver<Eigen::Matrix3f> es;
				es.compute(A, true);

				Eigen::Vector3f l = es.eigenvalues().real();
				Eigen::Matrix3f V = es.eigenvectors().real();

				Eigen::Vector3f evecs[3];
				evecs[0] = V.col(0).real();
				evecs[1] = V.col(1).real();
				evecs[2] = V.col(2).real();

				// sort the eigenvalues and eigenvectors from largest to smallest
				if (l[0] < l[1]) {
					std::swap(l[0], l[1]);
					std::swap(evecs[0], evecs[1]);
				}
				if (l[0] < l[2]) {
					std::swap(l[0], l[2]);
					std::swap(evecs[0], evecs[2]);
				}
				if (l[1] < l[2]) {
					std::swap(l[1], l[2]);
					std::swap(evecs[1], evecs[2]);
				}
				

				lambda(xi, yi, zi, 0) = l[0];
				eigenvectors(xi, yi, zi)[0][0] = evecs[0][0];
				eigenvectors(xi, yi, zi)[0][1] = evecs[0][1];
				eigenvectors(xi, yi, zi)[0][2] = evecs[0][2];

				lambda(xi, yi, zi, 1) = l[1];
				eigenvectors(xi, yi, zi)[1][0] = evecs[1][0];
				eigenvectors(xi, yi, zi)[1][1] = evecs[1][1];
				eigenvectors(xi, yi, zi)[1][2] = evecs[1][2];

				lambda(xi, yi, zi, 2) = l[2];
				eigenvectors(xi, yi, zi)[2][0] = evecs[2][0];
				eigenvectors(xi, yi, zi)[2][1] = evecs[2][1];
				eigenvectors(xi, yi, zi)[2][2] = evecs[2][2];
			}
		}
	}

}



int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("input", boost::program_options::value<std::string>(&in_filename)->default_value("psf.cw"), "output filename for the coupled wave structure")
		("help", "produce help message")
		("gamma,g", boost::program_options::value<float>(&in_gamma)->default_value(3), "glyph gamma (sharpness), 0 = spheroids")
		("cmap,c", boost::program_options::value<int>(&in_cmap)->default_value(0), "colormaped eigenvector (0 = longest, 2 = shortest)")
		;
	boost::program_options::variables_map vm;

	boost::program_options::positional_options_description p;
	p.add("input", -1);
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (!vm.count("input")) {
		std::cout << "ERROR: input file required" << std::endl;
		std::cout << desc << std::endl;
		return 1;
	}
	if (in_cmap != 0 && in_cmap != 2) {
		std::cout << "ERROR: invalid colormap component" << std::endl;
		return 1;
	}
	

	// Load the tensor field
	std::cout << "loading file...";
	T.load_npy<float>(in_filename);
	std::cout << "done." << std::endl;

	// Perform the eigendecomposition
	std::cout << "eigendecomposition...";
	CalculateEigendecomposition(T);
	std::cout << "done." << std::endl;

	// Initialize OpenGL
	window = InitGLFW();                                // create a GLFW window
	InitUI(window, glsl_version);
	InitGLEW();

	// Enable OpenGL environment parameters
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	
	float frame = std::max(T.X(), T.Y());
	camera.setPosition(frame / 2.0f, frame / 2.0f, frame);
	camera.LookAt(frame / 2.0f, frame / 2.0f, 0);


	glm::vec3 l(1.0f, 0.5f, 0.5f);
	float suml = l[0] + l[1] + l[2];
	float gamma = in_gamma;
	int component_color = in_cmap;
	

	tira::glGeometry glyph = tira::glGeometry::GenerateIcosphere<float>(3, false);	// create a square
	tira::glShader shader(vertex_shader_source, fragment_shader_source);

	glm::vec4 light0(0.0f, 100.0f, 100.0f, 0.7f);
	glm::vec4 light1(0.0f, -100.0f, 0.0f, 0.5f);
	float ambient = 0.3;	

	

	while (!glfwWindowShouldClose(window)){

		
		glfwPollEvents();													// Poll and handle events (inputs, window resize, etc.)
		RenderUI();
		int display_w, display_h;                                           // size of the frame buffer (openGL display)
		glfwGetFramebufferSize(window, &display_w, &display_h);
		
		if (reset) resetPlane(frame);
		float aspect = (float)display_w / (float)display_h;
		
		
		glm::mat4 Mprojection;
		if (aspect > 1) {
			if (!perspective)
				Mprojection = glm::ortho(-aspect * frame / 2.0f * zoom + right, aspect * frame / 2.0f * zoom + right, -frame / 2.0f * zoom + up, frame / 2.0f * zoom + up, -2.0f * frame, 2.0f * frame);
			else
				Mprojection = glm::perspective(60.0f * (float)std::numbers::pi / 180.0f, aspect, 0.1f, 200.0f);
		}
		else {
			if(!perspective)
				Mprojection = glm::ortho(-frame/2.0f, frame/2.0f, -frame/2.0f/aspect, frame/2.0f/aspect, -2.0f * frame, 2.0f * frame);
			else
				Mprojection = glm::perspective(60.0f * (float)std::numbers::pi / 180.0f, aspect, 0.1f, 200.0f);
		}

		glm::mat4 Mview = camera.getMatrix();								// generate a view matrix from the camera

		glViewport(0, 0, display_w, display_h);								// specifies the area of the window where OpenGL can render
		
		glClearColor(0, 0, 0, 0);
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (size_t xi = 0; xi < T.X(); xi++) {
			for (size_t yi = 0; yi < T.Y(); yi++) {
				/*if (xi == 10 && yi == 11) 
				{
					for(int i = 0 ; i < 3; i++)
						std::cout << eigenvectors(xi, yi, zi)[i][0] << " " << eigenvectors(xi, yi, zi)[i][1] << " " << eigenvectors(xi, yi, zi)[i][2] << std::endl;
					std::cout << lambda(xi, yi, zi, 0) << " " << lambda(xi, yi, zi, 1) << " " << lambda(xi, yi, zi, 2) << std::endl;
				}*/

				glm::mat4 Mtran = glm::translate(glm::mat4(1.0f), glm::vec3((float)xi + 0.5f, (float)yi + 0.5f, 0.0f));

				//glm::mat3 evmatrix = glm::transpose(eigenvectors(xi, yi, zi));
				glm::mat3 evmatrix = eigenvectors(xi, yi, zi);
				glm::mat4 Mrot = glm::mat4(evmatrix);
				Mrot[3][3] = 1.0f;

				glm::mat4 Mmodel = Mtran * Mrot;
				/// Render Something Here
				shader.Bind();
				shader.SetUniformMat4f("ProjMat", Mprojection);
				shader.SetUniformMat4f("ViewMat", Mview);
				shader.SetUniformMat4f("ModelMat", Mmodel);
				shader.SetUniform4f("light0", light0);
				shader.SetUniform4f("light1", light1);
				shader.SetUniform1f("ambient", ambient);
				shader.SetUniform1i("ColorComponent", component_color);
				glm::vec3 eval(lambda(xi, yi, zi, 0), lambda(xi, yi, zi, 1), lambda(xi, yi, zi, 2));
				shader.SetUniform3f("lambda", glm::normalize(eval) * 0.5f);
				//shader.SetUniform3f("lambda", l * 0.5f);
				shader.SetUniform1f("gamma", gamma);

				glyph.Draw();

			}
		}

		
		

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer
		glfwSwapBuffers(window);

	}

	DestroyUI();
	glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
	glfwTerminate();                                                // Terminate GLFW

}