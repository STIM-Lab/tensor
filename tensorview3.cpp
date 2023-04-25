#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <numbers>

#include "tira/graphics_gl.h"
#include "tira/volume.h"
#include <glm/gtc/quaternion.hpp>

#include <boost/program_options.hpp>

#include <Eigen/Eigenvalues>

GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
tira::camera camera;

bool dragging = false;
double xprev, yprev;

tira::volume<glm::mat3> T;
tira::volume<float> lambda;
tira::volume<glm::mat3> eigenvectors;

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
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
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

void CalculateEigendecomposition(tira::volume< glm::mat3 > T) {

	/// I'm looking for better methods to do this, and eventually have to put it on the GPU
	/// Unstable (roots method): https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
	/// Quaternion method: https://stackoverflow.com/questions/4372224/fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition

	lambda = tira::volume<float>(T.X(), T.Y(), T.Z(), 3);
	eigenvectors = tira::volume<glm::mat3>(T.X(), T.Y(), T.Z());

	double lambda0, lambda1, lambda2;
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

		int display_w, display_h;                                           // size of the frame buffer (openGL display)
		glfwGetFramebufferSize(window, &display_w, &display_h);
		

		float aspect = (float)display_w / (float)display_h;
		
		
		glm::mat4 Mprojection;
		if (aspect > 1) {
			Mprojection = glm::ortho(-aspect*frame/2.0f, aspect * frame/2.0f, -frame/2.0f, frame/2.0f, -2.0f * frame, 2.0f * frame);
		}
		else {
			Mprojection = glm::ortho(-frame/2.0f, frame/2.0f, -frame/2.0f/aspect, frame/2.0f/aspect, -2.0f * frame, 2.0f * frame);
		}

		glm::mat4 Mview = camera.getMatrix();								// generate a view matrix from the camera

		glViewport(0, 0, display_w, display_h);								// specifies the area of the window where OpenGL can render
		
		glClearColor(0, 0, 0, 0);
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (size_t xi = 0; xi < T.X(); xi++) {
			for (size_t yi = 0; yi < T.Y(); yi++) {
				
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

		
		


		glfwSwapBuffers(window);

	}

	glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
	glfwTerminate();                                                // Terminate GLFW

}