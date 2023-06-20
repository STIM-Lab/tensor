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
float right = 0.0f;
float up = 0.0f;
bool ctrl = false;
bool dragging = false;
double xprev, yprev;
size_t axes[] = { 0, 0, 0 };
int scroll_value = 0;

tira::volume<glm::mat3> T;
tira::volume<float> lambda;
tira::volume<glm::mat3> eigenvectors;


// input variables for arguments
std::string in_filename;
float in_gamma;
int in_cmap;
int in_size = 50;
int step;

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
	scroll_value += yoffset;
	if (scroll_value < 0) scroll_value = 0;
	if (scroll_value >= T.Z()) scroll_value = T.Z() - 1;
	std::cout << scroll_value << std::endl;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		perspective = !perspective;
	}
	if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS) {
		ctrl = true;
	}
	if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE) {
		ctrl = false;
	}
	if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS && ctrl) {
		zoom -= 0.1;
		if (zoom < 0.1) zoom = 0.1;
	}
	if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS && ctrl) {
		zoom += 0.1;
		if (zoom > 2) zoom = 2;
	}
	if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
		up += 5;
		if (up > 100) up = 100;
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		up -= 5;
		if (up < -100) up = -100;
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		right += 5;
		if (right > 100) right = 100;
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		right -= 5;
		if (right < -100) right = -100;
	}
}

void resetPlane(float frame) {
	camera.setPosition(frame / 2.0f, frame / 2.0f, frame);
	camera.LookAt(frame / 2.0f, frame / 2.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	right = 0.0f;
	up = 0.0f;
	zoom = 1.0f;
	axes[0] = 0; axes[1] = 0; axes[2] = 0;
	scroll_axis = 2;
	scroll_value = 0;
	in_size = 50;
	anisotropy = 0;
	filter = 0.1f;
}

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

// uses orthogonal complement to compute the eigenvectors correspoinding to the two identical eigenvalues W = U x V
void ComputeOrthogonalComplement(glm::vec3& U, glm::vec3& V, glm::vec3& W) {
	
	float invLength;
	if (fabs(W[0]) > fabs(W[1])) {
		// the component of maximum absolute value is either W[0] or W[2]
		invLength = 1 / (sqrtf(W[0] * W[0] + W[2] * W[2]));
		U = glm::vec3(-W[2] * invLength, 0.0f, W[0] * invLength);
	}
	else {
		// the component of maximum absolute value is either W[1] or W[2]
		invLength = 1 / (sqrtf(W[1] * W[1] + W[2] * W[2]));
		U = glm::vec3(0.0f, W[2] * invLength, -W[1] * invLength);
	}
	
	// vector V is the cross-product of W and U
	// cross-product: W = [a b c] and U = [x y z] -> V = W x U = [(bz-cy) (cx-az) (ay-bx)]
	V = glm::vec3(W[1] * U[2] - W[2] * U[1], W[2] * U[0] - W[0] * U[2], W[0] * U[1] - W[1] * U[0]);

}

glm::vec3 ComputeEigenvector(glm::mat3 matrix, float eigvalue) {

	glm::vec3 eigvec;
	float a, b, c, d, e, f, g, h, i;
	a = matrix[0][0]; b = matrix[0][1]; c = matrix[0][2];
	d = matrix[1][0]; e = matrix[1][1]; f = matrix[1][2];
	g = matrix[2][0]; h = matrix[2][1]; i = matrix[2][2];
	
	// rows of (A - lambda*I)
	// all the rows multiplied by the eigenvector yield zero vector => eigenvector is prependicular to at least two of the rows
	std::vector<float> row0 = { a - eigvalue, b, c };
	std::vector<float> row1 = { d, e - eigvalue, f };
	std::vector<float> row2 = { g, h, i - eigvalue };

	// calculate the cross-product of each two rows
	// v is parallel to the cross product of two of these rows
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

	if (imax == 0) {
		eigvec = glm::vec3(r0xr1[0] / std::sqrtf(d0), r0xr1[1] / std::sqrtf(d0), r0xr1[2] / std::sqrtf(d0));
	}
	else if (imax == 1) {
		eigvec = glm::vec3(r0xr2[0] / std::sqrtf(d1), r0xr2[1] / std::sqrtf(d1), r0xr2[2] / std::sqrtf(d1));
	}
	else {
		eigvec = glm::vec3(r1xr2[0] / std::sqrtf(d2), r1xr2[1] / std::sqrtf(d2), r1xr2[2] / std::sqrtf(d2));
	}

	return eigvec;
}

glm::vec3 ComputeEigenvalues(glm::mat3 A){
	glm::vec3 l;

	// to avoid precision loss, cast from float to double
	double a, b, c, d, e, f, g, h, i;
	a = static_cast<double>(A[0][0]); b = static_cast<double>(A[0][1]); c = static_cast<double>(A[0][2]);
	d = static_cast<double>(A[1][0]); e = static_cast<double>(A[1][1]); f = static_cast<double>(A[1][2]);
	g = static_cast<double>(A[2][0]); h = static_cast<double>(A[2][1]); i = static_cast<double>(A[2][2]);
	
	// Case: matrix is diagonal
	double p1 = b * b + c * c + f * f;
	
	if (p1 == 0)
	{
		l[0] = a;
		l[1] = e;
		l[2] = i;
	}
	else
	{
		double q = (a + e + i) / 3.0f;
		double p2 = (a - q) * (a - q) + (e - q) * (e - q) + (i - q) * (i - q) + 2 * p1;
		double p = std::sqrtf(p2 / 6.0f);
		glm::mat3 B = (float)(1 / p) * glm::mat3(glm::vec3(a - q, d, g), glm::vec3(b, e - q, h), glm::vec3(c, f, i - q));
		float r = determinant(B) / 2.0f;

		// In exact arithmetic for a symmetric matrix - 1 <= r <= 1
		// but computation error can leave it slightly outside this range.
		
		float phi = 0.0f;
		if (r <= -1)
			phi = PI / 3.0f;
		else if (r > 1)
			phi = 0.0f;
		else
			phi = std::acosf(r) / 3.0f;

		// the eigenvalues satisfy l[0] <= l[1] <= l[2]
		l[2] = q + 2 * p * std::cosf(phi);
		l[0] = q + 2 * p * std::cosf(phi + (2 * PI / 3.0f));
		l[1] = 3 * q - l[2] - l[0];					// since trace(A) = eig1 + eig2 + eig3
	}
	
	return l;
}

void CalculateEigendecomposition(tira::volume< glm::mat3 > T) {
	lambda = tira::volume<float>(T.X(), T.Y(), T.Z(), 3);
	eigenvectors = tira::volume<glm::mat3>(T.X(), T.Y(), T.Z());

	for (size_t zi = 0; zi < T.Z(); zi++) {
		for (size_t yi = 0; yi < T.Y(); yi++) {
			for (size_t xi = 0; xi < T.X(); xi++) {

				glm::mat3 A = T(xi, yi, zi);
				glm::vec3 lambdas = ComputeEigenvalues(A);			// l[0] <= l[1] <= l[2]
				glm::mat3 eigvecs;
				
				// case 1: input matrix is diagonal
				if (A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2] == 0) {
					eigvecs[0] = { 1.0f, 0.0f, 0.0f };
					eigvecs[1] = { 0.0f, 1.0f, 0.0f };
					eigvecs[2] = { 0.0f, 0.0f, 1.0f };
				}
					
				// case 2: two identical eigenvalues
				// with two identical eigenvalues, the rank of (A-lambda*I) is 1 -> cross-product of any two rows is zero.
				else if (lambdas[0] == lambdas[1]) {
					eigvecs[2] = ComputeEigenvector(A, lambdas[2]);
					ComputeOrthogonalComplement(eigvecs[0], eigvecs[1], eigvecs[2]);
				}

				else {
					eigvecs[0] = ComputeEigenvector(A, lambdas[0]);
					eigvecs[1] = ComputeEigenvector(A, lambdas[1]);
					eigvecs[2] = ComputeEigenvector(A, lambdas[2]);
				}

				// sort the eigenvalues and eigenvectors from largest to smallest
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

				// assign the values to the global variables
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

glm::mat4 GetCameraView() {
	float frame = (scroll_axis == 2) ? std::max(T.X(), T.Y()) : ((scroll_axis == 1) ? std::max(T.X(), T.Z()) : std::max(T.Y(), T.Z()));
	if (scroll_axis == 2 && axis_change)
	{
		camera.setPosition(frame / 2.0f, frame / 2.0f, frame);				// adjust the eye position
		camera.LookAt(frame / 2.0f, frame / 2.0f, 0, 0.0f, 1.0f, 0.0f);
	}
	else if (scroll_axis == 1 && axis_change)
	{
		camera.setPosition(frame / 2.0f, -frame, frame / 2.0f);
		camera.LookAt(frame / 2.0f, 0, frame / 2.0f, 0.0f, 0.0f, 1.0f);
	}
	else if (scroll_axis == 0 && axis_change)
	{
		camera.setPosition(frame, frame / 2.0f, frame / 2.0f);
		camera.LookAt(0, frame / 2.0f, frame / 2.0f, 0.0f, 0.0f, 1.0f);
	}
	axis_change = false;
	return camera.getMatrix();
}

tira::volume<float> GetDiagValues(tira::volume<glm::mat3> T) {
	tira::volume<float> diagonal_elem(T.X(), T.Y(), T.Z(), 3);
	glm::mat3 tensor;
	for (size_t m = 0; m < T.X(); m++) {
		for (size_t n = 0; n < T.Y(); n++) {
			for (size_t p = 0; p < T.Z(); p++) {
				tensor = T(m, n, p);
				diagonal_elem(m, n, p, 0) = tensor[0][0];
				diagonal_elem(m, n, p, 1) = tensor[1][1];
				diagonal_elem(m, n, p, 2) = tensor[2][2];
			}
		}
	}

	return diagonal_elem;
}
tira::volume<float> GetOffDiagValues(tira::volume<glm::mat3> T) {
	tira::volume<float> triangular_elem(T.X(), T.Y(), T.Z(), 3);
	glm::mat3 tensor;
	for (size_t m = 0; m < T.X(); m++) {
		for (size_t n = 0; n < T.Y(); n++) {
			for (size_t p = 0; p < T.Z(); p++) {
				tensor = T(m, n, p);
				triangular_elem(m, n, p, 0) = tensor[0][1];
				triangular_elem(m, n, p, 1) = tensor[0][2];
				triangular_elem(m, n, p, 2) = tensor[1][2];
			}
		}
	}

	return triangular_elem;
}

int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("input", boost::program_options::value<std::string>(&in_filename)->default_value("psf.cw"), "output filename for the coupled wave structure")
		("help", "produce help message")
		("gamma,g", boost::program_options::value<float>(&in_gamma)->default_value(3), "glyph gamma (sharpness), 0 = spheroids")
		("cmap,c", boost::program_options::value<int>(&in_cmap)->default_value(0), "colormaped eigenvector (0 = longest, 2 = shortest)")
		("size,s", boost::program_options::value<int>(&in_size)->default_value(50), "number of tensor along X axis (along Y and Z based on ratio)")
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
	//T.load_npy<float>("oval3d.npy");
	T.load_npy<float>("blurred2_3D.npy");
	std::cout << "done." << std::endl;

	// Set two separate RGB volumes with diagonal and off-diagonal values of the tensor (tensor is symmetric)
	tira::volume<float> diagonal_elem = GetDiagValues(T);
	tira::volume<float> triangular_elem = GetOffDiagValues(T);


	std::cout << "Size of volume:\t" << T.X() << " x " << T.Y() << " x " << T.Z() << std::endl;
	
	// Load the volume for texture-map image
	tira::glVolume<unsigned char> volume;
	volume.load_npy("cube1.npy");

	

	// Initialize OpenGL
	window = InitGLFW();                                // create a GLFW window
	InitUI(window, glsl_version);
	InitGLEW();

	// Enable OpenGL environment parameters
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	
	float frame = (scroll_axis == 2) ? std::max(T.X(), T.Y()) : ((scroll_axis == 1) ? std::max(T.X(), T.Z()) : std::max(T.Y(), T.Z()));
	
	glm::vec3 l(1.0f, 0.5f, 0.5f);
	float suml = l[0] + l[1] + l[2];
	float gamma = in_gamma;
	int component_color;	// = in_cmap;
	in_size = 50;
	bool planar = true;

	tira::glGeometry glyph = tira::glGeometry::GenerateIcosphere<float>(3, false);	// create a square
	tira::glGeometry rect = tira::glGeometry::GenerateRectangle<float>();           // create a rectangle for rendering volume

	// Load texture-map image shader
	tira::glMaterial material("volume.shader");
	material.SetTexture("volumeTexture", volume, GL_RGB, GL_NEAREST);

	// Copy the tensor field (including diagonal and off-diagonal RGB volumes) to GPU as texture maps
	tira::glMaterial shader("source.shader");
	shader.SetTexture("Diagonal", diagonal_elem, GL_RGBA32F, GL_LINEAR);
	shader.SetTexture("Upper_trian", triangular_elem, GL_RGBA32F, GL_LINEAR);

	// Set light position
	glm::vec4 light0(0.0f, 100.0f, 100.0f, 0.7f);
	glm::vec4 light1(0.0f, -100.0f, 0.0f, 0.5f);
	float ambient = 0.3;	

	bool file_loaded = false;
	
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
				Mprojection = glm::ortho(-aspect * frame / 2.0f / zoom + right, aspect * frame / 2.0f / zoom + right, -frame / 2.0f / zoom + up, frame / 2.0f / zoom + up, -2.0f * frame, 2.0f * frame);
			else
				Mprojection = glm::perspective(60.0f * (float)std::numbers::pi / 180.0f, aspect, 0.1f, 4.0f * frame);
		}
		else {
			if(!perspective)
				Mprojection = glm::ortho(-frame/2.0f, frame/2.0f, -frame/2.0f/aspect, frame/2.0f/aspect, -2.0f * frame, 2.0f * frame);
			else
				Mprojection = glm::perspective(60.0f * (float)std::numbers::pi / 180.0f, aspect, 0.1f, 4.0f * frame);
		}

		glm::mat4 Mview = GetCameraView(); // camera.getMatrix();								// generate a view matrix from the camera

		glViewport(0, 0, display_w, display_h);								// specifies the area of the window where OpenGL can render
		
		glClearColor(0, 0, 0, 0);
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glm::mat4 Mtran;

		/// HELIA: Render the plane with texture-mapped image here
		
		if (image_plane)
		{
			// Enable alpha blending for transparency and set blending function
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			size_t xi, yi, zi;
			xi = (scroll_axis == 0) ? scroll_value : axes[0];
			yi = (scroll_axis == 1) ? scroll_value : axes[1];
			zi = (scroll_axis == 2) ? scroll_value : axes[2];

			// Translation matrix
			Mtran = glm::translate(glm::mat4(1.0f), glm::vec3(T.X() * 0.5f, T.Y() * 0.5f, 0.0f));
			// Scale matrix
			glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(T.X(), T.Y(), 1.0f));

			// Scroll_value should be mapped from range [0, T.Z() - 1] to range [0, 1]
			float mappep_scroll_value = (float)scroll_value / (float)(T.Z() - 1);
			material.Begin();
			{
				material.SetUniformMat4f("MVP", Mprojection * Mview * Mtran * scale);
				material.SetUniform1f("slider", mappep_scroll_value);
				material.SetUniform1f("opacity", opacity);
				rect.Draw();
			}
			material.End();

			glDisable(GL_BLEND);
		}

		glClear(GL_DEPTH_BUFFER_BIT);

		step = T.X() / in_size;
		// Rendering the tensor field for the selected axis
		for (axes[0] = 0; axes[0] < T.X(); axes[0] += step) {
			for (axes[1] = 0; axes[1] < T.Y(); axes[1] += step) {
				for (axes[2] = 0; axes[2] < T.Z(); axes[2] += step)
				{
					size_t xi, yi, zi;
					xi = (scroll_axis == 0) ? scroll_value : axes[0];
					yi = (scroll_axis == 1) ? scroll_value : axes[1];
					zi = (scroll_axis == 2) ? scroll_value : axes[2];

					if		(scroll_axis == 2) Mtran = glm::translate(glm::mat4(1.0f), glm::vec3((float)xi + 0.5f, (float)yi + 0.5f, 0.0f));
					else if (scroll_axis == 1) Mtran = glm::translate(glm::mat4(1.0f), glm::vec3((float)xi + 0.5f, 0.0f, (float)zi + 0.5f));
					else if (scroll_axis == 0) Mtran = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, (float)yi + 0.5f, (float)zi + 0.5f));
					
					component_color = (cmap) ? 2 : 0;

					shader.Begin();
					{
						shader.SetUniformMat4f("ProjMat", Mprojection);
						shader.SetUniformMat4f("ViewMat", Mview);
						shader.SetUniformMat4f("Trans", Mtran);
						shader.SetUniform4f("light0", light0);
						shader.SetUniform4f("light1", light1);
						shader.SetUniform1f("ambient", ambient);
						shader.SetUniform1i("ColorComponent", component_color);
						shader.SetUniform1f("gamma", gamma);
						shader.SetUniform3ui("position", xi, yi, zi);
						shader.SetUniform1i("size", step);
						shader.SetUniform1f("filter", filter);
						shader.SetUniform1i("anisotropy", anisotropy);
						//shader.SetUniform1f("thresh", thresh);

						glyph.Draw();
					}
					shader.End();
					if (scroll_axis == 2) break;
				}
				if (scroll_axis == 1) break;
			}
			if (scroll_axis == 0) break;
		}
		


		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer
		glfwSwapBuffers(window);

	}

	DestroyUI();
	glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
	glfwTerminate();                                                // Terminate GLFW

}