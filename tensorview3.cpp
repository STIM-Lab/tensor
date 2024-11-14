#include <numbers>
#include <complex>
#include <GL/glew.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileBrowser/ImGuiFileBrowser.h"
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include "tira/volume.h"
#include "tira/graphics_gl.h"
#include <glm/gtc/quaternion.hpp>
#include <boost/program_options.hpp>

#define PI 3.14159265358979323846


glm::mat3* cudaGaussianBlur3D(glm::mat3* source, unsigned int width, unsigned int height, unsigned int depth,
	float sigma_w, float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height,
	unsigned int& out_depth, int deviceID = 0);
float* cudaEigenvalues3(float* tensors, unsigned int n, int device);
float* cudaEigenvectors3(float* tensors, float* lambda, size_t n, int device);

GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
//tira::camera camera;

extern bool perspective;
float move[2] = { 0.0f, 0.0f };							// UP and RIGHT, respectively
bool ctrl = false;
bool dragging = false;
double xprev, yprev;
size_t axes[] = { 0, 0, 0 };
int scroll_value = 0;

tira::volume<glm::mat3> T0;								// 3D tensor field (3x3 voxels)
tira::volume<glm::mat3> Tn(0, 0, 0);					// processed tensor field
tira::volume<float> Ln;									// eigenvalues of the processed tensor field (smallest to largest magnitude)
tira::volume<float> Vn;									// eigenvectors of the processed tensor field in polar coordinates (medium and large eigenvectors)

// input variables for arguments
std::string in_filename;
std::string in_image;
int in_device;
float in_gamma;
int in_cmap;
int step = 4;

// rendering variables
tira::camera Camera;

tira::glMaterial* SCALAR_MATERIAL;
tira::glMaterial* AXIS_MATERIAL;

tira::glGeometry* axis;
tira::glGeometry planes[2][2];

enum ScalarType { NoScalar, TensorElement, EVal, EVec, Anisotropy };
int SCALAR_TYPE = ScalarType::EVal;
int SCALAR_EVAL = 2;
int SCALAR_EVEC = 2;
int SCALAR_ANISOTROPY = 0;
enum ProcessingType { NoProcessing, Gaussian, Vote };
int PROCESSINGTYPE = ProcessingType::NoProcessing;

// scalar plane variables
bool RENDER_PLANE[] = { true, true, true };
int PLANE_POSITION[] = { 0, 0, 0 };

const std::string glyph_shader_string =
#include "shaders/glyph3d.shader"
;
const std::string volume_shader_string =
#include "shaders/volume.shader"
;
const std::string scalar_shader_string =
#include "shaders/volumemap.shader"
;
const std::string AXIS_MATERIAL_string =
#include "shaders/axis.shader"
;


float ui_scale = 1.5f;                                  // scale value for the UI and UI text
bool RESET = false;
bool window_focused = true;
bool axis_change = true;                                // gets true when the axis plane is changes
extern int step;                                        // the steps between each glyph along all axis

// tensor fields
int scroll_axis = 2;				                    // default axis is Z
int anisotropy = 0;                                     // 0: all tensors               1: linear tensors only
														// 2: planar tensors only       3: spherical tensors only
float filter = 0.1f;
float zoom = 1.0f;
int cmap = 1;
float opacity = 1.0f;
float thresh = 0.0f;
bool perspective = false;

bool OPEN_TENSOR = false;
bool RENDER_GLYPHS = false;
bool TENSOR_LOADED = false;
bool SET_CAMERA = false;
bool BLUR = false;
float SIGMA;
std::string TensorFileName;

bool OPEN_VOLUME = false;
bool RENDER_IMAGE = false;
bool VOLUME_LOADED = false;
std::string VolumeFileName;

bool tensor_data;
bool volume_data;
imgui_addons::ImGuiFileBrowser file_dialog;


void OpenFileDialog() {
	std::cout << "Loading \"" << file_dialog.selected_fn << "\" ..." << std::endl;
	if (tensor_data) {
		OPEN_TENSOR = true;
		TensorFileName = file_dialog.selected_path;
	}
	if (volume_data)
	{
		OPEN_VOLUME = true;
		VolumeFileName = file_dialog.selected_path;
	}

	tensor_data = false;
	volume_data = false;
}

/// <summary>
/// Initialize the GUI
/// </summary>
/// <param name="window">Pointer to the GLFW window that will be used for rendering</param>
/// <param name="glsl_version">Version of GLSL that will be used</param>
void InitUI(GLFWwindow* window, const char* glsl_version) {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	ImGui::GetStyle().ScaleAllSizes(ui_scale);
	ImGui::GetIO().FontGlobalScale = ui_scale;

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Fonts
	io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", ui_scale * 16.0f);

}

/// <summary>
/// Destroys the ImGui rendering interface (usually called when the program closes)
/// </summary>
void DestroyUI() {
	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

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
		Camera.orbit(dx * ANGLE_SCALE, dy * ANGLE_SCALE);
		xprev = xpos;
		yprev = ypos;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	if (TENSOR_LOADED) {
		/*scroll_value += yoffset;
		if (scroll_value < 0) scroll_value = 0;
		if (scroll_value >= Tn.Z()) scroll_value = Tn.Z() - 1;
		std::cout << scroll_value << std::endl;*/
		zoom += 0.2 * yoffset;
		zoom = (zoom < 1.0f) ? 1.0f : ((zoom > 5) ? 5 : zoom);
	}
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
		move[0] += 5;
		if (move[0] > 100) move[0] = 100;
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		move[0] -= 5;
		if (move[0] < -100) move[0] = -100;
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		move[1] += 5;
		if (move[1] > 100) move[1] = 100;
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		move[1] -= 5;
		if (move[1] < -100) move[1] = -100;
	}
}

void resetPlane() {
	// Reset camera view to initial state
	Camera.position({ 0.0f, 0.0f, -2 * Tn.smax() });
	Camera.up({ 0.0f, 1.0f, 0.0f });
	Camera.lookat({ Tn.X() / 2.0f, Tn.Y() / 2.0f, Tn.Z() / 2.0f });
	Camera.fov(60);

	PLANE_POSITION[0] = static_cast<int>(Tn.X() / 2);
	PLANE_POSITION[1] = static_cast<int>(Tn.Y() / 2);
	PLANE_POSITION[2] = static_cast<int>(Tn.Z() / 2);

	move[1] = 0.0f;
	move[0] = 0.0f;
	zoom = 1.0f;
	axes[0] = 0; axes[1] = 0; axes[2] = 0;
	scroll_axis = 2;
	scroll_value = 0;
	anisotropy = 0;
	filter = 0.1f;
	thresh = 0.0f;
	step = 4;
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

void ColormapEvec(size_t vi) {
	tira::volume<unsigned char> C(Tn.X(), Tn.Y(), Tn.Z(), 3);

	for (size_t zi = 0; zi < Tn.Z(); zi++) {
		float z = (float)zi / (float)Tn.Z();
		for (size_t yi = 0; yi < Tn.Y(); yi++) {
			float y = (float)yi / (float)Tn.Y();
			for (size_t xi = 0; xi < Tn.X(); xi++) {
				float theta, phi, x, y, z;
				if (vi == 1) {
					theta = Vn(xi, yi, zi, 0);
					phi = Vn(xi, yi, zi, 1);

					x = sin(theta) * cos(phi);
					y = sin(theta) * sin(phi);
					z = cos(theta);
				}
				else if (vi == 2) {
					theta = Vn(xi, yi, zi, 2);
					phi = Vn(xi, yi, zi, 3);

					x = sin(theta) * cos(phi);
					y = sin(theta) * sin(phi);
					z = cos(theta);
				}
				else {
					float theta1 = Vn(xi, yi, zi, 0);
					float phi1 = Vn(xi, yi, zi, 1);

					float x1 = sin(theta1) * cos(phi1);
					float y1 = sin(theta1) * sin(phi1);
					float z1 = cos(theta1);

					float theta2 = Vn(xi, yi, zi, 0);
					float phi2 = Vn(xi, yi, zi, 1);

					float x2 = sin(theta2) * cos(phi2);
					float y2 = sin(theta2) * sin(phi2);
					float z2 = cos(theta2);

					x = y1 * z2 - z1 * y2;
					y = z1 * x2 - x1 * z2;
					z = x1 * y2 - y1 * x2;
				}

				C(xi, yi, zi, 0) = abs(x) * 255;
				C(xi, yi, zi, 1) = abs(y) * 255;
				C(xi, yi, zi, 2) = abs(z) * 255;
			}
		}
	}
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
}

void ColormapEval(int eval) {
	tira::volume<float> E = Ln.channel(eval);
	tira::volume<unsigned char> C = E.cmap(ColorMap::Magma);
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
}

void ColormapFA() {
	tira::volume<float> A(Tn.X(), Tn.Y(), Tn.Z());

	for (size_t zi = 0; zi < Tn.Z(); zi++) {
		float z = (float)zi / (float)Tn.Z();
		for (size_t yi = 0; yi < Tn.Y(); yi++) {
			float y = (float)yi / (float)Tn.Y();
			for (size_t xi = 0; xi < Tn.X(); xi++) {
				float l0 = Ln(xi, yi, zi, 0);
				float l1 = Ln(xi, yi, zi, 1);
				float l2 = Ln(xi, yi, zi, 2);
				//float lhat = (l0 + l1 + l2) / 3.0f;

				float num = pow(l2 - l1, 2) + pow(l1 - l0, 2) + pow(l0 - l2, 2);
				float den = pow(l0, 2) + pow(l1, 2) + pow(l2, 2);
				A(xi, yi, zi) = sqrt(0.5 * num / den);
			}
		}
	}
	tira::volume<unsigned char> C = A.cmap(0, 1, ColorMap::Magma);
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
}

void ColormapLinearA() {
	tira::volume<float> A(Tn.X(), Tn.Y(), Tn.Z());

	for (size_t zi = 0; zi < Tn.Z(); zi++) {
		float z = (float)zi / (float)Tn.Z();
		for (size_t yi = 0; yi < Tn.Y(); yi++) {
			float y = (float)yi / (float)Tn.Y();
			for (size_t xi = 0; xi < Tn.X(); xi++) {
				float l0 = Ln(xi, yi, zi, 0);
				float l1 = Ln(xi, yi, zi, 1);
				float l2 = Ln(xi, yi, zi, 2);
				//float lhat = (l0 + l1 + l2) / 3.0f;

				float num = l2 - l1;
				float den = l0 + l1 + l2;
				A(xi, yi, zi) = num/den;
			}
		}
	}
	tira::volume<unsigned char> C = A.cmap(0, 1, ColorMap::Magma);
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
}

void ColormapPlateA() {
	tira::volume<float> A(Tn.X(), Tn.Y(), Tn.Z());

	for (size_t zi = 0; zi < Tn.Z(); zi++) {
		float z = (float)zi / (float)Tn.Z();
		for (size_t yi = 0; yi < Tn.Y(); yi++) {
			float y = (float)yi / (float)Tn.Y();
			for (size_t xi = 0; xi < Tn.X(); xi++) {
				float l0 = Ln(xi, yi, zi, 0);
				float l1 = Ln(xi, yi, zi, 1);
				float l2 = Ln(xi, yi, zi, 2);
				//float lhat = (l0 + l1 + l2) / 3.0f;

				float num = 2 * (l1 - l0);
				float den = l0 + l1 + l2;
				A(xi, yi, zi) = num / den;
			}
		}
	}
	tira::volume<unsigned char> C = A.cmap(0, 1, ColorMap::Magma);
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
}

void ColormapSphereA() {
	tira::volume<float> A(Tn.X(), Tn.Y(), Tn.Z());

	for (size_t zi = 0; zi < Tn.Z(); zi++) {
		float z = (float)zi / (float)Tn.Z();
		for (size_t yi = 0; yi < Tn.Y(); yi++) {
			float y = (float)yi / (float)Tn.Y();
			for (size_t xi = 0; xi < Tn.X(); xi++) {
				float l0 = Ln(xi, yi, zi, 0);
				float l1 = Ln(xi, yi, zi, 1);
				float l2 = Ln(xi, yi, zi, 2);
				//float lhat = (l0 + l1 + l2) / 3.0f;

				float num = 3 * l0;
				float den = l0 + l1 + l2;
				A(xi, yi, zi) = num / den;
			}
		}
	}
	tira::volume<unsigned char> C = A.cmap(0, 1, ColorMap::Magma);
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
}

void ColormapScalar() {
	if (SCALAR_TYPE == ScalarType::EVal) {
		ColormapEval(SCALAR_EVAL);
	}
	if (SCALAR_TYPE == ScalarType::Anisotropy) {
		if (SCALAR_ANISOTROPY == 0) ColormapFA();
		if (SCALAR_ANISOTROPY == 1) ColormapLinearA();
		if (SCALAR_ANISOTROPY == 2) ColormapPlateA();
		if (SCALAR_ANISOTROPY == 3) ColormapSphereA();
	}
}

inline float normaldist(const float x, const float sigma) {
	const float scale = 1.0f / (sigma * sqrt(2 * 3.14159));
	const float ex = -(x * x) / (2 * sigma * sigma);
	return scale * exp(ex);
}

/// <summary>
/// Blurs the tensor field and re-calculates the current scalar image
/// </summary>
/// <param name="sigma"></param>
void GaussianFilter(const float sigma) {
	// if a CUDA device is enabled, use a blur kernel
	if (in_device >= 0) {
		std::cout << "Gaussian on CUDA" << std::endl;
		unsigned int blur_width;
		unsigned int blur_height;
		unsigned int blur_depth;
		glm::mat3* blurred = cudaGaussianBlur3D(T0.data(), T0.X(), T0.Y(), T0.Z(), sigma, sigma, sigma,
			blur_width, blur_height, blur_depth, in_device);

		Tn = tira::volume<glm::mat3>(blurred, blur_width, blur_height, blur_depth);
		free(blurred);
	}
	// otherwise use the CPU
	else {
		const unsigned int size = ceil(sigma * 6);
		const float start = -static_cast<float>(size - 1) / 2.0f;
		tira::volume<float> Kx(size, 1, 1);
		tira::volume<float> Ky(1, size, 1);
		tira::volume<float> Kz(1, 1, size);

		// fill the kernels with the Gaussian values
		for (size_t i = 0; i < size; i++) {
			constexpr float dx = 1.0f;
			const float v = normaldist(start + dx * static_cast<float>(i), sigma);
			Kx(i, 0, 0) = v;
			Ky(0, i, 0) = v;
			Kz(0, 0, i) = v;
		}
		Tn = T0.convolve3D(Kx);
		Tn = Tn.convolve3D(Ky);
		Tn = Tn.convolve3D(Kz);
	}
}

void UpdateEigens() {
	float* eigenvalues_raw = cudaEigenvalues3(reinterpret_cast<float*>(Tn.data()), Tn.size(), in_device);
	Ln = tira::volume<float>(eigenvalues_raw, Tn.X(), Tn.Y(), Tn.Z(), 3);

	float* eigenvectors_raw = cudaEigenvectors3(reinterpret_cast<float*>(Tn.data()), eigenvalues_raw, Tn.size(), in_device);
	Vn = tira::volume<float>(eigenvectors_raw, Tn.X(), Tn.Y(), Tn.Z(), 4);

	free(eigenvalues_raw);
	free(eigenvectors_raw);
}

void ScalarRefresh() {
	switch (SCALAR_TYPE) {
	case ScalarType::EVal:
		ColormapEval(SCALAR_EVAL);
		break;
	case ScalarType::EVec:
		ColormapEvec(SCALAR_EVEC);
		break;
	case ScalarType::Anisotropy:
		ColormapScalar();
		break;
	default:
		throw std::runtime_error("Invalid scalar type");
	}
}

// Load a tensor field from a NumPy file
void LoadTensorField3(std::string npy_filename) {
	// Load the tensor field
	T0.load_npy<float>(npy_filename);
	Tn = T0;
	
	// Fix the plane position in the middle for better visualization
	PLANE_POSITION[0] = static_cast<int>(Tn.X() / 2);
	PLANE_POSITION[1] = static_cast<int>(Tn.Y() / 2);
	PLANE_POSITION[2] = static_cast<int>(Tn.Z() / 2);

	// Separate the diagonal and off-diagonal elements to be sent off to GPU as RGB volume texture-maps
	//tira::volume<float> diagonal_elem = GetDiagValues(T);
	//tira::volume<float> triangular_elem = GetOffDiagValues(T);

	// Copy the tensor field (including diagonal and off-diagonal RGB volumes) to GPU as texture maps
	//GLYPH_MATERIAL->SetTexture("Diagonal", diagonal_elem, GL_RGBA32F, GL_LINEAR);
	//GLYPH_MATERIAL->SetTexture("Upper_trian", triangular_elem, GL_RGBA32F, GL_LINEAR);

	// Mark tensor as loaded for rendering
	TENSOR_LOADED = true;
	SET_CAMERA = true;

	std::cout << "Tensor loaded successfully.\n" << std::endl;
	std::cout << "Size of volume:\t(" << T0.X() << " x " << T0.Y() << " x " << T0.Z() << ")" << std::endl;
	std::cout << "World size:\t(" << T0.sx() << " x " << T0.sy() << " x " << T0.sz() << ")" << std::endl;

	UpdateEigens();
	ColormapScalar();
}

void inline draw_axes(glm::mat4 Mview, int ax) {
	glm::mat4 Mobj = glm::mat4(1.0f);
	float radius = Tn.smax() / 150.0f;
	AXIS_MATERIAL->SetUniform1i("axis", ax);
	float x_location = (float)PLANE_POSITION[0] / (float)(Tn.sx() - 1);
	float y_location = (float)PLANE_POSITION[1] / (float)(Tn.sy() - 1);
	float z_location = (float)PLANE_POSITION[2] / (float)(Tn.sz() - 1);
	AXIS_MATERIAL->SetUniform3f("loc", x_location, y_location, z_location);

	glm::vec3 pos[4];
	glm::vec3 scale[2];
	glm::vec3 rot[2];

	if (ax == 0) {								// X axis
		pos[0] = glm::vec3((float)PLANE_POSITION[0], 0.0f, Tn.Z() / 2.0f);		
		pos[1] = glm::vec3((float)PLANE_POSITION[0], Tn.Y(), Tn.Z() / 2.0f);
		pos[2] = glm::vec3((float)PLANE_POSITION[0], Tn.Y() / 2.0f, Tn.Z());
		pos[3] = glm::vec3((float)PLANE_POSITION[0], Tn.Y() / 2.0f, 0);

		rot[0] = glm::vec3(0.0f, 1.0f, 0.0f);
		rot[1] = glm::vec3(0.0f, 0.0f, 1.0f);

		scale[0] = glm::vec3(Tn.sz(), radius, radius);
		scale[1] = glm::vec3(Tn.sy(), radius, radius);
	}
	else if (ax == 1) {								// Y axis
		pos[0] = glm::vec3(0.0f, (float)PLANE_POSITION[1], Tn.Z() / 2.0f);
		pos[1] = glm::vec3(Tn.X(), (float)PLANE_POSITION[1], Tn.Z() / 2.0f);
		pos[2] = glm::vec3(Tn.X() / 2.0f, (float)PLANE_POSITION[1], 0.0f);
		pos[3] = glm::vec3(Tn.X() / 2.0f, (float)PLANE_POSITION[1], Tn.Z());

		rot[0] = glm::vec3(1.0f, 0.0f, 0.0f);
		rot[1] = glm::vec3(0.0f, 0.0f, 1.0f);

		scale[0] = glm::vec3(radius, Tn.sz(), radius);
		scale[1] = glm::vec3(radius, Tn.sx(), radius);		
	}
	else if (ax == 2) {								// Z axis
		pos[0] = glm::vec3(Tn.X() / 2.0f, 0.0f, (float)PLANE_POSITION[2]);		
		pos[1] = glm::vec3(Tn.X() / 2.0f, Tn.Y(), (float)PLANE_POSITION[2]);
		pos[2] = glm::vec3(0.0f, Tn.Y() / 2.0f, (float)PLANE_POSITION[2]);
		pos[3] = glm::vec3(Tn.X(), Tn.Y() / 2.0f, (float)PLANE_POSITION[2]);

		rot[0] = glm::vec3(0.0f, 1.0f, 0.0f);
		rot[1] = glm::vec3(1.0f, 0.0f, 0.0f);

		scale[0] = glm::vec3(radius, radius, Tn.sx());
		scale[1] = glm::vec3(radius, radius, Tn.sy());
	}

	const float rad = glm::radians(-90.0f);
	Mobj = glm::translate(Mobj, pos[0]);
	Mobj = glm::rotate(Mobj, rad, rot[0]);
	Mobj = glm::scale(Mobj, scale[0]);
	AXIS_MATERIAL->SetUniformMat4f("MVP", Mview * Mobj);
	axis->Draw();

	Mobj = glm::mat4(1.0f);
	Mobj = glm::translate(Mobj, pos[1]);
	Mobj = glm::rotate(Mobj, rad, rot[0]);
	Mobj = glm::scale(Mobj, scale[0]);
	AXIS_MATERIAL->SetUniformMat4f("MVP", Mview * Mobj);
	axis->Draw();

	Mobj = glm::mat4(1.0f);
	Mobj = glm::translate(Mobj, pos[2]);
	Mobj = glm::rotate(Mobj, rad, rot[1]);
	Mobj = glm::scale(Mobj, scale[1]);
	AXIS_MATERIAL->SetUniformMat4f("MVP", Mview * Mobj);
	axis->Draw();

	Mobj = glm::mat4(1.0f);
	Mobj = glm::translate(Mobj, pos[3]);
	Mobj = glm::rotate(Mobj, rad, rot[1]);
	Mobj = glm::scale(Mobj, scale[1]);
	AXIS_MATERIAL->SetUniformMat4f("MVP", Mview * Mobj);
	axis->Draw();
}

/// <summary>
/// This function renders the user interface every frame
/// </summary>
void RenderUI() {
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	{
		// Use smaller font size
		float old_size = ImGui::GetFont()->Scale;
		ImGui::GetFont()->Scale *= 0.5;
		ImGui::PushFont(ImGui::GetFont());

		ImGui::Begin("Tensor");

		window_focused = (ImGui::IsWindowHovered()) ? false : true;

		// Change style 
		ImGuiStyle& style = ImGui::GetStyle();
		style.FrameRounding = 5.f;
		style.GrabRounding = 3.f;
		style.WindowRounding = 7.f;

		if (ImGui::BeginTabBar("MyTabBar"))
		{

			// First tab
			if (ImGui::BeginTabItem("Tensor Field"))
			{
				////////////////////////////////////////////////  Load tensor field  ///////////////////////////////////////////////
				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::SeparatorText("Load");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				if (ImGui::Button("Load Tensor"))					                                // create a button for loading the shader
				{
					ImGui::OpenPopup("Open File");
					tensor_data = true;
				}

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				///////////////////////////////////////////////  Render Planes  //////////////////////////////////////////////////
				ImGui::SeparatorText("Planes");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::Checkbox("X", &RENDER_PLANE[0]);
				ImGui::SetItemTooltip("Render plane X of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("X Position", &PLANE_POSITION[0], 0, Tn.X() - 1);

				ImGui::Checkbox("Y", &RENDER_PLANE[1]);
				ImGui::SetItemTooltip("Render plane Y of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("Y Position", &PLANE_POSITION[1], 0, Tn.Y() - 1);

				ImGui::Checkbox("Z", &RENDER_PLANE[2]);
				ImGui::SetItemTooltip("Render plane Z of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("Z Position", &PLANE_POSITION[2], 0, Tn.Z() - 1);

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				////////////////////////////////////////////  Scalar Visualization  /////////////////////////////////////////////
				ImGui::SeparatorText("Scalar Visualization");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				if (ImGui::RadioButton("Eigenvalues", &SCALAR_TYPE, ScalarType::EVal)) {
					ColormapEval(SCALAR_EVAL);
				}
				if (ImGui::InputInt("Eigenvalue", &SCALAR_EVAL)) {
					if (SCALAR_EVAL < 0) SCALAR_EVAL = 0;
					if (SCALAR_EVAL > 2) SCALAR_EVAL = 2;
					if (SCALAR_TYPE == ScalarType::EVal) ColormapEval(SCALAR_EVAL);
				}
				if (ImGui::RadioButton("Anisotropy", &SCALAR_TYPE, ScalarType::Anisotropy)) {
					ColormapScalar();
				}
				const char* items[] = { "Fractional", "Linear", "Plate", "Spherical" };

				// Pass in the preview value visible before opening the combo (it could technically be different contents or not pulled from items[])
				const char* combo_preview_value = items[SCALAR_ANISOTROPY];

				if (ImGui::BeginCombo("##AnisotropyType", combo_preview_value))
				{
					for (int n = 0; n < IM_ARRAYSIZE(items); n++)
					{
						const bool is_selected = (SCALAR_ANISOTROPY == n);
						if (ImGui::Selectable(items[n], is_selected)) {
							SCALAR_ANISOTROPY = n;
							ColormapScalar();
						}

						// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
						if (is_selected)
							ImGui::SetItemDefaultFocus();
					}

					ImGui::EndCombo();
				}

				if (ImGui::RadioButton("Eigenvectors", &SCALAR_TYPE, ScalarType::EVec)) {
					ColormapEvec(2);
				}
				if (ImGui::InputInt("Eigenvector", &SCALAR_EVEC)) {
					if (SCALAR_EVEC < 0) SCALAR_EVEC = 0;
					if (SCALAR_EVEC > 2) SCALAR_EVEC = 2;
					if (SCALAR_TYPE == ScalarType::EVec) ColormapEvec(SCALAR_EVEC);
				}

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				//////////////////////////////////////////  Orthogonal / Perspective View  /////////////////////////////////////////
				// Use perspective view instead of ortho view
				ImGui::SeparatorText("Projection");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				if (ImGui::RadioButton("Ortho", !perspective))
					perspective = false;
				ImGui::SameLine();
				if (ImGui::RadioButton("Perspective", perspective))
					perspective = true;
				ImGui::Spacing(); ImGui::Spacing();

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				//////////////////////////////////////////////////     Zoom      //////////////////////////////////////////////////
				// Zooming in and out option
				ImGui::SeparatorText("Zoom");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::InputFloat("##Zoom", &zoom, 0.1f, 2.0f);
				zoom = (zoom < 1.0f) ? 1.0f : ((zoom > 5) ? 5 : zoom);
				ImGui::SameLine();
				if (ImGui::Button("O", ImVec2(25, 25))) zoom = 1.0f;             // reset zoom
				ImGui::Spacing(); ImGui::Spacing();

				ImGui::Dummy(ImVec2(0.0f, 5.0f));
				///////////////////////////////////////////////////   Reset View   //////////////////////////////////////////////////
				ImGui::SeparatorText("Reset");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				float avail = ImGui::GetContentRegionAvail().x;
				float off = (avail - 50) * 0.5f;
				if (off > 0.0f)
					ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
				RESET = ImGui::Button("Reset", ImVec2(50, 50));

				ImGui::EndTabItem();
			}

			// Second tab
			if (ImGui::BeginTabItem("Processing"))
			{
				if (ImGui::RadioButton("None", &PROCESSINGTYPE, (int)ProcessingType::NoProcessing)) {
					Tn = T0;
					UpdateEigens();
				}
				///////////////////////////////////////////////  Gaussian Blur  ///////////////////////////////////////////////////
				ImGui::SeparatorText("Load");
				if (ImGui::RadioButton("Gaussian Blur", &PROCESSINGTYPE, (int)ProcessingType::Gaussian)) {
					if (PROCESSINGTYPE == ProcessingType::Gaussian) {
						GaussianFilter(SIGMA);
						UpdateEigens();
						ScalarRefresh();
					}
					else {
						Tn = T0;
						UpdateEigens();
						ScalarRefresh();
					}
				}
				ImGui::SameLine();
				if (ImGui::InputFloat("##Sigma", &SIGMA, 0.2f, 1.0f)) {
					if (SIGMA <= 0) SIGMA = 0.01;
					if (PROCESSINGTYPE == ProcessingType::Gaussian) {
						GaussianFilter(SIGMA);
						UpdateEigens();
						ScalarRefresh();
					}
				}
				if (file_dialog.showFileDialog("Open File", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 700), ".npy"))
					OpenFileDialog();

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::EndTabItem();
			}

			// Third tab
			if (ImGui::BeginTabItem("Previous"))
			{
				ImGui::Text("Implementations from the previous version of tensorview3d");

				// Filter anisotropy using a threshold filter option
				ImGui::SeparatorText("Anisotropy");
				ImGui::Combo(" ", &anisotropy, "All\0Linear\0Planar\0Spherical\0\0");
				ImGui::Spacing();
				ImGui::SliderFloat("Filter", &filter, 0.1f, 1.0f);
				ImGui::Spacing(); ImGui::Spacing();

				// Show colormapped image based on shortest/longest eigenvector
				ImGui::SeparatorText("IDK");
				ImGui::Combo("\t", &cmap, "LongestVector\0ShortestVector\0\0");
				ImGui::Spacing(); ImGui::Spacing();

				// Adjust a threshold for eigenvalues corresponding to each tensor
				ImGui::SeparatorText("Largest Eigenvalue Threshold");
				static float begin = 0.f, end = 125.f, step = 0.005f;
				ImGui::DragFloatRange2("Range", &begin, &end, 0.25f, 0.0f, 100, "Min: %.1f", "Max: %.1f");
				ImGui::DragFloat("Step", &step, 0.00001, 0.0f, 1.0f, "%.6f");
				ImGui::DragFloat("Threshold", &thresh, step, begin, end, "%.6f");
				ImGui::Spacing(); ImGui::Spacing();

				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}

		ImGui::GetFont()->Scale = old_size;
		ImGui::PopFont();
		ImGui::End();

	}
	//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

	ImGui::Render();                                                            // Render all windows
}

void RenderPlane(int p) {
	glm::mat4 Mobj = glm::mat4(1.0f);
	glm::mat4 Mtex = glm::mat4(1.0f);

	glm::vec3 pos;
	glm::vec3 rot;
	glm::vec3 scale = glm::vec3(Tn.sx(), Tn.sy(), Tn.sz());

	glm::vec3 tpos;
	glm::vec3 tscale;

	if (p == 0) {
		pos = glm::vec3((float)PLANE_POSITION[0], Tn.Y() / 2.0f, Tn.Z() / 2.0f);
		tpos = glm::vec3((float)PLANE_POSITION[0], 0.0f, 0.0f);

		rot = glm::vec3(0.0f, 1.0f, 0.0f);

		tscale = glm::vec3(1.0f, Tn.Y(), Tn.Z());
	}
	else if (p == 1) {
		pos = glm::vec3(Tn.X() / 2.0f, (float)PLANE_POSITION[1], Tn.Z() / 2.0f);
		tpos = glm::vec3(0.0f, (float)PLANE_POSITION[1], 0.0f);

		rot = glm::vec3(1.0f, 0.0f, 0.0f);

		tscale = glm::vec3(Tn.X(), 1.0f, Tn.Z());
	}
	else {
		pos = glm::vec3(Tn.X() / 2.0f, Tn.Y() / 2.0f, (float)PLANE_POSITION[2]);
		tpos = glm::vec3(0.0f, 0.0f, (float)PLANE_POSITION[2]);

		rot = glm::vec3(0.0f, 0.0f, 1.0f);

		tscale = glm::vec3(Tn.X(), Tn.Y(), 1.0f);
	}

	float radians = (p == 1) ? glm::radians(90.0f) : glm::radians(-90.0f);
	if (p == 2) radians = glm::radians(0.0f);
	Mobj = glm::translate(Mobj, pos);
	Mobj = glm::scale(Mobj, scale);
	Mobj = glm::rotate(Mobj, radians, rot);

	Mtex = glm::translate(Mtex, tpos);
	Mtex = glm::scale(Mtex, tscale);
	Mtex = glm::rotate(Mtex, radians, rot);

	SCALAR_MATERIAL->SetUniformMat4f("Mobj", Mobj);
	SCALAR_MATERIAL->SetUniformMat4f("Mtex", Mtex);
	SCALAR_MATERIAL->SetUniform1f("opacity", opacity);
	planes[0][0].Draw();
	planes[0][1].Draw();
	planes[1][0].Draw();
	planes[1][1].Draw();
}

void RenderPlanes() {
	if(RENDER_PLANE[0]) RenderPlane(0);
	if(RENDER_PLANE[1]) RenderPlane(1);
	if(RENDER_PLANE[2]) RenderPlane(2);
}

int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("input", boost::program_options::value<std::string>(&in_filename), "input filename for the tensorfields")
		("volume", boost::program_options::value<std::string>(&in_image), "optional image field corresponding to the tensors")
		("gamma, g", boost::program_options::value<float>(&in_gamma)->default_value(3), "glyph gamma (sharpness), 0 = spheroids")
		("cmap,c", boost::program_options::value<int>(&in_cmap)->default_value(0), "colormaped eigenvector (0 = longest, 2 = shortest)")
		("cuda", boost::program_options::value<int>(&in_device)->default_value(0), "CUDA device ID (-1 for CPU only)")
		("help", "produce help message");
	boost::program_options::variables_map vm;

	// make sure there the specified CUDA device is available (otherwise switch to CPU)
	int ndevices;
	cudaError_t error = cudaGetDeviceCount(&ndevices);
	if (error != cudaSuccess) ndevices = 0;
	if (ndevices <= in_device) {
		std::cout << "WARNING: Specified CUDA device " << in_device << " is unavailable (" << ndevices << " compatible devices found), defaulting to CPU" << std::endl;
		in_device = -1;
	}

	boost::program_options::positional_options_description p;
	p.add("input", -1);
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (in_cmap != 0 && in_cmap != 2) {
		std::cout << "ERROR: invalid colormap component. Options (0 = longest, 2 = shortest)." << std::endl;
		return 1;
	}

	// Initialize OpenGL
	window = InitGLFW();                                // create a GLFW window
	InitUI(window, glsl_version);
	InitGLEW();

	// Enable OpenGL environment parameters
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);


	SCALAR_MATERIAL = new tira::glMaterial(scalar_shader_string);
	AXIS_MATERIAL = new tira::glMaterial(AXIS_MATERIAL_string);


	// If the tensor field is loaded using command-line argument
	if (vm.count("input")) {
		LoadTensorField3(in_filename);
		OPEN_TENSOR = false;
	}

	float gamma = in_gamma;
	int component_color = in_cmap;

	// Set light position
	glm::vec4 light0(0.0f, 100.0f, 100.0f, 0.7f);
	glm::vec4 light1(0.0f, -100.0f, 0.0f, 0.5f);
	float ambient = 0.3;

	tira::geometry<float> r = tira::rectangle<float>();
	r = r.scale({0.5f, 0.5f});
	r = r.scale({0.5f, 0.5f}, 1);
	r = r.translate({-0.25, -0.25});

	planes[0][0] = r.translate({ 0.0f, 0.0f }).translate({ 0.0f, 0.0f }, 1);
	planes[0][1] = r.translate({ 0.5f, 0.0f }).translate({ 0.5f, 0.0f }, 1);
	planes[1][0] = r.translate({ 0.0f, 0.5f }).translate({ 0.0f, 0.5f }, 1);
	planes[1][1] = r.translate({ 0.5f, 0.5f }).translate({ 0.5f, 0.5f }, 1);

	axis = new tira::glGeometry();
	*axis = tira::glGeometry::GenerateCylinder<float>(10, 20);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();													// Poll and handle events (inputs, window resize, etc.)
		RenderUI();
		int display_w, display_h;                                           // size of the frame buffer (openGL display)
		glfwGetFramebufferSize(window, &display_w, &display_h);

		float aspect = (float)display_w / (float)display_h;
		glViewport(0, 0, display_w, display_h);									// specifies the area of the window where OpenGL can render
		glm::mat4 Mproj;
		if (perspective)
			Mproj = Camera.perspectivematrix(aspect);
		else
			Mproj = Camera.orthomatrix(aspect, zoom, move[0], move[1]);
		glm::mat4 Mview = Camera.viewmatrix();

		// If the load command for tensor field is called from ImGui file dialog
		if (OPEN_TENSOR) {
			LoadTensorField3(TensorFileName);								// Load the tensor field and set the texture-map
			OPEN_TENSOR = false;
		}

		if (TENSOR_LOADED && SET_CAMERA) {
			Camera.position({ 0.0f, 0.0f, -2 * Tn.smax() });
			Camera.up({ 0.0f, 1.0f, 0.0f });
			Camera.lookat({ Tn.X() / 2.0f, Tn.Y() / 2.0f, Tn.Z() / 2.0f });
			Camera.fov(60);
			SET_CAMERA = false;
		}
		else if (TENSOR_LOADED && RESET) {
			// Reset the visualization to initial state if reset button is pushed
			resetPlane();
		}

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		SCALAR_MATERIAL->Begin();
		SCALAR_MATERIAL->SetUniformMat4f("Mview", Mproj * Mview);

		if (TENSOR_LOADED) {
			// Enable alpha blending for transparency and set blending function
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		RenderPlanes();
		SCALAR_MATERIAL->End();

		// Draw the axes
		if (TENSOR_LOADED) {
			glDisable(GL_BLEND);
			AXIS_MATERIAL->Bind();
			if (RENDER_PLANE[0]) draw_axes(Mproj * Mview, 0);
			if (RENDER_PLANE[1]) draw_axes(Mproj * Mview, 1);
			if (RENDER_PLANE[2]) draw_axes(Mproj * Mview, 2);
		}

		glClear(GL_DEPTH_BUFFER_BIT);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	DestroyUI();
	glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
	glfwTerminate();                                                // Terminate GLFW

}