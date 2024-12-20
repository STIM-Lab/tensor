#include <numbers>
#include <complex>
#include <chrono>
#include <limits>

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
void cudaEigendecomposition3D(float* tensors, float*& evals, float*& evecs, unsigned int n, int device);
void cudaVote3D(float* input_field, float* output_field, float* L, float* V, unsigned int s0, unsigned int s1, unsigned int s2, float sigma, float sigma2,
	unsigned int w, unsigned int power, unsigned int device, bool STICK, bool PLATE, bool debug);


GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL

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
std::vector<float> in_voxelsize;
float epsilon = 0.001;

// rendering variables
tira::camera Camera;

tira::glMaterial* SCALAR_MATERIAL;
tira::glMaterial* PLANES_MATERIAL;
tira::glMaterial* AXIS_MATERIAL;

tira::glGeometry* axis;
tira::glGeometry plane;

enum ScalarType { NoScalar, TensorElement, EVal, EVec, Anisotropy };
int SCALAR_TYPE = ScalarType::EVal;
int SCALAR_EVAL = 2;
int SCALAR_EVEC = 2;
int SCALAR_ANISOTROPY = 0;
enum ProcessingType { NoProcessing, Gaussian, Vote };
int PROCESSINGTYPE = ProcessingType::NoProcessing;

// scalar plane variables
bool RENDER_PLANE[] = { false, false, false };
int PLANE_POSITION[] = { 0, 0, 0 };
bool RENDER_EN_FACE = false;
float EN_FACE_POSITION = 0.0f;
bool RENDER_VOLUMETRIC = true;
int VOLUMETRIC_PLANES = 10;

// shader string files
const std::string glyph_shader_string =
#include "shaders/glyph3d.shader"
;
const std::string volume_shader_string =
#include "shaders/volume.shader"
;
const std::string enface_shader_string =
#include "shaders/volumemap_enface.shader"
;
const std::string planes_shader_string =
#include "shaders/volumemap_planes.shader"
;
const std::string axis_material_string =
#include "shaders/axis.shader"
;

// cropping variables
static int position_values[3] = { 0, 0, 0 };
static int width_values[3] = { 0, 0, 0 };

float ui_scale = 1.5f;                                  // scale value for the UI and UI text
bool RESET = false;
bool window_focused = true;
bool axis_change = true;                                // gets true when the axis plane is changes
extern int step;                                        // the steps between each glyph along all axis

/// profiling Timers
float inf = std::numeric_limits<float>::infinity();
float t_eigendecomposition = inf;
float t_loading = inf;
float t_resetfield = inf;
float t_cmap_evec = inf;
float t_cmap_eval = inf;
float t_cmap_fa = inf;
float t_cmap_linear = inf;
float t_cmap_plate = inf;
float t_cmap_sphere = inf;
float t_gaussian = inf;

auto tic() {
	return std::chrono::high_resolution_clock::now();
}
float duration(auto t0, auto t1) {
	std::chrono::duration< float > fs = t1 - t0;
	return fs.count();
}

// tensor fields
int anisotropy = 0;                                     // 0: all tensors               1: linear tensors only
														// 2: planar tensors only       3: spherical tensors only
float filter = 0.1f;
float zoom = 1.0f;
int cmap = 1;
float opacity = 1.0f;
float alpha = 1.0f;
float thresh = 0.0f;
bool perspective = true;

bool OPEN_TENSOR = false;
bool RENDER_GLYPHS = false;
bool TENSOR_LOADED = false;
bool BLUR = false;
float SIGMA;
float TV_SIGMA1 = 3;
float TV_SIGMA2 = 0;
bool TV_STICK = true;
bool TV_PLATE = false;
int TV_POWER = 1;
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
	window = glfwCreateWindow(1600, 1200, "TensorView 3D", NULL, NULL);
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

tira::volume<float> FA() {
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
	return A;
}

void ColormapEvec(size_t vi) {
	auto t0 = tic();
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

	tira::volume<float> Anisotropy = FA();
	tira::volume<float> TensorSize = Ln.channel(2);	
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
	PLANES_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);

	tira::volume<unsigned char> Alpha = Anisotropy * (TensorSize * (1.0f / Ln.maxv())) * 255;
	SCALAR_MATERIAL->SetTexture("opacity", Alpha, GL_LUMINANCE, GL_NEAREST);

	auto t1 = tic();
	t_cmap_evec = duration(t0, t1);	
}

void ColormapEval(int eval) {
	auto t0 = tic();
	tira::volume<float> E = Ln.channel(eval);
	tira::volume<unsigned char> C = E.cmap(ColorMap::Magma);

	//tira::volume<float> Anisotropy = FA();
	tira::volume<float> TensorSize = Ln.channel(2);

	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
	PLANES_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);

	tira::volume<unsigned char> Alpha = TensorSize * (1.0f / Ln.maxv()) * 255;
	SCALAR_MATERIAL->SetTexture("opacity", Alpha, GL_LUMINANCE, GL_NEAREST);
	auto t1 = tic();
	t_cmap_eval = duration(t0, t1);
}

void ColormapFA() {
	auto t0 = tic();
	tira::volume<float> A = FA();	
	tira::volume<unsigned char> C = A.cmap(0, 1, ColorMap::Magma);
	SCALAR_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);
	PLANES_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);

	tira::volume<unsigned char> Alpha = (tira::volume<unsigned char>)(A * 255);
	SCALAR_MATERIAL->SetTexture("opacity", Alpha, GL_LUMINANCE, GL_NEAREST);
	auto t1 = tic();
	t_cmap_fa = duration(t0, t1);
}

void ColormapLinearA() {
	auto t0 = tic();
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
	PLANES_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);

	tira::volume<unsigned char> Alpha = A * 255;
	SCALAR_MATERIAL->SetTexture("opacity", Alpha, GL_LUMINANCE, GL_NEAREST);
	auto t1 = tic();
	t_cmap_linear = duration(t0, t1);
}

void ColormapPlateA() {
	auto t0 = tic();
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
	PLANES_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);

	tira::volume<unsigned char> Alpha = A * 255;
	SCALAR_MATERIAL->SetTexture("opacity", Alpha, GL_LUMINANCE, GL_NEAREST);
	auto t1 = tic();
	t_cmap_plate = duration(t0, t1);
}

void ColormapSphereA() {
	auto t0 = tic();
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
	PLANES_MATERIAL->SetTexture("mapped_volume", C, GL_RGB, GL_NEAREST);

	tira::volume<unsigned char> Alpha = A * 255;
	SCALAR_MATERIAL->SetTexture("opacity", Alpha, GL_LUMINANCE, GL_NEAREST);
	auto t1 = tic();
	t_cmap_sphere = duration(t0, t1);
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

void ResetField() {
	auto t0 = tic();
	Tn = T0;
	auto t1 = tic();
	t_resetfield = duration(t0, t1);

	position_values[0] = Tn.sx() / 2;
	position_values[1] = Tn.sy() / 2;
	position_values[2] = Tn.sz() / 2;

	width_values[0] = Tn.sx();
	width_values[1] = Tn.sy();
	width_values[2] = Tn.sz();
}

/// <summary>
/// Blurs the tensor field and re-calculates the current scalar image
/// </summary>
/// <param name="sigma"></param>
void GaussianFilter(const float sigma) {
	auto t0 = tic();
	// if a CUDA device is enabled, use a blur kernel
	if(sigma == 0) {
		ResetField();
		return;
	}
	if (in_device >= 0) {
		std::cout << "Gaussian on CUDA" << std::endl;
		unsigned int blur_width;
		unsigned int blur_height;
		unsigned int blur_depth;
		glm::mat3* blurred = cudaGaussianBlur3D(T0.data(), T0.X(), T0.Y(), T0.Z(), 
			sigma / T0.dx(), sigma / T0.dy(), sigma / T0.dz(),
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
	auto t1 = tic();
	t_gaussian = duration(t0, t1);
}

void TensorVote(const float sigma, const float sigma2, const unsigned int p, const bool stick, const bool plate) {
	Tn = tira::volume<glm::mat3>(T0.X(), T0.Y(), T0.Z());								// allocate space for the new field

	const auto w = static_cast<unsigned int>(6.0f * std::max(sigma, sigma2) + 1.0f);	// calculate the window size for the vote field
	cudaVote3D(reinterpret_cast<float*>(T0.data()), reinterpret_cast<float*>(Tn.data()), Ln.data(), Vn.data(),
		static_cast<unsigned int>(T0.shape()[0]), static_cast<unsigned int>(T0.shape()[1]), static_cast<unsigned int>(T0.shape()[2]),
		sigma, sigma2, w, p, in_device, stick, plate, false);
}

void UpdateEigens() {
	auto t0 = tic();
	float* eigenvalues_raw;
	float* eigenvectors_raw;
	cudaEigendecomposition3D((float*)Tn.data(), eigenvalues_raw, eigenvectors_raw, Tn.size(), in_device);
	Ln = tira::volume<float>(eigenvalues_raw, Tn.X(), Tn.Y(), Tn.Z(), 3);
	Vn = tira::volume<float>(eigenvectors_raw, Tn.X(), Tn.Y(), Tn.Z(), 4);
	free(eigenvalues_raw);
	free(eigenvectors_raw);
	auto t1 = tic();
	t_eigendecomposition = duration(t0, t1);
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

void UpdateCamera() {
	Camera.lookat({ Tn.sx() / 2.0f, Tn.sy() / 2.0f, Tn.sz() / 2.0f });
	Camera.distance(Tn.smax() * 2);
}

void ResetPlanes() {

	PLANE_POSITION[0] = static_cast<int>(Tn.X() / 2);
	PLANE_POSITION[1] = static_cast<int>(Tn.Y() / 2);
	PLANE_POSITION[2] = static_cast<int>(Tn.Z() / 2);
	EN_FACE_POSITION = 0.0f;
	VOLUMETRIC_PLANES = std::max(Tn.X(), std::max(Tn.Y(), Tn.Z()));
}

void LoadDefaultField() {
	T0.resize({ 21, 21, 21 });

	glm::mat3 t;
	for (size_t zi = 0; zi < T0.Z(); zi++) {
		for (size_t yi = 0; yi < T0.Y(); yi++) {
			for (size_t xi = 0; xi < T0.X(); xi++) {
				t[0][0] = T0.px(xi) * T0.px(xi);
				t[0][1] = T0.px(xi) * T0.py(yi);
				t[0][2] = T0.px(xi) * T0.pz(zi);
				t[1][0] = t[0][1];
				t[1][1] = T0.py(yi) * T0.py(yi);
				t[1][2] = T0.py(yi) * T0.pz(zi);
				t[2][0] = t[0][2];
				t[2][1] = t[1][2];
				t[2][2] = T0.pz(zi) * T0.pz(zi);
				T0(xi, yi, zi) = t;
			}
		}
	}

	ResetField();

	ResetPlanes();

	TENSOR_LOADED = true;													// mark a tensor field as loaded

	UpdateCamera();

	std::cout << "Tensor loaded successfully.\n" << std::endl;
	std::cout << "Size of volume:\t(" << T0.X() << " x " << T0.Y() << " x " << T0.Z() << ")" << std::endl;
	std::cout << "World size:\t(" << T0.sx() << " x " << T0.sy() << " x " << T0.sz() << ")" << std::endl;

	UpdateEigens();
	ColormapScalar();
}

// Load a tensor field from a NumPy file
void LoadTensorField(std::string npy_filename) {
	// Load the tensor field
	auto t0 = tic();
	T0.load_npy<float>(npy_filename);
	auto t1 = tic();
	t_loading = duration(t0, t1);

	T0.spacing(in_voxelsize[0], in_voxelsize[1], in_voxelsize[2]);
	ResetField();

	ResetPlanes();

	TENSOR_LOADED = true;													// mark a tensor field as loaded

	UpdateCamera();

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

		window_focused = (ImGui::IsAnyItemHovered()) ? false : true;

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
				if (file_dialog.showFileDialog("Open File", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 700), ".npy"))
					OpenFileDialog();

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				if (ImGui::InputFloat3("Voxel Sizes", &in_voxelsize[0])) {
					if (in_voxelsize[0] < epsilon) in_voxelsize[0] = epsilon;
					if (in_voxelsize[1] < epsilon) in_voxelsize[1] = epsilon;
					if (in_voxelsize[2] < epsilon) in_voxelsize[2] = epsilon;
					T0.spacing(in_voxelsize[0], in_voxelsize[1], in_voxelsize[2]);
					Tn.spacing(in_voxelsize[0], in_voxelsize[1], in_voxelsize[2]);
					UpdateCamera();
				}

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::PushItemWidth(-100);
				ImGui::SliderFloat("alpha", &alpha, 0.01f, 1.0f, "%.2f");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				///////////////////////////////////////////////  Render Planes  //////////////////////////////////////////////////
				ImGui::SeparatorText("Planes");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::Checkbox("X", &RENDER_PLANE[0]);
				ImGui::SetItemTooltip("Render plane X of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("##XPosition", &PLANE_POSITION[0], 0, Tn.X() - 1);

				ImGui::Checkbox("Y", &RENDER_PLANE[1]);
				ImGui::SetItemTooltip("Render plane Y of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("##YPosition", &PLANE_POSITION[1], 0, Tn.Y() - 1);

				ImGui::Checkbox("Z", &RENDER_PLANE[2]);
				ImGui::SetItemTooltip("Render plane Z of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("##ZPosition", &PLANE_POSITION[2], 0, Tn.Z() - 1);

				ImGui::Checkbox("F", &RENDER_EN_FACE);
				ImGui::SetItemTooltip("Render a camera-oriented en face plane.");
				ImGui::SameLine();
				ImGui::SliderFloat("##EFPosition", &EN_FACE_POSITION, -Tn.smax(), Tn.smax() );

				///////////////////////////////////////////////  Render Volume  //////////////////////////////////////////////////
				ImGui::SeparatorText("Volume");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::Checkbox("V", &RENDER_VOLUMETRIC);
				ImGui::SetItemTooltip("Render a camera-oriented en face plane.");
				ImGui::SameLine();
				ImGui::SliderInt("##VPlanes", &VOLUMETRIC_PLANES, 0, 4 * std::max(Tn.X(), std::max(Tn.Y(), Tn.Z())));
				ImGui::Dummy(ImVec2(0.0, 5.0f));

				///////////////////////////////////////////////   Cropping   ///////////////////////////////////////////////////
				ImGui::Dummy(ImVec2(0.0, 5.0f));

				ImGui::Indent();
				ImGui::Text("Position");
				ImGui::SameLine();
				ImGui::Text("Width");
				ImGui::Unindent();

				float child_width = ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x * 2 - ImGui::GetStyle().ItemInnerSpacing.x * 2;

				// position control
				ImGui::BeginChild("position", ImVec2(child_width / 2.0f, 175), ImGuiChildFlags_Border);
				for (int i = 0; i < 3; i++) {
					if (i > 0) ImGui::SameLine();
					ImGui::PushID(i);
					ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.5f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.6f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.7f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.9f, 0.9f));
					int size = (i == 0) ? Tn.sx() : ((i == 1) ? Tn.sy() : Tn.sz());
					ImGui::VSliderInt("##pos", ImVec2(50, 150), &position_values[i], 1, size);
					ImGui::PopStyleColor(4);
					ImGui::PopID();
				}
				ImGui::EndChild();
				ImGui::SameLine();

				// width control
				ImGui::BeginChild("width", ImVec2(child_width / 2.0f, 175), ImGuiChildFlags_Border);
				for (int i = 0; i < 3; i++) {
					ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(child_width / 2.0f, child_width / 2.0f));
					if (i > 0) ImGui::SameLine();
					ImGui::PushID(i);
					ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.5f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.6f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.7f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.9f, 0.9f));
					int width_max = (i == 0) ? Tn.sx() : ((i == 1) ? Tn.sy() : Tn.sz());
					ImGui::VSliderInt("##width", ImVec2(50, 150), &width_values[i], 0, width_max);
					ImGui::PopStyleColor(4);
					ImGui::PopStyleVar();
					ImGui::PopID();
				}
				ImGui::EndChild();

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
				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::SeparatorText("Filter");
				ImGui::Dummy(ImVec2(0.0, 5.0f));

				if (ImGui::RadioButton("None", &PROCESSINGTYPE, (int)ProcessingType::NoProcessing)) {
					ResetField();
					UpdateEigens();
					ScalarRefresh();
				}
				ImGui::Dummy(ImVec2(0.0f, 5.0f));
				///////////////////////////////////////////////  Gaussian Blur  ///////////////////////////////////////////////////
				
				if (ImGui::RadioButton("Gaussian Blur", &PROCESSINGTYPE, (int)ProcessingType::Gaussian)) {
					if (PROCESSINGTYPE == ProcessingType::Gaussian) {
						GaussianFilter(SIGMA);
						UpdateEigens();
						ScalarRefresh();
					}
				}
				ImGui::SameLine();
				if (ImGui::InputFloat("##Sigma", &SIGMA, 0.2f, 1.0f)) {
					if (SIGMA < 0.0f) SIGMA = 0.0f;
					if (PROCESSINGTYPE == ProcessingType::Gaussian) {
						GaussianFilter(SIGMA);
						UpdateEigens();
						ScalarRefresh();
					}
				}

				///////////////////////////////////////////////  Tensor Voting  ///////////////////////////////////////////////////

				if (ImGui::RadioButton("Tensor Voting", &PROCESSINGTYPE, (int)ProcessingType::Vote)) {
					ResetField();
					TensorVote(TV_SIGMA1, TV_SIGMA2, TV_POWER, TV_STICK, TV_PLATE);
					UpdateEigens();
					ScalarRefresh();
				}

				if (ImGui::InputFloat("Sigma 1", &TV_SIGMA1, 0.2f, 1.0f)) {
					if (TV_SIGMA1 < 0) TV_SIGMA1 = 0.0;
					if (PROCESSINGTYPE == ProcessingType::Vote) {
						TensorVote(TV_SIGMA1, TV_SIGMA2, TV_POWER, TV_STICK, TV_PLATE);
						UpdateEigens();
						ScalarRefresh();
					}
				}
				if (ImGui::InputFloat("Sigma 2", &TV_SIGMA2, 0.2f, 1.0f)) {
					if (TV_SIGMA2 < 0) TV_SIGMA2 = 0.0;
					if (PROCESSINGTYPE == ProcessingType::Vote) {
						TensorVote(TV_SIGMA1, TV_SIGMA2, TV_POWER, TV_STICK, TV_PLATE);
						UpdateEigens();
						ScalarRefresh();
					}
				}
				if (ImGui::InputInt("Power", &TV_POWER, 1, 5)) {
					if (TV_POWER < 1) TV_POWER = 1;
					if (PROCESSINGTYPE == ProcessingType::Vote) {
						TensorVote(TV_SIGMA1, TV_SIGMA2, TV_POWER, TV_STICK, TV_PLATE);
						UpdateEigens();
						ScalarRefresh();
					}
				}
				if (ImGui::Checkbox("Stick", &TV_STICK)) {
					TensorVote(TV_SIGMA1, TV_SIGMA2, TV_POWER, TV_STICK, TV_PLATE);
					UpdateEigens();
					ScalarRefresh();
				}
				ImGui::SameLine();
				if (ImGui::Checkbox("Plate", &TV_PLATE)) {
					TensorVote(TV_SIGMA1, TV_SIGMA2, TV_POWER, TV_STICK, TV_PLATE);
					UpdateEigens();
					ScalarRefresh();
				}
				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Profiling")) {
				///////////////////////////////////////////////  Memory bar  ///////////////////////////////////////////////////
				ImGui::SeparatorText("CUDA Device");
				ImGui::Text("Device ID: %d", in_device);
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				float free_m, total_m, used_m;
				size_t free_t, total_t;
				char buf[64];
				if (in_device >= 0) {
					cudaMemGetInfo(&free_t, &total_t);
					free_m = static_cast<float>(free_t) / (1048576.0f);
					total_m = static_cast<float>(total_t) / (1048576.0f);
					used_m = total_m - free_m;
				}
				else {								// CPU memory will be added later
					used_m = 0.0f;
					total_m = 0.0f;
				}
				sprintf(buf, "%.1f/%.1f MB", used_m, total_m);
				float bar_size = ImGui::GetWindowWidth() - ImGui::CalcTextSize("Memory Usage").x -
					ImGui::GetStyle().WindowPadding.x * 2 - ImGui::GetStyle().ItemInnerSpacing.x;
				ImGui::ProgressBar((used_m / total_m), ImVec2(bar_size, 0.0f), buf);
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Memory Usage");
				ImGui::EndTabItem();

				ImGui::SeparatorText("Timers");
				ImGui::Text("Load Field: %f s", t_loading);
				ImGui::Text("Reset Field: %f s", t_resetfield);
				ImGui::Text("Eigendecomposition Calculation: %f s", t_eigendecomposition);
				ImGui::Text("Colormap Eigenvalues: %f s", t_cmap_eval);
				ImGui::Text("Colormap Eigenvectors: %f s", t_cmap_evec);
				ImGui::Text("Colormap Fractional Anisotropy: %f s", t_cmap_fa);
				ImGui::Text("Colormap Linear Anisotropy: %f s", t_cmap_linear);
				ImGui::Text("Colormap Plate Anisotropy: %f s", t_cmap_plate);
				ImGui::Text("Colormap Spherical Anisotropy: %f s", t_cmap_sphere);
				ImGui::Text("Gaussian Blur: %f s", t_gaussian);
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

	// calculate the transformation matrix for each plane
	if (p == 0) {
		pos = glm::vec3((float)Tn.px(PLANE_POSITION[0]), Tn.sy() / 2.0f, Tn.sz() / 2.0f);
		rot = glm::vec3(0.0f, 1.0f, 0.0f);
		tpos = glm::vec3((float)Tn.px(PLANE_POSITION[0]), 0.0f, 0.0f);
		tscale = glm::vec3(1.0f, Tn.sy(), Tn.sz());
	}
	else if (p == 1) {
		pos = glm::vec3(Tn.sx() / 2.0f, (float)Tn.py(PLANE_POSITION[1]), Tn.sz() / 2.0f);
		rot = glm::vec3(1.0f, 0.0f, 0.0f);
		tpos = glm::vec3(0.0f, (float)Tn.py(PLANE_POSITION[1]), 0.0f);
		tscale = glm::vec3(Tn.sx(), 1.0f, Tn.sz());
	}
	else if (p == 2) {
		pos = glm::vec3(Tn.sx() / 2.0f, Tn.sy() / 2.0f, (float)Tn.pz(PLANE_POSITION[2]));
		rot = glm::vec3(0.0f, 0.0f, 1.0f);
		tpos = glm::vec3(0.0f, 0.0f, (float)Tn.pz(PLANE_POSITION[2]));
		tscale = glm::vec3(Tn.sx(), Tn.sy(), 1.0f);
	}

	float radians = (p == 1) ? glm::radians(90.0f) : glm::radians(-90.0f);
	if (p == 2) radians = glm::radians(0.0f);
	Mobj = glm::translate(Mobj, pos);
	Mobj = glm::scale(Mobj, scale);
	Mobj = glm::rotate(Mobj, radians, rot);

	Mtex = glm::translate(Mtex, pos);
	Mtex = glm::scale(Mtex, tscale);
	Mtex = glm::rotate(Mtex, radians, rot);

	PLANES_MATERIAL->SetUniformMat4f("Mobj", Mobj);
	PLANES_MATERIAL->SetUniformMat4f("Mtex", Mtex);
	plane.Draw();
}

void RenderEnFacePlane(float p) {
	glm::mat4 Mrot = glm::mat4(1.0f);
	

	glm::vec3 zp = Camera.view();
	glm::vec3 yp = Camera.up();
	glm::vec3 xp = Camera.side();

	Mrot[0][0] = xp[0];
	Mrot[0][1] = xp[1];
	Mrot[0][2] = xp[2];

	Mrot[1][0] = yp[0];
	Mrot[1][1] = yp[1];
	Mrot[1][2] = yp[2];

	Mrot[2][0] = zp[0];
	Mrot[2][1] = zp[1];
	Mrot[2][2] = zp[2];

	glm::mat4 Mobj(1.0f);
	
	Mobj = glm::translate(Mobj, glm::vec3(Tn.sx() / 2.0f, Tn.sy() / 2.0f, Tn.sz() / 2.0f));
	Mobj = glm::translate(Mobj, p * Camera.view());
	float plane_size = std::sqrt(3 * Tn.smax() * Tn.smax());
	Mobj = glm::scale(Mobj, glm::vec3(plane_size, plane_size, plane_size));
	Mobj = Mobj * Mrot;
	
	glm::vec3 tscale(1.0f / Tn.sx(), 1.0f / Tn.sy(), 1.0f / Tn.sz());
	glm::mat4 Mobj2tex = glm::scale(glm::mat4(1.0f), tscale);

	glm::vec4 test(0.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 t = Mobj * test;

	SCALAR_MATERIAL->SetUniformMat4f("Mobj", Mobj);
	SCALAR_MATERIAL->SetUniformMat4f("Mtex", Mobj2tex * Mobj);

	plane.Draw();
}

void RenderVolumetric(unsigned int n) {

	float depth = Tn.smax();
	float dfar = depth / 2.0f;
	float dstep = depth / n;

	for (unsigned int di = 0; di < n; di++) {
		RenderEnFacePlane(dfar - di * dstep);
	}

}

void RenderVolume() {
	if (RENDER_EN_FACE) RenderEnFacePlane(EN_FACE_POSITION);
	if (RENDER_VOLUMETRIC) RenderVolumetric(VOLUMETRIC_PLANES);
}

void RenderPlanes() {
	if (RENDER_PLANE[0]) RenderPlane(0);
	if (RENDER_PLANE[1]) RenderPlane(1);
	if (RENDER_PLANE[2]) RenderPlane(2);
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
		("voxel", boost::program_options::value<std::vector<float>>(&in_voxelsize)->multitoken()->default_value(std::vector<float>{1.0f, 1.0f, 1.0f}, "1.0 1.0 1.0"), "voxel size")
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


	SCALAR_MATERIAL = new tira::glMaterial(enface_shader_string);
	PLANES_MATERIAL = new tira::glMaterial(planes_shader_string);
	AXIS_MATERIAL = new tira::glMaterial(axis_material_string);


	// If the tensor field is loaded using command-line argument
	if (vm.count("input")) {
		LoadTensorField(in_filename);
	}
	else {
		LoadDefaultField();
	}

	float gamma = in_gamma;
	int component_color = in_cmap;

	// Set light position
	glm::vec4 light0(0.0f, 100.0f, 100.0f, 0.7f);
	glm::vec4 light1(0.0f, -100.0f, 0.0f, 0.5f);
	float ambient = 0.3;

	plane = tira::rectangle<float>().translate({ -0.5f, -0.5f }, 1);
	axis = new tira::glGeometry();
	*axis = tira::glGeometry::GenerateCylinder<float>(10, 20);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();													// Poll and handle events (inputs, window resize, etc.)
		RenderUI();
		int display_w, display_h;                                           // size of the frame buffer (openGL display)
		glfwGetFramebufferSize(window, &display_w, &display_h);

		float aspect = (float)display_w / (float)display_h;
		glViewport(0, 0, display_w, display_h);								// specifies the area of the window where OpenGL can render
		glm::mat4 Mproj;													// calculate the projection matrix from the camera
		if (perspective) Mproj = Camera.perspectivematrix(aspect);
		else Mproj = Camera.orthomatrix(aspect, zoom, move[0], move[1]);

		glm::mat4 Mview = Camera.viewmatrix();								// get the view transformation matrix

		if (OPEN_TENSOR) {
			LoadTensorField(TensorFileName);
			OPEN_TENSOR = false;
		}

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		if (TENSOR_LOADED) {
			// Enable alpha blending for transparency and set blending function
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		SCALAR_MATERIAL->Begin();
		SCALAR_MATERIAL->SetUniformMat4f("Mview", Mproj * Mview);
		SCALAR_MATERIAL->SetUniform3f("crop_pos", (float)position_values[0] / Tn.sx(),
			(float)position_values[1] / Tn.sy(), (float)position_values[2] / Tn.sz());
		SCALAR_MATERIAL->SetUniform3f("crop_wid", (float)width_values[0] / Tn.sx(), 
			(float)width_values[1] / Tn.sy(), (float)width_values[2] / Tn.sz());
		SCALAR_MATERIAL->SetUniform1f("alpha", alpha);
		RenderVolume();
		SCALAR_MATERIAL->End();
		PLANES_MATERIAL->Begin();
		PLANES_MATERIAL->SetUniformMat4f("Mview", Mproj * Mview);
		PLANES_MATERIAL->SetUniform1f("alpha", alpha);
		RenderPlanes();
		PLANES_MATERIAL->End();

		// Draw the axes
		if (TENSOR_LOADED) {
			glDisable(GL_BLEND);
			AXIS_MATERIAL->Bind();
			if (RENDER_PLANE[0]) draw_axes(Mproj * Mview, 0);
			if (RENDER_PLANE[1]) draw_axes(Mproj * Mview, 1);
			if (RENDER_PLANE[2]) draw_axes(Mproj * Mview, 2);
			AXIS_MATERIAL->End();
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
