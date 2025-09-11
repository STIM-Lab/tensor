#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers


#include <glm/glm.hpp>

#include <tira/volume.h>
#include <tira/graphics/camera.h>

static float inf = std::numeric_limits<float>::infinity();
enum ProcessingType { NoProcessing, Gaussian, Vote };
enum ScalarType { NoScalar, Tensor00, Tensor01, Tensor11, Tensor02, Tensor12, Tensor22, EVal0, EVal1, EVal2, EVec0, EVec1, EVec2, FractionalAnisotropy };

struct TV3_UI {
	float scale = 2.0f;					// user interface scale (font size, etc.)

	bool window_focused = true;			// indicates that the user's mouse is hovered over the UI
	bool mouse_dragging = false;
	double raw_mouse_position[2];
	tira::camera camera;
	bool perspective = true;
	bool ctrl_pressed = false;

	// impulse window inputs
	bool impulse_window = false;
	bool impulse_field_active = false;
	int impulse_resolution = 13;
	glm::vec2 impulse_stick = glm::vec2(0.0f);
	float impulse_plate = 0.0f;
	glm::vec2 impulse_lambdas = glm::vec2(1.0f);
	float impulse_anisotropy = 0;

	// settings for tensor voting
	bool tv_stick = true;
	bool tv_plate = false;
	int platevote_samples = 4;
	float tv_sigma1 = 3.0f;
	float tv_sigma2 = 1.0f;
	int tv_power = 1;

	// visualization
	glm::vec3 slice_positions = glm::vec3(0.0f);

	std::string loaded_filename = "";
	bool field_loaded = false;			// indicates that a tensor field has been loaded
	bool image_loaded = false;

	int cuda_device;					// CUDA device ID (-1 for CPU)

	// settings for processing the tensor field
	int processing_type = ProcessingType::NoProcessing;

	float sigma = 1.0f;						// sigma for Gaussian blur

	// display settings for scalar fields
	int scalar_type = ScalarType::Tensor00;

	//timers
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
};



// File IO function declarations
void LoadTensorField(std::string npy_filename, tira::volume<glm::mat3>* tensor);
glm::mat3 GenerateImpulse(glm::vec2 stick_polar, float plate_theta, glm::vec2 lambdas);
void GenerateImpulseField(tira::volume<glm::mat3>* tensor, unsigned resolution, glm::vec2 stick_polar, float plate_theta, glm::vec2 lambdas);


//void LoadTensorField(const std::string& filename, tira::image<glm::mat2>* tensor);
//void SaveTensorField(const std::string& filename, tira::image<glm::mat2>* tensor);
//void GenerateImpulse(tira::image<glm::mat2>* tensor, unsigned resolution, float theta, float anisotropy);

// Function to initialize Opengl, including the rendering context, window, and callback functions

GLFWwindow* InitWindow(int width, int height);

// Function to initialize the 3D objects that will be rendered through OpenGL

//void InitShaders();
//void InitCmapGeometry();
//void GenerateGlyphs();

// Draw function for the 3D rendering loop

//void RenderFieldOpenGL(GLint display_w, GLint display_h);

// ImGui display functions
void ImGuiRender();
void ImGuiInit(GLFWwindow* window, const char* glsl_version);
void ImGuiDestroy();
//void RefreshScalarField();

// callback function signatures
void glfw_error_callback(int error, const char* description);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

//void glfw_error_callback(int error, const char* description);
//void mouse_button_callback(GLFWwindow* window, const int button, const int action, const int mods);
//void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
//void scroll_callback(GLFWwindow* window, const double xoffset, const double yoffset);

// Data processing functions
void ReprocessTensors();
void ReprocessField();
void EigenDecomposition(tira::volume<glm::mat3>* tensor, tira::volume<float>* lambda, tira::volume<glm::vec2>* theta, int cuda_device);
void GaussianFilter(const tira::volume<glm::mat3>* tensors_in, tira::volume<glm::mat3>* tensors_out, const float sigma, 
	glm::vec3 pixel_size, const int cuda_device);
void TensorVote(const tira::volume<glm::mat3>* tensors_in, tira::volume<glm::mat3>* tensors_out, const float sigma, const float sigma2, 
	const unsigned int p, const bool stick, const bool plate, const int cuda_device, const unsigned samples);
void VolumeFrom_Eigenvalue(const tira::volume<float>* lambda, tira::volume<float>* scalar, const unsigned i);
void VolumeFrom_TensorElement3D(tira::volume<glm::mat3>* tensors, tira::volume<float>* elements, const unsigned int u, const unsigned int v);

// Visualization functions (generating colormaps, etc.)
void RefreshVisualization();
void UpdateColormap();
glm::vec3 ColormapEigenvector(unsigned vi, float l0, float l1, float l2, float theta, float phi, float l2_max = 1.0f);

// Heterogeneous system architecture calls
void hsa_tensorvote3(const float* input_field, float* output_field, unsigned int s0, unsigned int s1, unsigned int s2, float sigma, float sigma2, 
	unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug, unsigned samples);
float* hsa_eigenvalues3(float* tensors, unsigned int n, int device);
float* hsa_eigenvectors3spherical(float* tensors, float* evals, unsigned int n, int device);
glm::mat3* hsa_gaussian3(const glm::mat3* source, const unsigned int s0, const unsigned int s1, const unsigned int s2, const float sigma, glm::vec3 pixel_size,
	unsigned int& out_s0, unsigned int& out_s1, unsigned int& out_s2, const int deviceID = 0);