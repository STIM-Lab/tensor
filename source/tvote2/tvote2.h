#pragma once

#include <string>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <glm/glm.hpp>
#include <tira/image.h>

enum ScalarType { NoScalar, Tensor00, Tensor01, Tensor11, EVal0, EVal1, EVec0, EVec1, Eccentricity, LinearEccentricity };
enum ProcessingType { NoProcessing, Gaussian, Vote };
enum AdjustColorType { NoAdjustment, Darken, Lighten };

struct TV2_UI {
    float scale = 2.0f;         // user interface scale (font size, etc.)

    // if the current tensor field is from a file, store the name
    std::string loaded_filename = "";
    bool field_loaded = false;
    bool field_impulse = false;

    bool inspection_window = false;
    bool impulse_window = false;

    // GUI variables for controlling the impulse function parameters
    int impulse_resolution = 13;
    float impulse_theta = 0;
    float impulse_anisotropy = 0;

    // tensor display settings
    float largest_eigenvalue_magnitude = 1.0f;                   // largest eigenvalue in the tensor field (used to scale tensors)

    // display settings for scalar fields
    int scalar_type = ScalarType::EVec1;
    int eccentricity_color_mode = AdjustColorType::Lighten;
    int magnitude_color_mode = AdjustColorType::Darken;
    float magnitude_color_threshold = -1;

    // settings for processing the tensor field
    int processing_type = ProcessingType::NoProcessing;

    // processing settings for a gaussian blur
    float sigma = 1.0f;

    // processing for tensor voting
    bool stick_voting = true;
    bool plate_voting = true;
    int platevote_samples = 4;
    float sigma1 = 3.0f;
    float sigma2 = 0.0f;
    int vote_refinement = 1;

    // general tensor display settings
    int signed_eigenvalues = 0;             // display eigenvalues with a specific sign (-1 negative, +1 positive, 0 all)

    // general information about the scalar field
    float scalar_min;
    float scalar_max;

    // glyph display settings
    bool render_glyphs = true;
    float glyph_scale = 0.8;
    bool glyph_normalize = false;
    int glyph_tesselation = 100;

    // human-computer interface variables
    float raw_mouse_position[2];
    float field_mouse_position[2];
    float viewport[2];
    glm::vec2 camera_position;
    float camera_zoom = 1.0f;

    // CUDA device information
    int num_devices;
    int cuda_device = -1;
    std::vector<std::string> device_names;
};

// File IO function declarations
void LoadTensorField(const std::string& filename, tira::image<glm::mat2>* tensor);
void SaveTensorField(const std::string& filename, tira::image<glm::mat2>* tensor);
void GenerateImpulse(tira::image<glm::mat2>* tensor, unsigned resolution, float theta, float anisotropy);

// Function to initialize Opengl, including the rendering context, window, and callback functions
GLFWwindow* InitWindow(int width, int height);

// Function to initialize the 3D objects that will be rendered through OpenGL
void InitShaders();
void InitCmapGeometry();
void GenerateGlyphs();

// Draw function for the 3D rendering loop
void RenderFieldOpenGL(GLint display_w, GLint display_h);

// ImGui display functions
void ImGuiRender();
void ImGuiInit(GLFWwindow* window, const char* glsl_version);
void ImGuiDestroy();
void RefreshScalarField();

// callback function signatures
void glfw_error_callback(int error, const char* description);
void mouse_button_callback(GLFWwindow* window, const int button, const int action, const int mods);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, const double xoffset, const double yoffset);

// Data processing functions
glm::vec2 Eigenvector2D(glm::mat2 T, float lambda);
glm::vec2 Eigenvalues2D(glm::mat2 T);
float Eccentricity2(float l0, float l1);
float LinearEccentricity2(float l0, float l1);


void GaussianFilter(const tira::image<glm::mat2>* tensors_in, tira::image<glm::mat2>* tensors_out, const float sigma, int cuda_device);
void TensorVote(const tira::image<glm::mat2>* tensors_in, tira::image<glm::mat2>* tensors_out,
    const float sigma, const unsigned int p, const float sigma2, const bool stick, const bool plate, const int cuda_device, const unsigned samples);
void ImageFrom_Eccentricity(tira::image<float>* lambda, tira::image<float>* eccentricity);
void ImageFrom_LinearEccentricity(tira::image<float>* lambda, tira::image<float>* eccentricity);
void ImageFrom_Eigenvalue(const tira::image<float>* lambda, tira::image<float>* scalar, unsigned int i);
void ImageFrom_TensorElement2D(tira::image<glm::mat2>* tensors, tira::image<float>* elements, const unsigned int u, const unsigned int v);
void ImageFrom_Theta(const tira::image<float>* theta, tira::image<float>* scalar, unsigned int i);
void EigenDecomposition(tira::image<glm::mat2>* tensor, tira::image<float>* lambda, tira::image<float>* theta, int cuda_device);

// Visualization functions (generating colormaps, etc.)
void RefreshVisualization();

// Heterogeneous system architecture calls
void hsa_tensorvote2(const float* input_field, float* output_field, unsigned int s0, unsigned int s1, float sigma, float sigma2,
        unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug, unsigned samples);
float* hsa_eigenvectors2polar(float* tensors, float* evals, unsigned int n, int device);
float* hsa_eigenvalues2(float* tensors, unsigned int n, int device);
glm::mat2* hsa_gaussian2(const glm::mat2* source, const unsigned int width, const unsigned int height, const float sigma,
    unsigned int& out_width, unsigned int& out_height, const int deviceID = 0);

void InitializeCuda(int& device_id, std::vector<std::string>& device_names);