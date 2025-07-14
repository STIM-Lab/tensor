#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <tira/graphics/glOrthoView.h>
#include <tira/cmap.h>

#include "tvote3.h"

extern TV3_UI UI;
extern const char* glsl_version;                                // specify the version of GLSL
extern tira::glOrthoView<unsigned char>* OrthoViewer;
extern tira::volume<float> Scalar;								// scalar field that is currently being visualized
extern tira::volume<glm::vec2> ThetaPhi;
extern tira::volume<glm::mat3> Tn;

void RefreshVisualization() {
    OrthoViewer->slice_positions(UI.slice_positions);
}

void ColormapEigenvector(unsigned vi) {
    std::cout<<"Colormap Eigenvector"<<std::endl;
    for (unsigned zi=0; zi<OrthoViewer->Z(); zi++) {
        for (unsigned yi=0; yi<OrthoViewer->Y(); yi++) {
            for (unsigned xi=0; xi<OrthoViewer->X(); xi++) {
                glm::vec2 theta_phi = ThetaPhi(xi, yi, zi, vi);
                float cos_theta = std::cos(theta_phi.x);
                float sin_theta = std::sin(theta_phi.x);
                float cos_phi = std::cos(theta_phi.y);
                float sin_phi = std::sin(theta_phi.y);

                float x = std::abs(cos_theta * sin_phi);
                float y = std::abs(sin_theta * sin_phi);
                float z = std::abs(cos_phi);

                (*OrthoViewer)(xi, yi, zi, 0) = (unsigned char)(x * 255.0);
                (*OrthoViewer)(xi, yi, zi, 1) = (unsigned char)(y * 255.0);
                (*OrthoViewer)(xi, yi, zi, 2) = (unsigned char)(z * 255.0);
            }
        }
    }
}

void UpdateColormap() {
    OrthoViewer->resize(Tn.X(), Tn.Y(), Tn.Z(), 3);

    switch (UI.scalar_type) {
    case ScalarType::Tensor00:
    case ScalarType::Tensor01:
    case ScalarType::Tensor02:
    case ScalarType::Tensor11:
    case ScalarType::Tensor12:
    case ScalarType::Tensor22:
    case ScalarType::EVal0:
    case ScalarType::EVal1:
    case ScalarType::EVal2:
        tira::cmap::colormap(Scalar.data(), OrthoViewer->data(), Scalar.size(), ColorMap::Brewer);
        break;
    case ScalarType::EVec0:
        ColormapEigenvector(0);
        break;
    case ScalarType::EVec1:
        ColormapEigenvector(1);
        break;
    case ScalarType::EVec2:
        ColormapEigenvector(2);
        break;


    }
    OrthoViewer->update_texture();

    std::cout<<"Updating colormap"<<std::endl;
}

GLFWwindow* InitWindow(int width, int height) {

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1600, 1200, "TensorView 3D", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("Failed to create GLFW window");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // input callback functions
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    ImGuiInit(window, glsl_version);
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("Failed to initialize GLEW");

    return window;
}


