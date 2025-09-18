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
extern tira::volume<float> Lambda;								// eigenvalues of the tensor field
extern tira::volume<glm::mat3> Tn;

void RefreshVisualization() {
    OrthoViewer->slice_positions(UI.slice_positions);
}


glm::vec3 ColormapEigenvector(unsigned vi, float l0, float l1, float l2, float theta, float phi, float l2_max) {

    // The color of the eigenvector may be updated by the eigenvalue magnitudes
    l0 = std::abs(l0);
    l1 = std::abs(l1);
    l2 = std::abs(l2);

    // Convert the spherical coordinates for the eigenvector to Cartesian coordinates for visualization
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);
    float cos_phi = std::cos(phi);
    float sin_phi = std::sin(phi);
    float x = std::abs(cos_theta * sin_phi);
    float y = std::abs(sin_theta * sin_phi);
    float z = std::abs(cos_phi);

    // The color map incorporates how "certain" we are about the eigenvector direction. This
    // certainty is based on the size of the corresponding eigenvalue to the other eigenvalues.
    // If the other eigenvalues have a similar size to this eigenvalue, the orientation tensor is
    // more isotropic and the orientation of this eigenvector is less "certain".

    // alpha reflects the "certainty" of the current eigenvector orientation
    float alpha = 1.0f;

    // When visualizing the eigenvector associated with the largest eigenvalue, the certainty
    // will be based on how "stick-like" the tensor is. The more stick-like it is, the more certain
    // we are about the orientation of eigenvector 2.
    if (vi == 2) {
        alpha = (l2 - l1) / (l0 + l1 + l2);
    }
    // For the middle eigenvector, there are two cases where the uncertainty is high:
    //  1) A plate-like tensor where l1 == l2
    //  2) A stick-like tensor where l1 == l0
    // We calculate the certainty with respect to each option and then take the smallest value
    // to reflect the least amount of certainty.
    if (vi == 1) {
        float a1 = (l2 - l1) / (l0 + l1 + l2);
        float a2 = (l1 - l0) / (l0 + l1);
        alpha = std::min(a1, a2);
    }
    // For the smallest eigenvector, the uncertainty is only high when it is close to the middle
    // eigenvalue, and this can happen for both plate-like and ball-like tensors.
    if (vi == 0) {
        alpha = (l1 - l0) / (l0 + l1);
    }
    // In the unique case where all eigenvalues are zero, we have absolutely no certainty
    if  (l2 == 0)
        alpha = 0.0f;

    float r = alpha * x + (1.0f - alpha);
    float g = alpha * y + (1.0f - alpha);
    float b = alpha * z + (1.0f - alpha);

    r *= l2 / l2_max;
    g *= l2 / l2_max;
    b *= l2 / l2_max;

    return {r, g, b};

}

/**
 * @brief This function calculates the color map for a specified eigenvector and stores the result in the OrthoView
 * class for visualization.
 * @param vi is the eigenvector to be visualized
 */
void ColormapEigenvectors(unsigned vi) {
    float l2_max = Lambda.maxv();

    // iterate through each voxel in the data set
    for (unsigned zi=0; zi<OrthoViewer->Z(); zi++) {
        for (unsigned yi=0; yi<OrthoViewer->Y(); yi++) {
            for (unsigned xi=0; xi<OrthoViewer->X(); xi++) {

                glm::vec3 color = ColormapEigenvector(
                    vi,
                    Lambda(xi, yi, zi, 0),
                    Lambda(xi, yi, zi, 1),
                    Lambda(xi, yi, zi, 2),
                    ThetaPhi(xi, yi, zi, vi).x,
                    ThetaPhi(xi, yi, zi, vi).y,
                    l2_max
                    );

                (*OrthoViewer)(xi, yi, zi, 0) = (unsigned char)(color.r * 255.0); // set the color
                (*OrthoViewer)(xi, yi, zi, 1) = (unsigned char)(color.g * 255.0);
                (*OrthoViewer)(xi, yi, zi, 2) = (unsigned char)(color.b * 255.0);
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
        tira::cmap::colormap(Scalar.Data(), OrthoViewer->Data(), Scalar.Size(), ColorMap::Brewer);
        break;
    case ScalarType::EVec0:
        ColormapEigenvectors(0);
        break;
    case ScalarType::EVec1:
        ColormapEigenvectors(1);
        break;
    case ScalarType::EVec2:
        ColormapEigenvectors(2);
        break;


    }
    OrthoViewer->update_texture();
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
    GLFWwindow* window = glfwCreateWindow(width, height, "TensorView 3D", nullptr, nullptr);
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


