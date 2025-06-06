#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include "tv2.h"




#include "tira/image.h"







#include <sstream>
#include <string>
#include <complex>

// Function signatures for processing the tensor field
void EigenDecomposition(tira::image<glm::mat2>* tensor, tira::image<float>* lambda, tira::image<float>* theta, int cuda_device = 0);

glm::mat2* cudaGaussianBlur(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
    unsigned int& out_width, unsigned int& out_height, int deviceID = 0);

void cudaVote2D(float* input_field, float* output_field,
    unsigned int s0, unsigned int s1,
    float sigma, float sigma2,
    unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug);



void RefreshVisualization();

TV2_UI UI;                            // structure stores the GUI information for tensorview2

GLFWwindow* window;
const char* glsl_version = "#version 130";

// command line arguments
std::string in_inputname;
std::string in_l0_outputname;
std::string in_l1_outputname;
std::string in_v0_outputname;
std::string in_v1_outputname;
float in_blur_strength;
int in_device;                              // CUDA device ID



// TENSOR FIELD DATA
tira::image<glm::mat2> T0;              // initial tensor field passed to the visualization program
tira::image<glm::mat2> Tn;              // current tensor field being rendered (after processing)

// TENSOR FIELD DERIVATIVES
tira::image<float> Lambda;              // eigenvalues of the current tensor field
tira::image<float> Theta;               // eigenvectors of the current tensor field (in polar coordinates)
tira::image<float> Scalar;              // scalar component for the current tensor field

tira::image<unsigned char> CurrentColormap;

//float MINVAL, MAXVAL;
//float MAXNORM;                          // maximum matrix norm in the field (size of the largest tensor)
//int DIMENSION;

//float SCALE = 0.3;


// Functions for updating individual components of the processed tensor field
void UpdateEigenDecomposition() {
    EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
}











/*tira::image<unsigned char> ColormapTensor(const unsigned int row, const unsigned int col) {
    SCALAR = tira::image<float>(Tn.shape()[1], Tn.shape()[0], 1);

    for (unsigned int yi = 0; yi < Tn.shape()[0]; yi++) {
        for (unsigned int xi = 0; xi < Tn.shape()[1]; xi++) {
            const float val = Timg(xi, yi, row, col);
            SCALAR(xi, yi, 0) = val;
        }
    }
    float maxmag = std::max(std::abs(SCALAR.maxv()), std::abs(SCALAR.minv()));  // calculate the highest magnitude scalar value
    MAXVAL = maxmag;
    MINVAL = -maxmag;
    tira::image<unsigned char> C = SCALAR.cmap(MINVAL, MAXVAL, ColorMap::Brewer);
    return C;
}

tira::image<unsigned char> ColormapEval(const unsigned int i) {
    tira::image<float> L = Lambda.channel(i);
    MAXVAL = Lambda.maxv();
    MINVAL = Lambda.minv();
    float MaxMag = std::max(std::abs(MAXVAL), std::abs(MINVAL));
    tira::image<unsigned char> C = Lambda.channel(i).cmap(-MaxMag, MaxMag, ColorMap::Brewer);
    return C;
}



tira::image<unsigned char> ColormapEccentricity() {
    tira::image<float> ecc = CalculateEccentricity();
    MAXVAL = 1.0;
    MINVAL = 0.0;
    CurrentColormap = ecc.cmap(0, 1, ColorMap::Magma);
    return CurrentColormap;
}
*/






/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/*void ScalarFrom_Eccentricity() {
    CurrentColormap = ColormapEccentricity();
    CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvector and component
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Evec(const unsigned int i) {
    Scalar = Theta.channel(i);
    CurrentColormap = ColormapEvec(i);
    CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
}
*/


/*void ScalarRefresh() {
    if (!FIELD_LOADED) return;              // return if there's no field loaded

    // calculate the max norm
    ScalarFrom_Eval(0);
    MAXNORM = MAXVAL;

    switch (UI.scalar_type) {
    case ScalarType::EVal0:
        ScalarFrom_Eval(0);
        break;
    case ScalarType::EVal1:
        ScalarFrom_Eval(1);
        break;
    case ScalarType::EVec0:
        ScalarFrom_Evec(0);
        break;
    case ScalarType::EVec1:
        ScalarFrom_Evec(1);
        break;
    case ScalarType::Tensor00:
        ScalarFrom_TensorElement2D(0, 0);
        break;
    case ScalarType::Tensor11:
        ScalarFrom_TensorElement2D(1, 1);
        break;
    case ScalarType::Tensor01:
        ScalarFrom_TensorElement2D(0, 1);
        break;
    case ScalarType::Eccentricity:
        ScalarFrom_Eccentricity();
        break;
    default:
        throw std::runtime_error("Invalid scalar type");
    }
    MAXVAL = Scalar.maxv();
    MINVAL = Scalar.minv();
}*/
















int main(int argc, char** argv) {
    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("input", boost::program_options::value<std::string>(&in_inputname), "output filename for the coupled wave structure")
        ("nogui", "do not provide a user interface (only files are saved)")
        ("l0", boost::program_options::value<std::string>(&in_l0_outputname), "color map image file for the smallest eigenvalue")
        ("l1", boost::program_options::value<std::string>(&in_l1_outputname), "color map image file for the largest eigenvalue")
        ("v0", boost::program_options::value<std::string>(&in_v0_outputname), "color map image file for the smallest eigenvector")
        ("v1", boost::program_options::value<std::string>(&in_v1_outputname), "color map image file for the largest eigenvector")
        ("blur", boost::program_options::value<float>(&in_blur_strength), "sigma for gaussian blur")
        ("cuda", boost::program_options::value<int>(&in_device)->default_value(0), "CUDA device ID (-1 for CPU only)")
        ("help", "produce help message");
    boost::program_options::variables_map vm;


    boost::program_options::positional_options_description p;
    p.add("input", -1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

    boost::program_options::notify(vm);

    // if the user passes the help parameter, output command line details
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    // make sure there the specified CUDA device is available (otherwise switch to CPU)
    int ndevices;
    cudaError_t error = cudaGetDeviceCount(&ndevices);
    if (error != cudaSuccess) ndevices = 0;
    if (ndevices <= in_device) {
        std::cout << "WARNING: Specified CUDA device " << in_device << " is unavailable (" << ndevices << " compatible devices found), defaulting to CPU" << std::endl;
        in_device = -1;
    }

    // Load the tensor field if it is provided as a command-line argument
    if (vm.count("input")) {
        LoadTensorField(in_inputname, &T0);
        UI.loaded_filename = in_inputname.c_str();
        UI.glyph_rows = static_cast<int>(Tn.shape()[0]);
    }
    else {
    }

    /*if (FIELD_LOADED) {
        if (vm.count("blur")) {
            GaussianFilter(&T0, &Tn, in_blur_strength, in_device);
        }
        if (vm.count("l0")) ColormapEval(0).save(in_l0_outputname);
        if (vm.count("l1")) ColormapEval(1).save(in_l1_outputname);
        if (vm.count("v0")) ColormapEvec(0).save(in_v0_outputname);
        if (vm.count("v1")) ColormapEvec(1).save(in_v1_outputname);

        if (vm.count("nogui")) {
            return 0;
        }
    }*/

    // create a GLSL window, initialize the OpenGL context, and assign callback functions
    window = InitWindow(1600, 1200);

    // initialize everything that will be rendered to the OpenGL context
    InitActors();




    if (UI.field_loaded) {
        Tn = T0;
        EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
        UI.camera_position = glm::vec3(static_cast<float>(Tn.width()) / 2.0f, static_cast<float>(Tn.height()) / 2.0f, 0.0f);
    }


    // Main loop
    while (!glfwWindowShouldClose(window)) {

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);                     // specifies the area of the window where OpenGL can render
        //glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        ImGuiRender();

        // if a tensor field is loaded
        if (UI.field_loaded) {
            RenderFieldOpenGL(display_w, display_h);

        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer

        //glfwPollEvents();
    }

    ImGuiDestroy();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW

    return 0;
}