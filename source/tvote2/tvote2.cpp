#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

#include "tvote2.h"
#include "tira/image.h"

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <sstream>
#include <string>

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



// Functions for updating individual components of the processed tensor field
void UpdateEigenDecomposition() {
    EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
}

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

    // set up CUDA devices
    UI.device_names.emplace_back("CPU");                            // push the CPU as the first device option
    InitializeCuda(UI.cuda_device, UI.device_names);
    UI.num_devices = UI.device_names.size();



    if (UI.num_devices <= in_device) UI.cuda_device = -1;            // if the specified device is out of range, set it to the CPU
    else UI.cuda_device = in_device;                                 // otherwise use the specified CUDA device


    // Load the tensor field if it is provided as a command-line argument
    if (vm.count("input")) {
        LoadTensorField(in_inputname, &T0);
        Tn = T0;
        UI.loaded_filename = in_inputname;
        UI.field_loaded = true;
    }
    else {
    }


    // create a GLSL window, initialize the OpenGL context, and assign callback functions
    window = InitWindow(1600, 1200);    


    InitShaders();
    InitCmapGeometry();

    if (UI.field_loaded) {
        Tn = T0;
        EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
        RefreshScalarField();
        RefreshVisualization();
    }
    


    // Main loop
    while (!glfwWindowShouldClose(window)) {

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);                     // specifies the area of the window where OpenGL can render
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        ImGuiRender();

        // if a tensor field is loaded
        if (UI.field_loaded) {
            RenderFieldOpenGL(display_w, display_h);

        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer

    }

    ImGuiDestroy();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW

    return 0;
}