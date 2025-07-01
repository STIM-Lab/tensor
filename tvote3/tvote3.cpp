#include <boost/program_options.hpp>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <cuda_runtime.h>

#include "tvote3.h"

TV3_UI UI;

tira::volume<glm::mat3> T0;								// 3D tensor field (3x3 voxels)
tira::volume<glm::mat3> Tn;								// processed tensor field
tira::volume<float> Lambda;								// eigenvalues of the tensor field
tira::volume<glm::vec2> ThetaPhi;						// eigenvectors of the tensor field (in spherical coordinates)
tira::volume<float> Scalar;								// scalar field that is currently being visualized

GLFWwindow* window;
const char* glsl_version = "#version 130";

int main(int argc, char** argv) {

    std::string in_filename;
    int in_device;
    std::vector<float> in_voxel_size;

    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("input", boost::program_options::value<std::string>(&in_filename), "input filename for the tensor field (*.npy)")
        //("volume", boost::program_options::value<std::string>(&in_image), "optional image field corresponding to the tensors")
        //("gamma, g", boost::program_options::value<float>(&in_gamma)->default_value(3), "glyph gamma (sharpness), 0 = spheroids")
        //("cmap,c", boost::program_options::value<int>(&in_cmap)->default_value(0), "colormaped eigenvector (0 = longest, 2 = shortest)")
        ("cuda", boost::program_options::value<int>(&in_device)->default_value(0), "CUDA device ID (-1 for CPU only)")
        ("voxel", boost::program_options::value<std::vector<float>>(&in_voxel_size)->multitoken()->default_value(std::vector<float>{1.0f, 1.0f, 1.0f}, "1.0 1.0 1.0"), "voxel size")
        ("help", "produce help message");
    boost::program_options::variables_map vm;

    // make sure there the specified CUDA device is available (otherwise switch to CPU)
    if (in_device < 0) {
        UI.cuda_device = -1;
    }
    else {
        int ndevices;
        cudaError_t error = cudaGetDeviceCount(&ndevices);
        if (error != cudaSuccess) ndevices = 0;
        if (ndevices <= in_device) {
            std::cout << "WARNING: Specified CUDA device " << in_device << " is unavailable (" << ndevices << " compatible devices found), defaulting to CPU" << std::endl;
            UI.cuda_device = -1;
        }
        else
            UI.cuda_device = in_device;
    }

    boost::program_options::positional_options_description p;
    p.add("input", -1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    // create a GLSL window, initialize the OpenGL context, and assign callback functions
    window = InitWindow(1600, 1200);

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
            //RenderFieldOpenGL(display_w, display_h);
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer

    }

    ImGuiDestroy();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW
}