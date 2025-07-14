#include <boost/program_options.hpp>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <cuda_runtime.h>

#include "tvote3.h"

#include <tira/graphics/glOrthoView.h>
#include <tira/eigen.h>

TV3_UI UI;

tira::volume<glm::mat3> T0;								// 3D tensor field (3x3 voxels)
tira::volume<glm::mat3> Tn;								// processed tensor field
tira::volume<float> Lambda;								// eigenvalues of the tensor field
tira::volume<glm::vec2> ThetaPhi;						// eigenvectors of the tensor field (in spherical coordinates)
tira::volume<float> Scalar;								// scalar field that is currently being visualized

tira::glOrthoView<unsigned char>* OrthoViewer;

GLFWwindow* window;
const char* glsl_version = "#version 130";

float dot(float v0[3], float v1[3]) {
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

int main(int argc, char** argv) {

    glm::mat3 M(0.41863424, 0.27482766, -0.1500264, 0.27482766, 0.87008137, 0.07092162, -0.1500264, 0.07092162, 0.9612844);

    float evals[3];
    tira::eval3_symmetric(M[0][0], M[1][0], M[1][1], M[2][0], M[2][1], M[2][2], evals[0], evals[1], evals[2]);

    float evec0[3];
    float evec1[3];
    float evec2[3];
    tira::evec3_symmetric(M[0][0], M[1][0], M[1][1], M[2][0], M[2][1], M[2][2], evals, evec0, evec1, evec2);

    float dot_01 = dot(evec0, evec1);
    float dot_12 = dot(evec1, evec2);
    float dot_02 = dot(evec0, evec2);

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

    OrthoViewer = new tira::glOrthoView<unsigned char>();
    OrthoViewer->init();

    OrthoViewer->generate_rgb(32, 24, 10, 2);
    UI.field_loaded = true;

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        OrthoViewer->aspect((float)display_w / (float)display_h);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        ImGuiRender();

        // if a tensor field is loaded
        if (UI.field_loaded) {
            // render the XY plane
            glViewport(display_w / 2, 0, display_w/2, display_h/2);
            OrthoViewer->render_slice(2);

            // render the XZ plane
            glViewport(display_w / 2, display_h / 2, display_w / 2, display_h / 2);
            OrthoViewer->render_slice(1);

            // render the YZ plane
            glViewport(0, 0, display_w / 2, display_h / 2);
            OrthoViewer->render_slice(0);
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer

    }

    ImGuiDestroy();                                                 // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW
}