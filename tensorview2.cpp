#include <iostream>
#include <chrono>

#include "tira/graphics_gl.h"
#include "tira/graphics/glShader.h"
#include "tira/graphics/shapes/circle.h"
//#include "tira/image/tensorfield.h"
#include "tira/image/colormap.h"
#include "tira/image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <complex>

GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);   // specify the OpenGL color used to clear the back buffer
float ui_scale = 1.5f;                                  // scale value for the UI and UI text

// TENSOR FIELD DATA

tira::image<glm::mat2> T0;
tira::image<glm::mat2> Tn;
bool FIELD_LOADED = false;

float MINVAL, MAXVAL;
float MAXNORM;                          // maximum matrix norm in the field (size of the largest tensor)
int DIMENSION;
float SIGMA = 1;                        // blur kernel size
float SCALE = 0.3;
bool BLUR = false;

tira::glGeometry CMAP_GEOMETRY;
tira::glMaterial* CMAP_MATERIAL;

tira::glGeometry GLYPH_GEOMETRY;
tira::glMaterial* GLYPH_MATERIAL;
int GLYPH_ROWS = 100;
float GLYPH_SCALE = 0.3;
bool SCALE_BY_NORM = false;



enum ScalarType {NoScalar, Tensor00, Tensor01, Tensor02, Tensor11, Tensor12, Tensor22, EVal0, EVal1, EVal2, EVec0x, EVec0y, EVec1x, EVec1y, Eccentricity};
int SCALARTYPE = ScalarType::NoScalar;
bool RENDER_GLYPHS = false;

void FitRectangleToWindow(float rect_width, float rect_height, float window_width, float window_height, float& viewport_width, float& viewport_height) {
 
    float display_aspect = window_width / window_height;
    float image_aspect = rect_width / rect_height;
    if (image_aspect > display_aspect) {
        viewport_width = rect_width;
        viewport_height = rect_width / display_aspect;
    }
    else {
        viewport_height = rect_height;
        viewport_width = rect_height * display_aspect;
    }
}

void LoadTensorField(std::string filename) {
    T0.load_npy<float>(filename);
    Tn = T0;
    FIELD_LOADED = true;
}

/// <summary>
/// Function used to access individual elements of a 2D tensor
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="u"></param>
/// <param name="v"></param>
/// <returns></returns>
inline float Timg(unsigned int x, unsigned int y, unsigned int u, unsigned int v) {
    return Tn(x, y)[u][v];
}

inline float Trace2D(unsigned int x, unsigned int y) {
    float trace = Timg(x, y, 0, 0) + Timg(x, y, 1, 1);
    return trace;
}

inline float Determinant2D(unsigned int x, unsigned int y) {
    float det = Timg(x, y, 0, 0) * Timg(x, y, 1, 1) - pow(Timg(x, y, 0, 1), 2.0);
    return det;
}

inline float Eigenvalue2D(unsigned int x, unsigned int y, unsigned int i) {
    float d = Timg(x, y, 0, 0);
    float e = Timg(x, y, 0, 1);
    float f = e;
    float g = Timg(x, y, 1, 1);

    float dpg = d + g;
    float disc = sqrt((4 * e * f) + pow(d - g, 2));
    float a = (dpg + disc) / 2.0f;
    float b = (dpg - disc) / 2.0f;
    if (i == 0) return std::max(a, b);
    else return std::min(a, b);
}

inline glm::vec2 Eigenvector2D(unsigned int x, unsigned int y, unsigned int i) {

    float lambda = Eigenvalue2D(x, y, i);

    float d = Timg(x, y, 0, 0);                 // dx * dx
    float e = Timg(x, y, 0, 1);                 // dx * dy
    float f = e;
    float g = Timg(x, y, 1, 1);                 // dy * dy
   
    if(e != 0)
        return glm::normalize(glm::vec2(1.0f, (lambda - d) / e));
    else if (g == 0)
        return glm::vec2(1.0f, 0.0f);
    else
        return glm::vec2(0.0f, 1.0f);
}

inline float normaldist(float x, float sigma) {
    float scale = 1.0f / (sigma * sqrt(2 * 3.14159));
    float ex = -(x * x) / (2 * sigma * sigma);
    return scale * exp(ex);
}

/// <summary>
/// Blurs the tensor field and re-calculates the current scalar image
/// </summary>
/// <param name="sigma"></param>
void GaussianFilter(float sigma) {
    unsigned int size = ceil(sigma * 6);
    float dx = 1.0f;
    float start = -(float)(size - 1) / 2.0f;

    tira::image<float> K({ size, size });
    for (size_t vi = 0; vi < size; vi++) {
        float gv = normaldist(start + dx * vi, sigma);
        for (size_t ui = 0; ui < size; ui++) {
            float gu = normaldist(start + dx * ui, sigma);
            K(ui, vi) = gv * gu;
        }
    }

    auto t_start = std::chrono::steady_clock::now();
    
    //Tn = T0.convolve(K);
    Tn = T0.convolve2(K);
    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = t_end - t_start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

/// <summary>
/// Create a scalar image from the specified tensor component
/// </summary>
/// <param name="u"></param>
/// <param name="v"></param>
void ScalarFrom_TensorElement2D(unsigned int u, unsigned int v) {

    tira::image<float> img(Tn.shape()[1], Tn.shape()[0], 1);
    float val;
    for (int yi = 0; yi < Tn.shape()[0]; yi++) {
        for (int xi = 0; xi < Tn.shape()[1]; xi++) {
            val = Timg(xi, yi, u, v);
            img(xi, yi, 0) = val;
        }
    }
    MAXVAL = img.maxv();
    MINVAL = img.minv();
    CMAP_MATERIAL->SetTexture("scalar", img, GL_LUMINANCE32F_ARB, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Eval(unsigned int i) {

    tira::image<float> img(Tn.shape()[1], Tn.shape()[0], 1);      // allocate a scalar image
    //float* npy_data = NPY.data<float>();                        // get the raw data from the tensor field

    float t, d;
    for (int yi = 0; yi < Tn.shape()[0]; yi++) {                                     // for each tensor in the field
        for (int xi = 0; xi < Tn.shape()[1]; xi++) {
            img(xi, yi) = Eigenvalue2D(xi, yi, i);
        }
    }
    // update texture
    MAXVAL = img.maxv();
    MINVAL = img.minv();
    CMAP_MATERIAL->SetTexture("scalar", img, GL_LUMINANCE32F_ARB, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Eccentricity() {

    tira::image<float> img(Tn.shape()[1], Tn.shape()[0], 1);      // allocate a scalar image
    //float* npy_data = NPY.data<float>();                        // get the raw data from the tensor field

    float t, d;
    for (int yi = 0; yi < Tn.shape()[0]; yi++) {                                     // for each tensor in the field
        for (int xi = 0; xi < Tn.shape()[1]; xi++) {
            float l0 = Eigenvalue2D(xi, yi, 0);
            float l1 = Eigenvalue2D(xi, yi, 1);
            if (l0 == 0.0f)
                img(xi, yi) = 0.0f;
            else
                img(xi, yi) = sqrt(1.0f - (l1 * l1) / (l0 * l0));
            //img(xi, yi) = sqrt((l0 * l0) - (l1 * l1));
        }
    }
    // update texture
    MAXVAL = img.maxv();
    MINVAL = img.minv();
    CMAP_MATERIAL->SetTexture("scalar", img, GL_LUMINANCE32F_ARB, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvector and component
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Evec(unsigned int i, unsigned int component) {

    tira::image<float> img(Tn.width(), Tn.height(), 1);      // allocate a scalar image

    float t, d;
    for (int yi = 0; yi < Tn.height(); yi++) {                                     // for each tensor in the field
        for (int xi = 0; xi < Tn.width(); xi++) {
            glm::vec2 evec = Eigenvector2D(xi, yi, i);
            img(xi, yi) = evec[component];
        }
    }
    // update texture
    MAXVAL = img.maxv();
    MINVAL = img.minv();
    CMAP_MATERIAL->SetTexture("scalar", img, GL_LUMINANCE32F_ARB, GL_NEAREST);
}



void ScalarRefresh() {

    // calculate the max norm
    ScalarFrom_Eval(0);
    MAXNORM = MAXVAL;

    switch (SCALARTYPE) {
    case ScalarType::EVal0:
        ScalarFrom_Eval(0);
        break;
    case ScalarType::EVal1:
        ScalarFrom_Eval(1);
        break;
    case ScalarType::EVal2:
        ScalarFrom_Eval(2);
        break;
    case ScalarType::EVec0x:
        ScalarFrom_Evec(0, 0);
        break;
    case ScalarType::EVec0y:
        ScalarFrom_Evec(0, 1);
        break;
    case ScalarType::EVec1x:
        ScalarFrom_Evec(1, 0);
        break;
    case ScalarType::EVec1y:
        ScalarFrom_Evec(1, 1);
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
    }
}

void RenderFieldSpecs() {
    ImGui::Text("Tensor Field Specifications");
    if (Tn.size() == 0) {
        ImGui::Text("None Loaded");
        return;
    }

    std::stringstream ss;
    ss << Tn.shape()[0] << " x " << Tn.shape()[1];

    ImGui::Text(("Field Size: " + ss.str()).c_str());
    ImGui::Text("Maximum Norm: %f", MAXNORM);
}
/// This function renders the user interface every frame
void RenderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

    if (ImGui::Button("Load File"))					// create a button for loading the shader
        ImGuiFileDialog::Instance()->OpenDialog("ChooseNpyFile", "Choose NPY File", ".npy,.npz", ".");
    //ImGui::SameLine();
    if (ImGuiFileDialog::Instance()->Display("ChooseNpyFile")) {				    // if the user opened a file dialog
        if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
            std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file
            std::string extension = filename.substr(filename.find_last_of(".") + 1);
            if (extension == "npy") {
                LoadTensorField(filename);
                SCALARTYPE = ScalarType::EVal0;
                ScalarRefresh();
                RENDER_GLYPHS = true;
            }
        }
        ImGuiFileDialog::Instance()->Close();									// close the file dialog box		
    }

    RenderFieldSpecs();

    // select scalar component
    ImGui::RadioButton("None", &SCALARTYPE, (int)ScalarType::NoScalar);
    if (ImGui::RadioButton("[0, 0] = dxdx", &SCALARTYPE, (int)ScalarType::Tensor00)) {
        ScalarFrom_TensorElement2D(0, 0);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("[1, 1] = dydy", &SCALARTYPE, (int)ScalarType::Tensor11)) {
        ScalarFrom_TensorElement2D(1, 1);
    }
    if (ImGui::RadioButton("[0, 1] = dxdy", &SCALARTYPE, (int)ScalarType::Tensor01)) {
        ScalarFrom_TensorElement2D(0, 1);
    }
    if (ImGui::RadioButton("lambda 0", &SCALARTYPE, (int)ScalarType::EVal0)) {
        ScalarFrom_Eval(0);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("lambda 1", &SCALARTYPE, (int)ScalarType::EVal1)) {
        ScalarFrom_Eval(1);
    }
    if (ImGui::RadioButton("evec 0 (x)", &SCALARTYPE, (int)ScalarType::EVec0x)) {
        ScalarFrom_Evec(0, 0);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("evec 0 (y)", &SCALARTYPE, (int)ScalarType::EVec0y)) {
        ScalarFrom_Evec(0, 1);
    }
    if (ImGui::RadioButton("evec 1 (x)", &SCALARTYPE, (int)ScalarType::EVec1x)) {
        ScalarFrom_Evec(1, 0);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("evec 1 (y)", &SCALARTYPE, (int)ScalarType::EVec1y)) {
        ScalarFrom_Evec(1, 1);
    }
    if (ImGui::RadioButton("eccentricity", &SCALARTYPE, (int)ScalarType::Eccentricity)) {
        ScalarFrom_Eccentricity();
    }
    std::stringstream ss;
    ss << "Min: " << MINVAL << "\t Max: " << MAXVAL;
    ImGui::Text(ss.str().c_str());

    if (ImGui::Checkbox("Blur", &BLUR)) {
        if (BLUR) {
            GaussianFilter(SIGMA);
        }
        else {
            Tn = T0;
        }
        ScalarRefresh();
    }
    ImGui::SameLine();
    if (ImGui::InputFloat("##Sigma", &SIGMA, 0.2f, 1.0f)) {
        if (SIGMA <= 0) SIGMA = 0.01;
        if (BLUR) {
            GaussianFilter(SIGMA);
            ScalarRefresh();
        }
    }

    ImGui::Checkbox("Glyphs", &RENDER_GLYPHS);
    ImGui::SameLine();
    ImGui::InputInt("##Rows", &GLYPH_ROWS, 1, 10);
    ImGui::InputFloat("Scale", &SCALE, 0.01, 0.1);
    ImGui::Checkbox("Scale by Norm", &SCALE_BY_NORM);



    ImGui::Render();                                                            // Render all windows
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

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}



int main(int argc, char** argv) {

    //tira::field<double> test;
    //test.load_npy("D:\\Dropbox\\source\\baig-tensor\\test.npy");

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return -1;

    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    window = glfwCreateWindow(1600, 1200, "ImGui GLFW+OpenGL3 Hello World Program", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    InitUI(window, glsl_version);

    if (glewInit() != GLEW_OK)
        std::cout << "Error!" << std::endl;
	std::cout << "https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html" << std::endl;

    // Load the tensor field if it is provided as a command-line argument
    if (argc == 2) {
        LoadTensorField(argv[1]);
    }

    CMAP_GEOMETRY = tira::glGeometry::GenerateRectangle<float>();
    CMAP_MATERIAL = new tira::glMaterial("colormap.shader");
    SCALARTYPE = ScalarType::NoScalar;

    GLYPH_GEOMETRY = tira::glGeometry::GenerateCircle<float>(100);
    GLYPH_MATERIAL = new tira::glMaterial("glyph2d.shader");

    //ScalarFrom_TensorElement2D(1, 1);
    
    //colormap_material.SetTexture("scalar", SCALAR, GL_LUMINANCE, GL_NEAREST);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        RenderUI();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);                     // specifies the area of the window where OpenGL can render
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

        
        

        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        if (FIELD_LOADED) {

            float viewport_width, viewport_height;
            FitRectangleToWindow(Tn.width(), Tn.height(), display_w, display_h, viewport_width, viewport_height);
            glm::mat4 Mview = glm::ortho(-viewport_width / 2.0f, viewport_width / 2.0f, -viewport_height / 2.0f, viewport_height / 2.0f);   // create a view matrix

            if (SCALARTYPE != ScalarType::NoScalar) {

                glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), glm::vec3((float)Tn.width(), (float)Tn.height(), 1.0f));     // compose the scale matrix from the width and height of the tensor field
                glm::mat4 Mtrans = glm::mat4(1.0f);                                                                         // there is no translation (the 2D field is centered at the origin)
                glm::mat4 M = Mview * Mtrans * Mscale;                                                                      // create the transformation matrix
                CMAP_MATERIAL->Begin();                                                                                     // begin using the scalar colormap material
                CMAP_MATERIAL->SetUniform1f("maxval", MAXVAL);                                                              // pass the max and min values so that the color map can be scaled
                CMAP_MATERIAL->SetUniform1f("minval", MINVAL);
                CMAP_MATERIAL->SetUniformMat4f("Mview", M);                                                                 // pass the transformation matrix as a uniform
                CMAP_GEOMETRY.Draw();                                                                                       // draw the rectangle
                CMAP_MATERIAL->End();                                                                                       // stop using the material
            }
            if (RENDER_GLYPHS) {
                tira::image<float> Ti((float*)Tn.data(), Tn.X(), Tn.Y(), 4);                                                // create an image to store the tensor field data (2D tensor has 4 channels)
                GLYPH_MATERIAL->SetTexture<float>("tensorfield", Ti, GL_RGBA32F, GL_NEAREST);                               // set the texture to the tensor field image (use GL_RGBA for a 4-channel float)
                GLYPH_MATERIAL->Begin();                                                                                    // bind the material
                int glyph_cols = GLYPH_ROWS * (float)Ti.width() / (float)Ti.height();                                       // calculate the number of glyph columns based on glyph rows (so the glyphs are isotropic)
                float scale = (float)Ti.height() / (float)GLYPH_ROWS;                                                       // calculate the scale factor for the glyphs (so that they don't overlap)
                glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), glm::vec3(scale, scale, 1.0f));                              // create a scale matrix based on the calculated scale value

                float glyph_start_x = -(float)Ti.width() / 2.0f + scale / 2.0f;                                             // start from -x and -y
                float glyph_start_y = -(float)Ti.height() / 2.0f + scale / 2.0f;
                for (int yi = 0; yi < GLYPH_ROWS; yi++) {                                                                   // for each row
                    for (int xi = 0; xi < glyph_cols; xi++) {                                                               // for each column
                                                                                                                            // create a translation matrix moving the glyph to its appropriate position
                        glm::mat4 Mtrans = glm::translate(glm::mat4(1.0f), 
                                                          glm::vec3(glyph_start_x + xi * scale, 
                                                          glyph_start_y + yi * scale, 
                                                          1.0f));
                        glm::mat4 M = Mview * Mtrans * Mscale;                                                              // assemble the transformation matrix

                        GLYPH_MATERIAL->SetUniformMat4f("MVP", M);                                                          // pass the transformation matrix to the material
                        GLYPH_MATERIAL->SetUniform1f("tx", (float)xi/(float)glyph_cols + scale / (float)Ti.height());                                    // pass the glyph coordinate to the material
                        GLYPH_MATERIAL->SetUniform1f("ty", (float)yi/(float)GLYPH_ROWS + scale / (float)Ti.width());
                        if (SCALE_BY_NORM)
                            GLYPH_MATERIAL->SetUniform1f("maxnorm", MAXNORM);                                               // set a flag to determine of the glyphs are normalized (largest glyph takes up a single zone)
                        else
                            GLYPH_MATERIAL->SetUniform1f("maxnorm", 0.0f);
                        GLYPH_MATERIAL->SetUniform1f("scale", SCALE);                                                       // pass the glyph scale factor (input by the user)
                        GLYPH_GEOMETRY.Draw();                                                                              // draw the glyph
                    }
                }
                GLYPH_MATERIAL->End();                                                                                      // unbind the glyph material
                
            }
        }

        /****************************************************/
        /*      Draw Stuff To The Viewport                  */
        /****************************************************/

        /*
            projection matrix: world to screen coordinates
            view matrix: transforms from world space to camera
            model matrix: converts coordinates in space of a geometric object to world space
        */

        // FOR each pixel in the loaded image
            // calculate the position of the tensor (in the example you can use the image index)

        //for (int i = 0; i < T_npy.shape[0]; i++) {
        //    for (int j = 0; j < T_npy.shape[1]; j++) {
        //        int index = i * T_npy.shape[1] + j;

        //        glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), glm::vec3(gui_tensor_scale, gui_tensor_scale, gui_tensor_scale));    // generate a scale matrix
        //        glm::vec3 tensor_position((float)i, (float)j, 0.0f);
        //        glm::mat4 Mtrans = glm::translate(glm::mat4(1.0f), tensor_position);
        //        glm::mat4 Mrot = glm::rotate(glm::mat4(1.0f), T_theta[index], glm::vec3(0, 0, 1));
        //        glm::mat4 M = Mtrans * Mrot * Mscale;

        //        float l0 = T_lambda0[index];
        //        float l1 = T_lambda1[index];

        //        shader->Bind();
        //        shader->SetUniformMat4f("MVP", P * M);
        //        shader->SetUniform1f("l0", 1.0f);
        //        shader->SetUniform1f("l1", l1 / l0);
        //        circle[index].Draw();
        //    }
        //}

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer

        glfwPollEvents();
    }

    DestroyUI();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW

    return 0;
}