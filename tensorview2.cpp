#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include "tira/graphics_gl.h"
#include "tira/graphics/glShader.h"
#include "tira/graphics/shapes/circle.h"
#include "tira/image/colormap.h"
#include "tira/image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

#include <fstream>
#include <sstream>
#include <string>
#include <complex>

glm::mat2* cudaGaussianBlur(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
                            unsigned int& out_width, unsigned int& out_height, int deviceID = 0);

float* cudaEigenvalues(float* tensors, unsigned int n, int device);
float* cudaEigenvectorsPolar(float* tensors, float* evals, unsigned int n, int device);

void cudaVote2D(float* input_field, float* output_field,
    unsigned int s0, unsigned int s1,
    float sigma, float sigma2,
    unsigned int w, unsigned int power, unsigned int device, bool STICK, bool PLATE, bool debug);

glm::vec2 Eigenvector2D(glm::mat2 T, float lambda);

glm::vec2 Eigenvalues2D(glm::mat2 T);

// command line arguments
std::string in_inputname;
std::string in_l0_outputname;
std::string in_l1_outputname;
std::string in_v0_outputname;
std::string in_v1_outputname;
float in_blur_strength;
int in_device;                              // CUDA device ID

GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);   // specify the OpenGL color used to clear the back buffer
float ui_scale = 1.5f;                                  // scale value for the UI and UI text

const std::string colormap_shader_string =
#include "shaders/colormap.shader"
;

const std::string glyph_shader_string =
#include "shaders/glyph2d.shader"
;

const std::string test_shader_string =
#include "shaders/test.shader"
;

// TENSOR FIELD DATA
tira::image<glm::mat2> T0;              // initial tensor field passed to the visualization program
tira::image<glm::mat2> Tn;              // current tensor field being rendered (after processing)

// TENSOR FIELD DERIVATIVES
tira::image<float> Ln;                  // eigenvalues of the current tensor field
tira::image<float> THETAn;          // eigenvectors of the current tensor field (in polar coordinates)

tira::image<float> SCALAR;
bool FIELD_LOADED = false;

tira::image<unsigned char> CurrentColormap;

float MINVAL, MAXVAL;
float MAXNORM;                          // maximum matrix norm in the field (size of the largest tensor)
int DIMENSION;
float SIGMA = 1;                        // blur kernel size
float TV_SIGMA1 = 3;
float TV_SIGMA2 = 0;
int TV_P = 1;
float SCALE = 0.3;
float CameraZoom = 1.0f;               // scale factor for the field (for zooming)

int EIGENVALUE_SIGN = 0;                // limits eigenvector visualization to signed eigenvalues (-1 = negative eigenvalues, 0 = all, 1 = positive eigenvalues)

tira::glGeometry CMAP_GEOMETRY;
tira::glMaterial* CMAP_MATERIAL;

tira::glGeometry GLYPH_GEOMETRY;
tira::glMaterial* GLYPH_MATERIAL;

tira::glMaterial* testmaterial;

int GLYPH_ROWS = 100;
float GLYPH_SCALE = 0.8;
bool GLYPH_NORMALIZE = false;
int GLYPH_TESSELATION = 20;

glm::vec2 CameraPos;
glm::vec2 prevMousePos;                 // stores the last polled mouse position

float MousePos[2];
float Viewport[2];

const char* FileName = "";

enum ScalarType {NoScalar, Tensor00, Tensor01, Tensor11, EVal0, EVal1, EVec0, EVec1, Eccentricity};
int SCALARTYPE = ScalarType::EVec1;

enum ProcessingType {NoProcessing, Gaussian, Vote};
int PROCESSINGTYPE = ProcessingType::NoProcessing;

enum AdjustColorType {NoAdjustment, Darken, Lighten};
int ECCENTRICITYCOLORMODE = AdjustColorType::Lighten;
int MAGNITUDECOLORMODE = AdjustColorType::Darken;
float MAGNITUDECOLORTHRESHOLD = -1;
float L1MaxMag = -1;

bool TV_STICK = true;
bool TV_PLATE = true;


bool RENDER_GLYPHS = false;

// Calculate the viewport width and height in field pixels given the size of the field and window
void FitRectangleToWindow(float field_width_pixels, float field_height_pixels, 
                          float window_width, float window_height,// float scale,
                          float& viewport_width, float& viewport_height) {
 
    float display_aspect = window_width / window_height;
    float image_aspect = field_width_pixels / field_height_pixels;
    if (image_aspect > display_aspect) {
        viewport_width = field_width_pixels;// / scale;
        viewport_height = field_width_pixels / display_aspect;// / scale;
    }
    else {
        viewport_height = field_height_pixels;// / scale;
        viewport_width = field_height_pixels * display_aspect;// / scale;
    }
}

tira::image<float> CalculateEccentricity() {
    tira::image<float> ecc(Ln.X(), Ln.Y());

    for (size_t yi = 0; yi < ecc.Y(); yi++) {
        for (size_t xi = 0; xi < ecc.X(); xi++) {
            if (Ln(xi, yi, 1) == 0)
                ecc(xi, yi) = 0;
            else {
                const float l02 = std::pow(Ln(xi, yi, 0), 2.0f);
                const float l12 = std::pow(Ln(xi, yi, 1), 2.0f);
                ecc(xi, yi) = std::sqrt(1.0f - l02 / l12);
            }
        }
    }
    return ecc;
}

/// <summary>
/// Update the field eigenvectors and eigenvalues
/// </summary>
void UpdateEigens() {
    float* eigenvalues_raw = cudaEigenvalues((float*)Tn.data(), Tn.X() * Tn.Y(), in_device);
    Ln = tira::image<float>(eigenvalues_raw, Tn.X(), Tn.Y(), 2);
    float* eigenvectors_raw = cudaEigenvectorsPolar((float*)Tn.data(), eigenvalues_raw, Tn.X() * Tn.Y(), in_device);
    THETAn = tira::image<float>(eigenvectors_raw, Tn.X(), Tn.Y(), 2);

    free(eigenvalues_raw);
    free(eigenvectors_raw);
}

void LoadTensorField(const std::string& filename) {
    T0.load_npy<float>(filename);
    Tn = T0;
    UpdateEigens();
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
inline float Timg(const unsigned int x, const unsigned int y, const unsigned int u, const unsigned int v) {
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
    // if a CUDA device is enabled, use a blur kernel
    if(in_device >=0) {
        unsigned int blur_width;
        unsigned int blur_height;
        glm::mat2* blurred = cudaGaussianBlur(T0.data(), T0.X(), T0.Y(), sigma, blur_width, blur_height, in_device);

        Tn = tira::image<glm::mat2>(blurred, blur_width, blur_height);
        free(blurred);
    }
    // otherwise use the CPU
    else {
        unsigned int size = ceil(sigma * 6);
        float dx = 1.0f;
        float start = -(float)(size - 1) / 2.0f;
        std::vector<size_t> sx = { 1, size, 1 };
        std::vector<size_t> sy = { size, 1, 1 };
        tira::image<float> Kx(size, 1);
        tira::image<float> Ky(1, size);
        for (size_t i = 0; i < size; i++) {
            float v = normaldist(start + dx * i, sigma);
            Kx(i, 0, 0) = v;
            Ky(0, i, 0) = v;
        }
        Tn = T0.convolve2(Kx);
        Tn = Tn.convolve2(Ky);
    }

}

void TensorVote(float sigma, unsigned int p, float sigma2, bool stick, bool plate) {
    Tn = tira::image<glm::mat2>(T0.X(), T0.Y());

    unsigned int w = 6 * std::max(sigma, sigma2) + 1;
    cudaVote2D((float*)T0.data(), (float*)Tn.data(), 
        (unsigned int)T0.shape()[0], (unsigned int)T0.shape()[1], 
        sigma, sigma2, w, p, in_device, stick, plate, false);

    
}

tira::image<unsigned char> ColormapTensor(unsigned int row, unsigned int col) {
    SCALAR = tira::image<float>(Tn.shape()[1], Tn.shape()[0], 1);
    float val;
    for (int yi = 0; yi < Tn.shape()[0]; yi++) {
        for (int xi = 0; xi < Tn.shape()[1]; xi++) {
            val = Timg(xi, yi, row, col);
            SCALAR(xi, yi, 0) = val;
        }
    }
    float maxmag = std::max(std::abs(SCALAR.maxv()), std::abs(SCALAR.minv()));  // calculate the highest magnitude scalar value
    MAXVAL = maxmag;
    MINVAL = -maxmag;
    tira::image<unsigned char> C = SCALAR.cmap(MINVAL, MAXVAL, ColorMap::Brewer);
    return C;
}

tira::image<unsigned char> ColormapEval(unsigned int i) {
    tira::image<float> L = Ln.channel(i);
    MAXVAL = Ln.maxv();
    MINVAL = Ln.minv();
    float MaxMag = std::max(std::abs(MAXVAL), std::abs(MINVAL));
    tira::image<unsigned char> C = Ln.channel(i).cmap(-MaxMag, MaxMag, ColorMap::Brewer);
    return C;
}

tira::image<unsigned char> ColormapEvec(unsigned int i) {    

    // calculate a color representing the eigenvector angle
    tira::image<float> AngleColor = THETAn.channel(i).cmap(-std::numbers::pi, std::numbers::pi, ColorMap::RainbowCycle);


    // adjust the pixel color based on the eccentricity
    if (ECCENTRICITYCOLORMODE) {
        tira::image<float> EccentricityColor(AngleColor.X(), AngleColor.Y(), AngleColor.C());
        if (ECCENTRICITYCOLORMODE == AdjustColorType::Lighten)
            EccentricityColor = 255;
        else
            EccentricityColor = 0;

        tira::image<float> eccentricity = CalculateEccentricity();
        AngleColor = AngleColor * eccentricity + (-eccentricity + 1) * EccentricityColor;
    }

    if (MAGNITUDECOLORMODE) {
        tira::image<float> L1 = Ln.channel(1);                  // get the magnitude of the largest eigenvector
        if(EIGENVALUE_SIGN == -1)
            L1 = L1.clamp(-INFINITY, 0).abs();
        else if(EIGENVALUE_SIGN == +1)
            L1 = L1.clamp(0, INFINITY);
        else
            L1 = L1.abs();
        if(MAGNITUDECOLORTHRESHOLD < 0)
            MAGNITUDECOLORTHRESHOLD = L1.maxv();
        L1MaxMag = L1.maxv();
        tira::image<float> L1norm = (L1 / MAGNITUDECOLORTHRESHOLD).clamp(0.0f, 1.0f);

        tira::image<float> MagnitudeColor(AngleColor.X(), AngleColor.Y(), AngleColor.C());
        if (MAGNITUDECOLORMODE == AdjustColorType::Lighten)
            MagnitudeColor = 255;
        else
            MagnitudeColor = 0;

        AngleColor = AngleColor * L1norm + (-L1norm + 1) * MagnitudeColor;
    }
    CurrentColormap = AngleColor;
    return CurrentColormap;
}

tira::image<unsigned char> ColormapEccentricity() {

    tira::image<float> ecc = CalculateEccentricity();


    MAXVAL = 1.0;
    MINVAL = 0.0;
    CurrentColormap = ecc.cmap(0, 1, ColorMap::Magma);
    return CurrentColormap;
}


/// <summary>
/// Create a scalar image from the specified tensor component
/// </summary>
/// <param name="u"></param>
/// <param name="v"></param>
void ScalarFrom_TensorElement2D(unsigned int u, unsigned int v) {

    SCALAR = tira::image<float>(Tn.shape()[1], Tn.shape()[0], 1);
    float val;
    for (int yi = 0; yi < Tn.shape()[0]; yi++) {
        for (int xi = 0; xi < Tn.shape()[1]; xi++) {
            val = Timg(xi, yi, u, v);
            SCALAR(xi, yi, 0) = val;
        }
    }

    CurrentColormap = ColormapTensor(u, v);

    CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Eval(unsigned int i) {

    SCALAR = Ln.channel(i);

    // update texture
    MAXVAL = SCALAR.maxv();
    MINVAL = SCALAR.minv();
    if (CMAP_MATERIAL) {
        CurrentColormap = ColormapEval(i);
        CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
    }
}


/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Eccentricity() {

    CurrentColormap = ColormapEccentricity();
    CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvector and component
/// </summary>
/// <param name="i"></param>
void ScalarFrom_Evec(unsigned int i) {

    SCALAR = THETAn.channel(i);

    CurrentColormap = ColormapEvec(i);

    CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
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

    ImGui::Text("%s", ("Field Size: " + ss.str()).c_str());
    ImGui::Text("Maximum Norm: %f", MAXNORM);
}

void RegenerateGlyphs() {
    if (!RENDER_GLYPHS) return;                     // don't update if they aren't being displayed

    tira::geometry<float> circle = tira::circle<float>(GLYPH_TESSELATION).scale({ 0.0f, 0.0f });

    tira::geometry<float> glyphrow = circle.tile({1.0f, 0.0f, 0.0f}, Tn.width());
    tira::geometry<float> glyphgrid = glyphrow.tile({0.0f, 1.0f, 0.0f}, Tn.height());

    GLYPH_GEOMETRY = tira::glGeometry(glyphgrid);
    GLYPH_MATERIAL->SetTexture("lambda", Ln, GL_RG32F, GL_NEAREST);
    GLYPH_MATERIAL->SetTexture("evecs", THETAn, GL_RG32F, GL_NEAREST);
}

/// This function renders the user interface every frame
void RenderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS
    ImGui::Text("File: %s", FileName == NULL ? "N/A" : FileName);

    if (ImGui::Button("Load File"))					// create a button for loading the shader
        ImGuiFileDialog::Instance()->OpenDialog("ChooseNpyFile", "Choose NPY File", ".npy,.npz", ".");
    if (ImGuiFileDialog::Instance()->Display("ChooseNpyFile")) {				    // if the user opened a file dialog
        if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
            std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file
            std::string extension = filename.substr(filename.find_last_of(".") + 1);
            if (extension == "npy") {
                LoadTensorField(filename);
                SCALARTYPE = ScalarType::EVal0;
                ScalarRefresh();
                FIELD_LOADED = true;
                RENDER_GLYPHS = true;

                if (GLYPH_MATERIAL == NULL) {
                    GLYPH_GEOMETRY = tira::glGeometry::GenerateCircle<float>(100);
                    GLYPH_MATERIAL = new tira::glMaterial(glyph_shader_string);
                }
            }
        }
        ImGuiFileDialog::Instance()->Close();									// close the file dialog box		
    }

    RenderFieldSpecs();

    // select scalar component
    if (ImGui::Button("Reset View")) CameraZoom = 1.0f;

    if (ImGui::Button("Save Image"))					// create a button for loading the shader
        ImGuiFileDialog::Instance()->OpenDialog("ChooseBmpFile", "Choose BMP File", ".bmp", ".");
    if (ImGuiFileDialog::Instance()->Display("ChooseBmpFile")) {				    // if the user opened a file dialog
        if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
            std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file
            std::string extension = filename.substr(filename.find_last_of(".") + 1);
            if (extension == "bmp") {
                CurrentColormap.save(filename);
            }
        }
        ImGuiFileDialog::Instance()->Close();									// close the file dialog box		
    }

    ImGui::SeparatorText("Eigenvector Display");
    if (ImGui::RadioButton("evec 0 (theta)", &SCALARTYPE, (int)ScalarType::EVec0)) {
        ScalarRefresh();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("evec 1 (theta)", &SCALARTYPE, (int)ScalarType::EVec1)) {
        ScalarRefresh();
    }
    ImGui::Columns(2);
    ImGui::Text("Eccentricity");
    if (ImGui::RadioButton("None##1", &ECCENTRICITYCOLORMODE, AdjustColorType::NoAdjustment))
        ScalarRefresh();
    if (ImGui::RadioButton("Lighten##1", &ECCENTRICITYCOLORMODE, AdjustColorType::Lighten))
        ScalarRefresh();
    if (ImGui::RadioButton("Darken##1", &ECCENTRICITYCOLORMODE, AdjustColorType::Darken))
        ScalarRefresh();

    ImGui::NextColumn();
    ImGui::Text("Magnitude");
    if (ImGui::RadioButton("None##2", &MAGNITUDECOLORMODE, AdjustColorType::NoAdjustment))
        ScalarRefresh();
    if (ImGui::RadioButton("Lighten##2", &MAGNITUDECOLORMODE, AdjustColorType::Lighten))
        ScalarRefresh();
    if (ImGui::RadioButton("Darken##2", &MAGNITUDECOLORMODE, AdjustColorType::Darken))
        ScalarRefresh();
    ImGui::SetNextItemWidth(160);
    if (ImGui::InputFloat("Scale", &MAGNITUDECOLORTHRESHOLD, L1MaxMag * 0.01, L1MaxMag * 0.1)) {
        if (MAGNITUDECOLORTHRESHOLD < 0) MAGNITUDECOLORTHRESHOLD = 0;
        ScalarRefresh();
    }
    if (ImGui::Button("Reset")) {
        MAGNITUDECOLORTHRESHOLD = L1MaxMag;
        ScalarRefresh();
    }

    ImGui::Columns(1);

    ImGui::SeparatorText("Scalar Display");
    ImGui::RadioButton("None", &SCALARTYPE, (int)ScalarType::NoScalar);
    if (ImGui::RadioButton("[0, 0] = dxdx", &SCALARTYPE, (int)ScalarType::Tensor00)) {
        ScalarRefresh();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("[1, 1] = dydy", &SCALARTYPE, (int)ScalarType::Tensor11)) {
        ScalarRefresh();
    }
    if (ImGui::RadioButton("[0, 1] = dxdy", &SCALARTYPE, (int)ScalarType::Tensor01)) {
        ScalarRefresh();
    }
    if (ImGui::RadioButton("lambda 0", &SCALARTYPE, (int)ScalarType::EVal0)) {
        ScalarRefresh();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("lambda 1", &SCALARTYPE, (int)ScalarType::EVal1)) {
        ScalarRefresh();
    }
    
    if (ImGui::RadioButton("eccentricity", &SCALARTYPE, (int)ScalarType::Eccentricity)) {
        ScalarFrom_Eccentricity();
    }
    if(ImGui::RadioButton("Negative Evals", &EIGENVALUE_SIGN, -1)) {
        ScalarRefresh();
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Magnitude Evals", &EIGENVALUE_SIGN, 0)) {
        ScalarRefresh();
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Positive Evals", &EIGENVALUE_SIGN, 1)) {
        ScalarRefresh();
    }
    std::stringstream ss;
    ss << "Min: " << MINVAL << "\t Max: " << MAXVAL;
    ImGui::Text("%s", ss.str().c_str());

    if (ImGui::TreeNodeEx("Processing", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Bake")) {
            T0 = Tn;
            PROCESSINGTYPE = ProcessingType::NoProcessing;
        }
        if (ImGui::RadioButton("None", &PROCESSINGTYPE, (int)ProcessingType::NoProcessing)) {
            Tn = T0;
            UpdateEigens();
            RegenerateGlyphs();
            ScalarRefresh();
        }
        ImGui::SeparatorText("Gaussian Blur");
        if (ImGui::RadioButton("Blur", &PROCESSINGTYPE, (int)ProcessingType::Gaussian)) {
            if (PROCESSINGTYPE == ProcessingType::Gaussian) {
                GaussianFilter(SIGMA);
                UpdateEigens();
                RegenerateGlyphs();
            }
            else {
                Tn = T0;
                UpdateEigens();
                RegenerateGlyphs();
            }
            ScalarRefresh();
        }
        ImGui::SameLine();
        if (ImGui::InputFloat("##Sigma", &SIGMA, 0.2f, 1.0f)) {
            if (SIGMA <= 0) SIGMA = 0.01;
            if (PROCESSINGTYPE == ProcessingType::Gaussian) {
                GaussianFilter(SIGMA);
                UpdateEigens();
                RegenerateGlyphs();
                ScalarRefresh();
            }
        }

        ImGui::SeparatorText("Tensor Voting");
        if (ImGui::RadioButton("Tensor Voting", &PROCESSINGTYPE, (int)ProcessingType::Vote)) {
            TensorVote(TV_SIGMA1, TV_P, TV_SIGMA2, TV_STICK, TV_PLATE);
            UpdateEigens();
            RegenerateGlyphs();
            ScalarRefresh();
        }
        if (ImGui::InputFloat("Sigma 1", &TV_SIGMA1, 0.2f, 1.0f)) {
            if (TV_SIGMA1 < 0) TV_SIGMA1 = 0.0;
            if (PROCESSINGTYPE == ProcessingType::Vote) {
                TensorVote(TV_SIGMA1, TV_P, TV_SIGMA2, TV_STICK, TV_PLATE);
                UpdateEigens();
                RegenerateGlyphs();
                ScalarRefresh();
            }
        }
        if (ImGui::InputFloat("Sigma 2", &TV_SIGMA2, 0.2f, 1.0f)) {
            if (TV_SIGMA2 < 0) TV_SIGMA2 = 0.0;
            if (PROCESSINGTYPE == ProcessingType::Vote) {
                TensorVote(TV_SIGMA1, TV_P, TV_SIGMA2, TV_STICK, TV_PLATE);
                UpdateEigens();
                RegenerateGlyphs();
                ScalarRefresh();
            }
        }
        if (ImGui::InputInt("Power", &TV_P, 1, 5)) {
            if (TV_P < 1) TV_P = 1;
            if (PROCESSINGTYPE == ProcessingType::Vote) {
                TensorVote(TV_SIGMA1, TV_P, TV_SIGMA2, TV_STICK, TV_PLATE);
                UpdateEigens();
                RegenerateGlyphs();
                ScalarRefresh();
            }
        }
        if (ImGui::Checkbox("Stick", &TV_STICK)) {
            TensorVote(TV_SIGMA1, TV_P, TV_SIGMA2, TV_STICK, TV_PLATE);
            UpdateEigens();
            RegenerateGlyphs();
            ScalarRefresh();
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Plate", &TV_PLATE)) {
            TensorVote(TV_SIGMA1, TV_P, TV_SIGMA2, TV_STICK, TV_PLATE);
            UpdateEigens();
            RegenerateGlyphs();
            ScalarRefresh();
        }

        ImGui::TreePop();
    }

    if (ImGui::Checkbox("Glyphs", &RENDER_GLYPHS)) {
        RegenerateGlyphs();
    }
    ImGui::InputFloat("Glyph Scale", &GLYPH_SCALE, 0.1f, 1.0f);
    if(ImGui::InputInt("Tesselate", &GLYPH_TESSELATION, 1, 10)) {
        RegenerateGlyphs();
    }
    ImGui::Checkbox("Scale by Norm", &GLYPH_NORMALIZE);

    int FieldIndex[2] = { (int)MousePos[0], (int)MousePos[1] };

    if (FIELD_LOADED) {
        if ((FieldIndex[0] < 0 || FieldIndex[0] >= Tn.width()) && (FieldIndex[1] < 0 || FieldIndex[1] >= Tn.height()))
            ImGui::Text("Field Index: ---, ---");
        else if (FieldIndex[1] < 0 || FieldIndex[1] >= Tn.height())
            ImGui::Text("Field Index: %d, ---", FieldIndex[0]);
        else if (FieldIndex[0] < 0 || FieldIndex[0] >= Tn.width())
            ImGui::Text("Field Index: ---, %d", FieldIndex[1]);
        else {
            ImGui::Text("Field Index: %d, %d", FieldIndex[0], FieldIndex[1]);


            // get the current tensor value
            glm::mat2 T = Tn(FieldIndex[0], FieldIndex[1]);

            // display the current tensor as a matrix
            ImGui::Text("Tensor:");
            float Row0[2] = { T[0][0], T[0][1] };
            ImGui::InputFloat2("##Row0", Row0, "%1.5F");
            float Row1[2] = { T[1][0], T[1][1] };
            ImGui::InputFloat2("##Row1", Row1, "%1.5F");

            // calculate the eigenvalues
            glm::vec2 evals = Eigenvalues2D(T);

            // display the eigenvalues
            ImGui::Text("Eigenvalues:");
            float lambdas[2] = { evals[0], evals[1] };
            ImGui::InputFloat2("##lambdas", lambdas, "%1.5F");

            // calculate the eigenvectors
            glm::vec2 ev0 = Eigenvector2D(T, evals[0]);
            glm::vec2 ev1 = Eigenvector2D(T, evals[1]);

            // display the eigenvectors
            ImGui::Text("Eigenvectors:");
            ImGui::Columns(2);
            ImGui::InputFloat2("x0, y0", (float*)& ev0, "%1.5F");
            ImGui::InputFloat2("x1, y1", (float*)& ev1, "%1.5F");

        }
    }

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
    //io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", ui_scale * 16.0f);

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

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        prevMousePos[0] = xpos;
        prevMousePos[1] = ypos;
    }
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS &&
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        //int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        double xdist = (prevMousePos[0] - xposIn) / display_w * Viewport[0] * CameraZoom;
        double ydist = (prevMousePos[1] - yposIn) / display_h * Viewport[1] * CameraZoom;
        CameraPos[0] += xdist;
        CameraPos[1] += ydist;
        prevMousePos[0] = xposIn;
        prevMousePos[1] = yposIn;
    }
    else {
        
        float x_adjustment = (Viewport[0] - (float)Tn.width()) / 2.0f;
        float y_adjustment = (Viewport[1] - (float)Tn.height()) / 2.0f;
        MousePos[0] = (float)xposIn / (float)display_w * Viewport[0] - x_adjustment;
        MousePos[1] = (float)yposIn / (float)display_h * Viewport[1] - y_adjustment;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        CameraZoom -= CameraZoom * (yoffset * 0.25);
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
        LoadTensorField(in_inputname);
        FileName = in_inputname.c_str();
        GLYPH_ROWS = Tn.shape()[0];
    }
    else {
    }

    if(FIELD_LOADED) {
        if (vm.count("blur")) {
            GaussianFilter(in_blur_strength);
        }
        if (vm.count("l0")) ColormapEval(0).save(in_l0_outputname);
        if (vm.count("l1")) ColormapEval(1).save(in_l1_outputname);
        if (vm.count("v0")) ColormapEvec(0).save(in_v0_outputname);
        if (vm.count("v1")) ColormapEvec(1).save(in_v1_outputname);

        if (vm.count("nogui")) {
            return 0;
        }
    }

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
    if (FIELD_LOADED) {
        glfwSetCursorPosCallback(window, mouse_callback);
    }
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    InitUI(window, glsl_version);
    if (glewInit() != GLEW_OK)
        std::cout << "Error!" << std::endl;

    CMAP_GEOMETRY = tira::glGeometry::GenerateRectangle<float>();
    CMAP_MATERIAL = new tira::glMaterial(colormap_shader_string);
    GLYPH_MATERIAL = new tira::glMaterial(glyph_shader_string);
    
    if (FIELD_LOADED) {
        ScalarRefresh();
        RegenerateGlyphs();
        CameraPos = glm::vec3(Tn.width() / 2.0f, Tn.height() / 2.0f, 0.0f);
    }

    testmaterial = new tira::glMaterial(test_shader_string);
    testmaterial->Begin();
    tira::glGeometry testgeometry = tira::glGeometry::GenerateRectangle<float>();

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        RenderUI();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);                     // specifies the area of the window where OpenGL can render
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        // if a tensor field is loaded
        if (FIELD_LOADED) {

            // Calculate the viewport width and height based on the dimensions of the tensor field and the screen (so that the field isn't distorted)
            FitRectangleToWindow(Tn.width(), Tn.height(), display_w, display_h, Viewport[0], Viewport[1]);

            glm::vec2 viewport(Viewport[0], Viewport[1]);
            glm::vec2 view_extent = viewport * CameraZoom / 2.0f;
            glm::mat4 Mview = glm::ortho(-view_extent[0] + CameraPos[0], 
                view_extent[0] + CameraPos[0], 
                view_extent[1] + CameraPos[1], 
                -view_extent[1] + CameraPos[1]);   // create a view matrix
            glm::vec3 center((float)Tn.width() / 2.0f, (float)Tn.height() / 2.0f, 0.0f);
            
            
            // if the user is visualizing a scalar component of the tensor field as a color map
            if (SCALARTYPE != ScalarType::NoScalar) {

                glm::vec3 scale((float)Tn.width(), (float)Tn.height(), 1.0f);
                glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), scale);                                                      // compose the scale matrix from the width and height of the tensor field
                glm::mat4 Mtrans = glm::translate(glm::mat4(1.0f), center);
                //glm::mat4 Mtrans = glm::mat4(1.0f);                                                                         // there is no translation (the 2D field is centered at the origin)
                
                glm::mat4 Mobj = Mtrans * Mscale;                                                                      // create the transformation matrix
                CMAP_MATERIAL->Begin();                                                                                     // begin using the scalar colormap material
                CMAP_MATERIAL->SetUniformMat4f("Mview", Mview);                                                                 // pass the transformation matrix as a uniform
                CMAP_MATERIAL->SetUniformMat4f("Mobj", Mobj);
                CMAP_GEOMETRY.Draw();                                                                                       // draw the rectangle
                CMAP_MATERIAL->End();                                                                                       // stop using the material
            }
            // if the user is rendering glyphs
            if (RENDER_GLYPHS) {

                glm::mat4 Mtrans = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.0f));
                glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), glm::vec3(1.0, 1.0f, 1.0f));
                glm::mat4 Mobj = Mtrans * Mscale;


                GLYPH_MATERIAL->Begin();
                GLYPH_MATERIAL->SetUniform1f("scale", GLYPH_SCALE);
                if(GLYPH_NORMALIZE)
                    GLYPH_MATERIAL->SetUniform1f("norm", L1MaxMag);
                else
                    GLYPH_MATERIAL->SetUniform1f("norm", 0.0f);
                GLYPH_MATERIAL->SetUniformMat4f("Mview", Mview);
                GLYPH_MATERIAL->SetUniformMat4f("Mobj", Mobj);
                GLYPH_GEOMETRY.Draw();
                GLYPH_MATERIAL->End();
            }

        }


        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer

        glfwPollEvents();
    }

    DestroyUI();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW

    return 0;
}