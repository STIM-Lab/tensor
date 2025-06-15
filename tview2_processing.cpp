#include <glm/glm.hpp>
#include <tira/image.h>
#include "tview2.h"
#include <tira/math/eigen.h>
#include <tira/math/tensorvote.h>

extern TV2_UI UI;

float* EigenValues2(float* tensors, unsigned int n, int device);
float* EigenVectors2DPolar(float* tensors, float* evals, unsigned int n, int device);
glm::mat2* GaussianBlur2D(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
                            unsigned int& out_width, unsigned int& out_height, int deviceID = 0);
void cudaVote2D(float* input_field, float* output_field, unsigned int s0, unsigned int s1, float sigma, float sigma2,
    unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug);

inline float Timg(tira::image<glm::mat2>* tensors, const unsigned int x, const unsigned int y, const unsigned int u, const unsigned int v) {
    return (*tensors)(x, y)[static_cast<int>(u)][static_cast<int>(v)];
}

inline float Trace2D(tira::image<glm::mat2>* tensors, const unsigned int x, const unsigned int y) {
    return Timg(tensors, x, y, 0, 0) + Timg(tensors, x, y, 1, 1);
}

inline float Determinant2D(tira::image<glm::mat2>* tensors, const unsigned int x, const unsigned int y) {
    return Timg(tensors, x, y, 0, 0) * Timg(tensors, x, y, 1, 1) - pow(Timg(tensors, x, y, 0, 1), 2.0f);
}

inline float Eigenvalue2D(tira::image<glm::mat2>* tensors, const unsigned int x, const unsigned int y, const unsigned int i) {
    const float d = Timg(tensors, x, y, 0, 0);
    const float e = Timg(tensors, x, y, 0, 1);
    const float f = e;
    const float g = Timg(tensors, x, y, 1, 1);

    const float dpg = d + g;
    const float disc = sqrt((4.0f * e * f) + pow(d - g, 2.0f));
    const float a = (dpg + disc) / 2.0f;
    const float b = (dpg - disc) / 2.0f;
    if (i == 0) return std::max(a, b);
    else return std::min(a, b);
}

inline glm::vec2 Eigenvector2D(tira::image<glm::mat2>* tensors, unsigned int x, unsigned int y, unsigned int i) {

    const float lambda = Eigenvalue2D(tensors, x, y, i);

    const float d = Timg(tensors, x, y, 0, 0);                 // dx * dx
    const float e = Timg(tensors, x, y, 0, 1);                 // dx * dy
    const float g = Timg(tensors, x, y, 1, 1);                 // dy * dy

    if (e != 0)
        return glm::normalize(glm::vec2(1.0f, (lambda - d) / e));
    else if (g == 0)
        return { 1.0f, 1.0f };
    else
        return { 0.0f, 1.0f };
}

inline float normaldist(const float x, const float sigma) {
    const float scale = 1.0f / (sigma * sqrt(2 * 3.14159f));
    const float ex = -(x * x) / (2 * sigma * sigma);
    return scale * exp(ex);
}

/// <summary>
/// Calculate the eigenvectors and eigenvalues of a 2D tensor field using the CPU or CUDA
/// </summary>
void EigenDecomposition(tira::image<glm::mat2>* tensor, tira::image<float>* lambda, tira::image<float>* theta, int cuda_device) {

    float* eigenvalues_raw = EigenValues2(reinterpret_cast<float*>(tensor->data()), tensor->X() * tensor->Y(), cuda_device);
    *lambda = tira::image<float>(eigenvalues_raw, tensor->X(), tensor->Y(), 2);

    // calculate and store the largest eigenvalue magnitude
    float SmallestEigenvalue = lambda->minv();
    float LargestEigenvalue = lambda->maxv();
    UI.largest_eigenvalue_magnitude = std::max(abs(SmallestEigenvalue), abs(LargestEigenvalue));

    float* eigenvectors_raw = EigenVectors2DPolar(reinterpret_cast<float*>(tensor->data()), eigenvalues_raw, tensor->X() * tensor->Y(), cuda_device);
    *theta = tira::image<float>(eigenvectors_raw, tensor->X(), tensor->Y(), 2);
    free(eigenvalues_raw);
    free(eigenvectors_raw);
}

float Eccentricity2(float l0, float l1) {
    if (l1 == 0) return 0;
    const float l0_2 = std::pow(l0, 2.0f);
    const float l1_2 = std::pow(l1, 2.0f);
    return std::sqrt(1.0f - l0_2 / l1_2);
}

float LinearEccentricity2(float l0, float l1) {
    return std::sqrt((l1 * l1) - (l0 * l0));
}

void ImageFrom_Eccentricity(tira::image<float>* lambda, tira::image<float>* eccentricity) {
    //tira::image<float> ecc(lambda->X(), lambda->Y());
    eccentricity->resize({lambda->X(), lambda->Y()});

    for (size_t yi = 0; yi < eccentricity->Y(); yi++) {
        for (size_t xi = 0; xi < eccentricity->X(); xi++) {
            float l0 = (*lambda)(xi, yi, 0);
            float l1 = (*lambda)(xi, yi, 1);
            (*eccentricity)(xi, yi) = Eccentricity2(l0, l1);
        }
    }
}

void ImageFrom_LinearEccentricity(tira::image<float>* lambda, tira::image<float>* eccentricity) {
    //tira::image<float> ecc(lambda->X(), lambda->Y());
    eccentricity->resize({ lambda->X(), lambda->Y() });

    for (size_t yi = 0; yi < eccentricity->Y(); yi++) {
        for (size_t xi = 0; xi < eccentricity->X(); xi++) {
            float l0 = (*lambda)(xi, yi, 0);
            float l1 = (*lambda)(xi, yi, 1);
            (*eccentricity)(xi, yi) = LinearEccentricity2(l0, l1);
        }
    }
}

/// <summary>
/// Create a scalar image from the specified tensor component
/// </summary>
/// <param name="tensors">pointer to the tensor field</param>
/// <param name="elements">scalar image that will store the individual tensor components that are extracted</param>
/// <param name="u">first index of the tensor component to extract</param>
/// <param name="v">second index of the tensor component to extract</param>
void ImageFrom_TensorElement2D(tira::image<glm::mat2>* tensors, tira::image<float>* elements, const unsigned int u, const unsigned int v) {

    *elements = tira::image<float>(tensors->shape()[1], tensors->shape()[0], 1);
    for (unsigned int yi = 0; yi < tensors->shape()[0]; yi++) {
        for (unsigned int xi = 0; xi < tensors->shape()[1]; xi++) {
            const float val = Timg(tensors, xi, yi, u, v);
            (*elements)(xi, yi, 0) = val;
        }
    }

    //CurrentColormap = ColormapTensor(u, v);
    //CMAP_MATERIAL->SetTexture("mapped_image", CurrentColormap, GL_RGB8, GL_NEAREST);
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="i"></param>
void ImageFrom_Eigenvalue(tira::image<float>* lambda, tira::image<float>* scalar, unsigned int i) {
    *scalar = lambda->channel(i);
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="i"></param>
void ImageFrom_Theta(tira::image<float>* theta, tira::image<float>* scalar, unsigned int i) {
    *scalar = theta->channel(i);
}




/// <summary>
/// Blurs the tensor field and re-calculates the current scalar image
/// </summary>
/// <param name="sigma"></param>
void GaussianFilter(tira::image<glm::mat2>* tensors_in, tira::image<glm::mat2>* tensors_out, const float sigma, int cuda_device) {
    // if a CUDA device is enabled, use a blur kernel
    if (cuda_device >= 0) {
        unsigned int blur_width;
        unsigned int blur_height;
        glm::mat2* blurred = GaussianBlur2D(tensors_in->data(), tensors_in->X(), tensors_in->Y(), sigma, blur_width, blur_height, cuda_device);

        *tensors_out = tira::image<glm::mat2>(blurred, blur_width, blur_height);
        free(blurred);
    }
    // otherwise use the CPU
    else {
        const unsigned int size = ceil(sigma * 6);
        const float start = -static_cast<float>(size - 1) / 2.0f;
        tira::image<float> Kx(size, 1);
        tira::image<float> Ky(1, size);
        for (size_t i = 0; i < size; i++) {
            constexpr float dx = 1.0f;
            const float v = normaldist(start + dx * static_cast<float>(i), sigma);
            Kx(i, 0, 0) = v;
            Ky(0, i, 0) = v;
        }
        *tensors_out = tensors_in->convolve2(Kx);
        *tensors_out = tensors_out->convolve2(Ky);
    }
}

/// <summary>
/// Perform tensor voting. This function decides which voting method to use (CUDA or CPU) based on the device ID
/// </summary>
/// <param name="sigma"></param>
/// <param name="p"></param>
/// <param name="sigma2"></param>
/// <param name="stick"></param>
/// <param name="plate"></param>
void TensorVote(tira::image<glm::mat2>* tensors_in, tira::image<glm::mat2>* tensors_out,
    const float sigma, const unsigned int p, const float sigma2, const bool stick, const bool plate, int cuda_device) {
    *tensors_out = tira::image<glm::mat2>(tensors_in->X(), tensors_in->Y());

    const auto w = static_cast<unsigned int>(6.0f * std::max(sigma, sigma2) + 1.0f);

    if (cuda_device >= 0) {
        cudaVote2D(reinterpret_cast<float*>(tensors_in->data()), reinterpret_cast<float*>(tensors_out->data()),
            static_cast<unsigned int>(tensors_in->shape()[0]), static_cast<unsigned int>(tensors_in->shape()[1]),
            sigma, sigma2, w, p, cuda_device, stick, plate, false);
    }
    else {
        //throw std::runtime_error("ERROR: no CPU implementation of tensor voting");
        float* lambdas = tira::cpu::Eigenvalues2D<float>((float*)tensors_in->data(), tensors_in->size());
        float* evecs = tira::cpu::Eigenvectors2DPolar<float>((float*)tensors_in->data(), lambdas, tensors_in->size());
        tira::cpu::tensorvote2(tensors_out->data(), (glm::vec2*)lambdas, (glm::vec2*)evecs, glm::vec2(sigma, sigma2), p, 
            w, tensors_in->shape()[0], tensors_in->shape()[1], stick, plate);
    }
}
