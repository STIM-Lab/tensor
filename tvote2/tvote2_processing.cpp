#include <glm/glm.hpp>
#include <tira/image.h>
#include "tvote2.h"
#include <tira/eigen.h>
#include <tira/tensorvote.h>

extern TV2_UI UI;

inline static float Timg(tira::image<glm::mat2>* tensors, const unsigned int x, const unsigned int y, const unsigned int u, const unsigned int v) {
    return (*tensors)(x, y)[static_cast<int>(u)][static_cast<int>(v)];
}

inline static float normaldist(const float x, const float sigma) {
    const float scale = 1.0f / (sigma * std::sqrt(2 * 3.14159f));
    const float ex = -(x * x) / (2 * sigma * sigma);
    return scale * std::exp(ex);
}


/// <summary>
/// Calculate the eigenvectors and eigenvalues of a 2D tensor field using the CPU or CUDA
/// </summary>
void EigenDecomposition(tira::image<glm::mat2>* tensor, tira::image<float>* lambda, tira::image<float>* theta, int cuda_device) {

    float* eigenvalues_raw = hsa_eigenvalues2(reinterpret_cast<float*>(tensor->data()), tensor->X() * tensor->Y(), cuda_device);
    *lambda = tira::image<float>(eigenvalues_raw, tensor->X(), tensor->Y(), 2);

    // calculate and store the largest eigenvalue magnitude
    float SmallestEigenvalue = lambda->minv();
    float LargestEigenvalue = lambda->maxv();
    float SmallestEigenvalueMag = std::abs(SmallestEigenvalue);
    float LargestEigenvalueMag = std::abs(LargestEigenvalue);
    UI.largest_eigenvalue_magnitude = std::max(SmallestEigenvalueMag, LargestEigenvalueMag);

    float* eigenvectors_raw = hsa_eigenvectors2polar(reinterpret_cast<float*>(tensor->data()), eigenvalues_raw, tensor->X() * tensor->Y(), cuda_device);
    *theta = tira::image<float>(eigenvectors_raw, tensor->X(), tensor->Y(), 2);
    free(eigenvalues_raw);
    free(eigenvectors_raw);
}

float Eccentricity2(const float l0, const float l1) {
    if (l1 == 0) return 0;
    const float l0_2 = std::pow(l0, 2.0f);
    const float l1_2 = std::pow(l1, 2.0f);
    return std::sqrt(1.0f - l0_2 / l1_2);
}

float LinearEccentricity2(const float l0, const float l1) {
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
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="lambda">pointer to the image containing all of the eigenvalues</param>
/// <param name="scalar">pointer to the image that will be filled with the specified eigenvalues</param>
/// <param name="i">eigenvalue to turn into an image</param>
void ImageFrom_Eigenvalue(const tira::image<float>* lambda, tira::image<float>* scalar, const unsigned i) {
    *scalar = lambda->channel(i);
}

/// <summary>
/// Create a scalar image from the specified eigenvalue
/// </summary>
/// <param name="theta">pointer to the image containing the eigenvectors</param>
/// <param name="scalar">pointer to the image that will be filled with the specified eigenvector</param>
/// <param name="i">eigenvector to turn into an image</param>
void ImageFrom_Theta(const tira::image<float>* theta, tira::image<float>* scalar, const unsigned i) {
    *scalar = theta->channel(i);
}




/// <summary>
/// Blurs the tensor field and re-calculates the current scalar image
/// </summary>
/// <param name="tensors_in">input tensor field</param>
/// <param name="tensors_out">output tensor field</param>
/// <param name="sigma">standard deviation of the Gaussian kernel</param>
/// <param name="cuda_device">CUDA device used for filtering (-1 will use the CPU)</param>
void GaussianFilter(const tira::image<glm::mat2>* tensors_in, tira::image<glm::mat2>* tensors_out, const float sigma, const int cuda_device) {
    // if a CUDA device is enabled, use a blur kernel
    if (cuda_device >= 0) {
        unsigned int blur_width;
        unsigned int blur_height;
        glm::mat2* blurred = hsa_gaussian2(tensors_in->const_data(), tensors_in->X(), tensors_in->Y(), sigma, blur_width, blur_height, cuda_device);

        *tensors_out = tira::image<glm::mat2>(blurred, blur_width, blur_height);
        free(blurred);
    }
    // otherwise use the CPU
    else {
        const unsigned int size = std::ceil(sigma * 6);
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
/// <param name="tensors_in">pointer to the input tensor field</param>
/// <param name="tensors_out">output tensor field</param>
/// <param name="sigma">standard deviation for the directional vote component associated with the smallest eigenvector</param>
/// <param name="p">refinement term</param>
/// <param name="sigma2">standard deviation for the directional vote component associated with the largest eigenvector</param>
/// <param name="stick">boolean flag for stick voting</param>
/// <param name="plate">boolean flag for plate voting</param>
/// <param name="cuda_device">CUDA device ID (-1 for CPU)</param>
/// <param name="samples">number of samples used for numerical integration of the plate tensor (0 uses an analytical calculation)</param>
void TensorVote(const tira::image<glm::mat2>* tensors_in, tira::image<glm::mat2>* tensors_out,
    const float sigma, const unsigned int p, const float sigma2, const bool stick, const bool plate, const int cuda_device, const unsigned samples) {
    *tensors_out = tira::image<glm::mat2>(tensors_in->X(), tensors_in->Y());

    const auto w = static_cast<unsigned int>(6.0f * std::max(sigma, sigma2) + 1.0f);

    if (cuda_device >= 0) {
        hsa_tensorvote2(reinterpret_cast<const float*>(tensors_in->const_data()), reinterpret_cast<float*>(tensors_out->data()),
            static_cast<unsigned int>(tensors_in->shape()[0]), static_cast<unsigned int>(tensors_in->shape()[1]),
            sigma, sigma2, w, p, cuda_device, stick, plate, false, samples);
    }
    else {
        //throw std::runtime_error("ERROR: no CPU implementation of tensor voting");
        auto* lambdas = tira::cpu::evals2_symmetric<float>(reinterpret_cast<const float*>(tensors_in->const_data()), tensors_in->size());
        auto* evecs = tira::cpu::evecs2polar_symmetric<float>(reinterpret_cast<const float*>(tensors_in->const_data()), lambdas, tensors_in->size());
        tira::cpu::tensorvote2(tensors_out->data(), reinterpret_cast<glm::vec2*>(lambdas), reinterpret_cast<glm::vec2*>(evecs), glm::vec2(sigma, sigma2), p,
            w, tensors_in->shape()[0], tensors_in->shape()[1], stick, plate, samples);
    }
}
