#include <glm/glm.hpp>
#include <tira/cuda/cudaGaussianFilter2D.cuh>
#include <tira/cuda/cudaEigen2D.cuh>


glm::mat2* cudaGaussianBlur(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
                            unsigned int& out_width, unsigned int& out_height, int deviceID = 0) {

    cudaSetDevice(deviceID);
    glm::mat2* dest = tira::cuda::GaussianFilter2D<glm::mat2>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}

float* cudaGaussianBlur(float* source, unsigned int width, unsigned int height, float sigma,
    unsigned int& out_width, unsigned int& out_height, int deviceID = 0) {

    cudaSetDevice(deviceID);
    float* dest = tira::cuda::GaussianFilter2D<float>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}

float* cudaEigenvalues(float* tensors, unsigned int n) {
    return tira::cuda::Eigenvalues2D<float>(tensors, n);
}

float* cudaEigenvectorsPolar(float* tensors, float* evals, unsigned int n) {
    return tira::cuda::Eigenvectors2DPolar<float>(tensors, evals, n);
}

void cudaEigenvalue0(float* tensors, unsigned int n, float* evals) {
    float* both_evals = tira::cuda::Eigenvalues2D<float>(tensors, n);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i];
    }
    free(both_evals);
}

void cudaEigenvalue1(float* tensors, unsigned int n, float* evals) {
    float* both_evals = tira::cuda::Eigenvalues2D<float>(tensors, n);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i + 1];
    }
    free(both_evals);
}