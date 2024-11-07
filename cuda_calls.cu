#include <glm/glm.hpp>
#include <tira/cuda/cudaGaussianFilter.cuh>
#include <tira/cuda/cudaEigen.cuh>


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

glm::mat3* cudaGaussianBlur3D(glm::mat3* source, unsigned int width, unsigned int height, unsigned int depth, float sigma,
    unsigned int& out_width, unsigned int& out_height, unsigned int& out_depth, int deviceID = 0) {

    cudaSetDevice(deviceID);
    glm::mat3* dest = tira::cuda::GaussianFilter3D<glm::mat3>(source, width, height, depth, sigma, sigma, sigma, out_width, out_height, out_depth);

    return dest;
}

float* cudaGaussianBlur3D(float* source, unsigned int width, unsigned int height, unsigned int depth, float sigma,
    unsigned int& out_width, unsigned int& out_height, unsigned int& out_depth, int deviceID = 0) {

    cudaSetDevice(deviceID);
    float* dest = tira::cuda::GaussianFilter3D<float>(source, width, height, depth, sigma, sigma, sigma, out_width, out_height, out_depth);

    return dest;
}

float* cudaEigenvalues2(float* tensors, unsigned int n, int device) {
    return tira::cuda::Eigenvalues2D<float>(tensors, n, device);
}

float* cudaEigenvalues3(float* tensors, unsigned int n, int device) {
    return tira::cuda::Eigenvalues3D<float>(tensors, n, device);
}

float* cudaEigenvectors3(float* tensors, float* lambda, size_t n, int device) {
    return tira::cuda::Eigenvectors3DPolar<float>(tensors, lambda, n, device);
}

float* cudaEigenvectorsPolar(float* tensors, float* evals, unsigned int n, int device) {
    return tira::cuda::Eigenvectors2DPolar<float>(tensors, evals, n, device);
}

void cudaEigenvalue0(float* tensors, unsigned int n, float* evals, int device) {
    float* both_evals = tira::cuda::Eigenvalues2D<float>(tensors, n, device);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i];
    }
    free(both_evals);
}

void cudaEigenvalue1(float* tensors, unsigned int n, float* evals, int device) {
    float* both_evals = tira::cuda::Eigenvalues2D<float>(tensors, n, device);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i + 1];
    }
    free(both_evals);
}