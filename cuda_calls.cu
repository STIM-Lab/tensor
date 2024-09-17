#include <glm/glm.hpp>
#include <tira/cuda/cudaGaussianFilter2D.cuh>
#include <tira/cuda/cudaEigen2D.cuh>


glm::mat2* cudaGaussianBlur(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
                            unsigned int& out_width, unsigned int& out_height, int deviceID = 0) {

    cudaSetDevice(deviceID);
    glm::mat2* dest = GaussianFilter2D<glm::mat2>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}

void cudaEigenvalue0(float* tensors, unsigned int n, float* evals) {
    float* both_evals = Eigenvalues2D<float>(tensors, n);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i];
    }
    free(both_evals);
}

void cudaEigenvalue1(float* tensors, unsigned int n, float* evals) {
    float* both_evals = Eigenvalues2D<float>(tensors, n);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i + 1];
    }
    free(both_evals);
}