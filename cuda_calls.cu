#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <tira/cuda/cudaGaussianFilter2D.cuh>


glm::mat2* cudaGaussianBlur(glm::mat2* source, size_t width, size_t height, float sigma,
                            size_t& out_width, size_t& out_height, int deviceID = 0) {

    cudaSetDevice(deviceID);
    glm::mat2* dest = GaussianFilter2D_cpu<glm::mat2>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}