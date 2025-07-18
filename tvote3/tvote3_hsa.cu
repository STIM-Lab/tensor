#include <tira/tensorvote.cuh>
#include <tira/filter.cuh>

/// Heterogeneous System Architecture calls (functions that decide how to execute a function)

float* hsa_eigenvalues3(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evals3_symmetric<float>(tensors, n);

    HANDLE_ERROR(cudaSetDevice(device));
    return tira::cuda::evals3_symmetric<float>(tensors, n);
}

float* hsa_eigenvectors3spherical(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evecs3spherical_symmetric(tensors, evals, n);

    HANDLE_ERROR(cudaSetDevice(device));
    return tira::cuda::evecs3spherical_symmetric<float>(tensors, evals, n);
    //throw std::runtime_error("Not implemented");
}

glm::mat3* hsa_gaussian3(const glm::mat3* source, const unsigned int s0, const unsigned int s1, const unsigned int s2, const float sigma, glm::vec3 pixel_size,
    unsigned int& out_s0, unsigned int& out_s1, unsigned int& out_s2, const int deviceID = 0) {

	// CPU implementation
    if (deviceID < 0) {
        unsigned kernel_size0 = (unsigned)(6 * sigma * pixel_size[0]);
        float* gauss0 = tira::cpu::kernel_gaussian<float>(kernel_size0, 0, sigma, pixel_size[0]);
        glm::mat3* dest_0 = tira::cpu::convolve3<glm::mat3, float>(source, s0, s1, s2, gauss0, kernel_size0, 1, 1, out_s0, out_s1, out_s2);
        free(gauss0);

        unsigned kernel_size1 = (unsigned)(6 * sigma * pixel_size[1]);
        float* gauss1 = tira::cpu::kernel_gaussian<float>(kernel_size1, 0, sigma, pixel_size[1]);
        glm::mat3* dest_1 = tira::cpu::convolve3<glm::mat3, float>(dest_0, out_s0, out_s1, out_s2, gauss1, 1, kernel_size1, 1, out_s0, out_s1, out_s2);
        free(gauss1);
        free(dest_0);

        unsigned kernel_size2 = (unsigned)(6 * sigma * pixel_size[2]);
        float* gauss2 = tira::cpu::kernel_gaussian<float>(kernel_size2, 0, sigma, pixel_size[2]);
        glm::mat3* dest_2 = tira::cpu::convolve3<glm::mat3, float>(dest_1, out_s0, out_s1, out_s2, gauss2, 1, 1, kernel_size2, out_s0, out_s1, out_s2);
        free(gauss2);
        free(dest_1);

        return dest_2;
    }

	// CUDA implementation
    cudaSetDevice(deviceID);
    glm::mat3* dest = tira::cuda::gaussian_filter3d<glm::mat3>(source, s0, s1, s2, sigma, sigma, sigma, out_s0, out_s1, out_s2);
    return dest;
}

void hsa_tensorvote3(const float* input_field, float* output_field, unsigned int s0, unsigned int s1, unsigned int s2, glm::vec3 pixel_size, float sigma, float sigma2,
    unsigned int power, int device, bool STICK, bool PLATE, bool debug, unsigned samples) {

    throw std::runtime_error("Not implemented");
}


