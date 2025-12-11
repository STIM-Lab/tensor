#include <tira/functions/tensorvote.h>
#include <tira/functions/filter.h>
#include <tira/cuda/error.h>

/// Heterogeneous System Architecture calls (functions that decide how to execute a function)

int numCudaDevices() {
    int ndevices;
    HANDLE_ERROR(cudaGetDeviceCount(&ndevices));
    return ndevices;
}
float* hsa_eigenvalues3(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evals3_symmetric<float>(tensors, n);

    HANDLE_ERROR(cudaSetDevice(device));
    return tira::cuda::evals3_symmetric<float>(tensors, n, device);
}

float* hsa_eigenvectors3spherical(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evecs3spherical_symmetric(tensors, evals, n);

    HANDLE_ERROR(cudaSetDevice(device));
    return tira::cuda::evecs3spherical_symmetric<float>(tensors, evals, n, device);
}

glm::mat3* hsa_gaussian3(const glm::mat3* source, const unsigned int s0, const unsigned int s1, const unsigned int s2, const float sigma, glm::vec3 pixel_size,
    unsigned int& out_s0, unsigned int& out_s1, unsigned int& out_s2, const int device) {

    float sigma_w_scaled = sigma / pixel_size[0];
    float sigma_h_scaled = sigma / pixel_size[1];
    float sigma_d_scaled = sigma / pixel_size[2];

    // Calculate kernel sizes
    unsigned int kernel_size0 = static_cast<unsigned int>(std::ceil(6.0f * sigma_w_scaled));
    if (kernel_size0 % 2 == 0) kernel_size0++;          // ensure odd kernel size

    unsigned int kernel_size1 = static_cast<unsigned int>(std::ceil(6.0f * sigma_h_scaled));
    if (kernel_size1 % 2 == 0) kernel_size1++;          // ensure odd kernel size

    unsigned int kernel_size2 = static_cast<unsigned int>(std::ceil(6.0f * sigma_d_scaled));
    if (kernel_size2 % 2 == 0) kernel_size2++;          // ensure odd kernel size

	// CPU implementation
    if (device < 0) {
        return tira::cpu::gaussian_convolve3<glm::mat3>(source, s0, s1, s2, sigma_w_scaled, sigma_h_scaled, sigma_d_scaled,
                                                        out_s0, out_s1, out_s2);
    }

	// CUDA implementation
    cudaSetDevice(device);
    glm::mat3* dest = tira::cuda::gaussian_filter3d<glm::mat3>(source, s0, s1, s2, sigma_w_scaled, sigma_h_scaled, sigma_d_scaled, 
                                                               out_s0, out_s1, out_s2);
    return dest;
}

void hsa_tensorvote3(const float* input_field, float* output_field, unsigned int s0, unsigned int s1, unsigned int s2, float sigma, float sigma2,
    unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug, unsigned samples) {

	tira::cuda::tensorvote3(input_field, output_field, s0, s1, s2, sigma, sigma2, w,
		power, device, STICK, PLATE, debug, samples);
}


