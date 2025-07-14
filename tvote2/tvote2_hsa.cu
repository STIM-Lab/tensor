#include <tira/tensorvote.cuh>
#include <tira/filter.cuh>

void hsa_tensorvote2(const float* input_field, float* output_field, unsigned int s0, unsigned int s1, float sigma, float sigma2,
    unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug, unsigned samples) {
    tira::cuda::tensorvote2(input_field, output_field, s0, s1, sigma, sigma2,
        w, power, device, STICK, PLATE, debug, samples);
}

float* hsa_eigenvalues2(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evals2_symmetric<float>(tensors, n);

    return tira::cuda::evals2_symmetric<float>(tensors, n, device);
}

float* hsa_eigenvectors2polar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evecs2polar_symmetric(tensors, evals, n);

    return tira::cuda::evecs2polar_symmetric<float>(tensors, evals, n, device);
}

glm::mat2* hsa_gaussian2(const glm::mat2* source, const unsigned int width, const unsigned int height, const float sigma,
    unsigned int& out_width, unsigned int& out_height, const int deviceID = 0) {
    if (deviceID < 0) {
        unsigned kernel_size = (unsigned)(6 * sigma);
        float* gauss = tira::cpu::kernel_gaussian<float>(kernel_size, 0, sigma, 1);

        glm::mat2* dest_x = tira::cpu::convolve2<glm::mat2, float>(source, width, height, gauss, kernel_size, 1, out_width, out_height);
        glm::mat2* dest_xy = tira::cpu::convolve2<glm::mat2, float>(dest_x, out_width, out_height, gauss, 1, kernel_size, out_width, out_height);
        free(gauss);
        free(dest_x);
        return dest_xy;
    }

    cudaSetDevice(deviceID);
    glm::mat2* dest = tira::cuda::gaussian_filter2d<glm::mat2>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}