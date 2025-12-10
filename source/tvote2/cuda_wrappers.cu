#include<vector>
#include<string>
#include <cuda_runtime.h>

#include<tira/functions/eigen.h>
#include <tira/functions/filter.h>
#include <tira/functions/tensorvote.h>
#include <tira/filter.cuh>

void InitializeCuda(int& device_id, std::vector<std::string>& device_names) {
    int num_devices;
    cudaError_t error = cudaGetDeviceCount(&num_devices);    // get the number of CUDA devices

    // fill the main structure with a list of the names of every CUDA device
    for (int di = 0; di < num_devices; di++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, di);
        device_names.emplace_back(prop.name);
    }

    if (error != cudaSuccess) {
        // If the device count function returns an error, there are no devices
        device_id = -1;                    
    }
}

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

    if (deviceID < 0) 
        return tira::cpu::gaussian_convolve2<glm::mat2>(source, width, height, sigma, out_width, out_height);

    cudaSetDevice(deviceID);
    glm::mat2* dest = tira::cuda::gaussian_convolve2<glm::mat2>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}