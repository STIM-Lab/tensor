#include <glm/glm.hpp>
#include <tira/cuda/cudaGaussianFilter.cuh>

#include <tira/eigen.cuh>
#include <tira/filter.h>




float* GaussianBlur2D(float* source, unsigned int width, unsigned int height, float sigma,
    unsigned int& out_width, unsigned int& out_height, int deviceID = 0) {

    if (deviceID < 0) {
        unsigned kernel_size = (unsigned)(6 * sigma);
        float* gauss = tira::cpu::kernel_gaussian<float>(kernel_size, 0, sigma, 1);

        float* dest_x = tira::cpu::convolve2<float>(source, width, height, gauss, kernel_size, 1, out_width, out_height);
        float* dest_xy = tira::cpu::convolve2<float>(dest_x, out_width, out_height, gauss, 1, kernel_size, out_width, out_height);
        free(gauss);
        free(dest_x);
        return dest_xy;
    }

    cudaSetDevice(deviceID);
    float* dest = tira::cuda::GaussianFilter2D<float>(source, width, height, sigma, sigma, out_width, out_height);

    return dest;
}

glm::mat3* cudaGaussianBlur3D(glm::mat3* source, unsigned int width, unsigned int height, unsigned int depth, 
    float sigma_w, float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height,
    unsigned int& out_depth, int deviceID = 0) {

    cudaSetDevice(deviceID);
    glm::mat3* dest = tira::cuda::GaussianFilter3D<glm::mat3>(source, width, height, depth, sigma_w, sigma_h, sigma_d,
        out_width, out_height, out_depth);

    return dest;
}

float* cudaGaussianBlur3D(float* source, unsigned int width, unsigned int height, unsigned int depth, float sigma_w,
    float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height, 
    unsigned int& out_depth, int deviceID = 0) {

    cudaSetDevice(deviceID);
    float* dest = tira::cuda::GaussianFilter3D<float>(source, width, height, depth, sigma_w, sigma_h, sigma_d, out_width, out_height, out_depth);

    return dest;
}



float* cudaEigenvalues3(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::eigenvalues3_symmetric<float>(tensors, n);

    return tira::cuda::eigenvalues3_symmetric<float>(tensors, n);
}

float* EigenVectors2DPolar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::eigenvectors2polar_symmetric(tensors, evals, n);

    return tira::cuda::eigenvectors2polar_symmetric<float>(tensors, evals, n, device);
}

float* cudaEigenvectors3DPolar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evecs3spherical_symmetric(tensors, evals, n);

    return tira::cuda::evecs3spherical_symmetric<float>(tensors, evals, n, device);
}

void cudaEigendecomposition3D(float* tensors, float*& evals, float*& evecs, unsigned int n, int device) {
    if (device < 0) {
        evals = tira::cpu::eigenvalues3_symmetric(tensors, n);
        evecs = tira::cpu::eigenvectors3spherical_symmetric(tensors, evals, n);
    }

    float* gpu_tensors;
    HANDLE_ERROR(cudaMalloc(&gpu_tensors, n * sizeof(float) * 9));
    HANDLE_ERROR(cudaMemcpy(gpu_tensors, tensors, n * sizeof(float) * 9, cudaMemcpyHostToDevice));


    float* gpu_evals = tira::cuda::eigenvalues3_symmetric(gpu_tensors, n);
    evals = (float*)malloc(n * sizeof(float) * 3);
    HANDLE_ERROR(cudaMemcpy(evals, gpu_evals, n * sizeof(float) * 3, cudaMemcpyDeviceToHost));


    float* gpu_evecs = tira::cuda::eigenvectors3spherical_symmetric(gpu_tensors, gpu_evals, n);
    evecs = (float*)malloc(n * sizeof(float) * 4);    
    HANDLE_ERROR(cudaMemcpy(evecs, gpu_evecs, n * sizeof(float) * 4, cudaMemcpyDeviceToHost));
}

void cudaEigenvalue0(float* tensors, unsigned int n, float* evals, int device) {
    float* both_evals = tira::cuda::eigenvalues2<float>(tensors, n, device);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i];
    }
    free(both_evals);
}

void cudaEigenvalue1(float* tensors, unsigned int n, float* evals, int device) {
    float* both_evals = tira::cuda::evals2_symmetric<float>(tensors, n, device);
    for (size_t i = 0; i < n; i++) {
        evals[i] = both_evals[2 * i + 1];
    }
    free(both_evals);
}