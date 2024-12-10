#include <glm/glm.hpp>
#include <tira/cuda/cudaGaussianFilter.cuh>

#include <tira/math/eigen.cuh>


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

float* cudaEigenvalues2(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::Eigenvalues2D<float>(tensors, n);

    return tira::cuda::Eigenvalues2D<float>(tensors, n, device);
}

float* cudaEigenvalues3(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::Eigenvalues3D<float>(tensors, n);

    return tira::cuda::Eigenvalues3D<float>(tensors, n);
}

float* cudaEigenvectors2DPolar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::Eigenvectors2DPolar(tensors, evals, n);

    return tira::cuda::Eigenvectors2DPolar<float>(tensors, evals, n, device);
}

float* cudaEigenvectors3DPolar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::Eigenvectors3DPolar(tensors, evals, n);

    return tira::cuda::Eigenvectors3DPolar<float>(tensors, evals, n);
}

void cudaEigendecomposition3D(float* tensors, float*& evals, float*& evecs, unsigned int n, int device) {
    if (device < 0) {
        evals = tira::cpu::Eigenvalues3D(tensors, n);
        evecs = tira::cpu::Eigenvectors3DPolar(tensors, evals, n);
    }

    float* gpu_tensors;
    HANDLE_ERROR(cudaMalloc(&gpu_tensors, n * sizeof(float) * 9));
    HANDLE_ERROR(cudaMemcpy(gpu_tensors, tensors, n * sizeof(float) * 9, cudaMemcpyHostToDevice));


    float* gpu_evals = tira::cuda::Eigenvalues3D(gpu_tensors, n);
    evals = (float*)malloc(n * sizeof(float) * 3);
    HANDLE_ERROR(cudaMemcpy(evals, gpu_evals, n * sizeof(float) * 3, cudaMemcpyDeviceToHost));


    float* gpu_evecs = tira::cuda::Eigenvectors3DPolar(gpu_tensors, gpu_evals, n);    
    evecs = (float*)malloc(n * sizeof(float) * 4);    
    HANDLE_ERROR(cudaMemcpy(evecs, gpu_evecs, n * sizeof(float) * 4, cudaMemcpyDeviceToHost));
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