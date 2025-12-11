#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <tira/functions/eigen.h>
#include <tira/functions/filter.h>


// ----- Gaussian blur wrappers ----

glm::mat2* GaussianBlur2D(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
    unsigned int& out_width, unsigned int& out_height, int deviceID) {

    if (deviceID < 0) {
        return tira::cpu::gaussian_convolve2<glm::mat2>(source, width, height, sigma, out_width, out_height);
    }

    cudaSetDevice(deviceID);
    return tira::cuda::gaussian_convolve2<glm::mat2>(source, width, height, sigma, sigma, out_width, out_height);
}

float* GaussianBlur2D(float* source, unsigned int width, unsigned int height, float sigma,
    unsigned int& out_width, unsigned int& out_height, int deviceID) {

    if (deviceID < 0) {
        return tira::cpu::gaussian_convolve2<float>(source, width, height, sigma, out_width, out_height);
    }

    cudaSetDevice(deviceID);
    return tira::cuda::gaussian_convolve2<float>(source, width, height, sigma, sigma, out_width, out_height);
}

glm::mat3* GaussianBlur3D(glm::mat3* source, unsigned int width, unsigned int height, unsigned int depth,
    float sigma_w, float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height,
    unsigned int& out_depth, int deviceID) {

    if (deviceID < 0) {
		return tira::cpu::gaussian_convolve3<glm::mat3>(source, width, height, depth, sigma_w, sigma_h, sigma_d,
			out_width, out_height, out_depth);
	}

    cudaSetDevice(deviceID);
	return tira::cuda::gaussian_filter3d<glm::mat3>(source, width, height, depth, sigma_w, sigma_h, sigma_d,
		out_width, out_height, out_depth);
}

float* GaussianBlur3D(float* source, unsigned int width, unsigned int height, unsigned int depth, 
    float sigma_w, float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height, 
    unsigned int& out_depth, int deviceID) {
    
	if (deviceID < 0) {
		return tira::cpu::gaussian_convolve3<float>(source, width, height, depth, sigma_w, sigma_h, sigma_d,
			out_width, out_height, out_depth);
	}

    cudaSetDevice(deviceID);
    return tira::cuda::gaussian_filter3d<float>(source, width, height, depth, sigma_w, sigma_h, sigma_d,
		out_width, out_height, out_depth);
}

// ----- Eigendecomposition wrappers -----

float* EigenValues2(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evals2_symmetric<float>(tensors, n);
    return tira::cuda::evals2_symmetric<float>(tensors, static_cast<size_t>(n), device);
}

float* Eigenvalues3(float* tensors, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evals3_symmetric<float>(tensors, n);

    return tira::cuda::evals3_symmetric<float>(tensors, n, device);
}

static float* EigenVectors2DPolar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evecs2polar_symmetric(tensors, evals, n);

    return tira::cuda::evecs2polar_symmetric<float>(tensors, evals, n, device);
}

static float* Eigenvectors3DPolar(float* tensors, float* evals, unsigned int n, int device) {
    if (device < 0) return tira::cpu::evecs3spherical_symmetric(tensors, evals, n);

    return tira::cuda::evecs3spherical_symmetric<float>(tensors, evals, n, device);
}

static void Eigendecomposition3D(float* tensors, float*& evals, float*& evecs, unsigned int n, int device) {
    if (device < 0) {
        evals = tira::cpu::evals3_symmetric(tensors, n);
        evecs = tira::cpu::evecs3spherical_symmetric(tensors, evals, n);
		return;
    }

    float* gpu_tensors;
    HANDLE_ERROR(cudaMalloc(&gpu_tensors, n * sizeof(float) * 9));
    HANDLE_ERROR(cudaMemcpy(gpu_tensors, tensors, n * sizeof(float) * 9, cudaMemcpyHostToDevice));


    float* gpu_evals = tira::cuda::evals3_symmetric(gpu_tensors, n, device);
    evals = (float*)malloc(n * sizeof(float) * 3);
    HANDLE_ERROR(cudaMemcpy(evals, gpu_evals, n * sizeof(float) * 3, cudaMemcpyDeviceToHost));


    float* gpu_evecs = tira::cuda::evecs3spherical_symmetric(gpu_tensors, gpu_evals, n, device);
    evecs = (float*)malloc(n * sizeof(float) * 4);    
    HANDLE_ERROR(cudaMemcpy(evecs, gpu_evecs, n * sizeof(float) * 4, cudaMemcpyDeviceToHost));
}
