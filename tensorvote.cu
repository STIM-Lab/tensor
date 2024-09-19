
#include <glm/glm.hpp>
#define GLM_FORCE_CUDA

#include "tensorvote.cuh"

#include <iostream>
#include <numbers>
#include <chrono>
#include <driver_types.h>
#include <cuda_runtime_api.h>

extern float t_voting;
extern float t_device2host;
extern float t_host2device;
extern float t_eigendecomposition;
extern float t_total;
extern float t_devicealloc;
extern float t_devicefree;
extern float t_deviceprops;


static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line << std::endl;
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// small then large
__host__ __device__ glm::vec2 Eigenvalues2D(glm::mat2 T) {
    float d = T[0][0];
    float e = T[0][1];
    float f = e;
    float g = T[1][1];

    float dpg = d + g;
    float disc = sqrt((4 * e * f) + pow(d - g, 2));
    float a = (dpg + disc) / 2.0f;
    float b = (dpg - disc) / 2.0f;
    float min = a < b ? a : b;
    float max = a > b ? a : b;
    glm::vec2 out(min, max);
    return out;
}

// small then large
__host__ __device__ glm::vec2 Eigenvector2D(glm::mat2 T, glm::vec2 lambdas, unsigned int index) {
    float d = T[0][0];
    float e = T[0][1];
    //float f = e;
    float g = T[1][1];

    if (e != 0) {
        return glm::normalize(glm::vec2(1.0, (lambdas[index] - d) / e));
    }
    else if (g == 0) {
        return glm::vec2(1.0, 0.0);
    }
    else {
        return glm::vec2(0.0, 1.0);
    }
}

__host__ void cpuEigendecomposition(float *input_field, float *eigenvectors, float *eigenvalues, unsigned int sx, unsigned int sy) {

    unsigned int i;
    for (unsigned int yi = 0; yi < sy; yi++) { // for each tensor in the field
        for (unsigned int xi = 0; xi < sx; xi++) {
            i = (yi * sx + xi); // calculate a 1D index into the 2D image

            unsigned int ti = i * 4;                                // update the tensor index (each tensor is 4 elements)
                                                                    // store the tensor as a matrix to make this more readable
            glm::mat2 T(input_field[ti + 0],
                input_field[ti + 1], 
                input_field[ti + 2], 
                input_field[ti + 3]);

            glm::vec2 evals = Eigenvalues2D(T);                     // calculate the eigenvalues
            glm::vec2 evec = Eigenvector2D(T, evals, 1);              // calculate the largest (1) eigenvector

            unsigned int vi = i * 2;                                // update the vector/value index (each is 2 elements)
            eigenvectors[vi + 0] = evec[0];                         // save the eigenvectors to the output array
            eigenvectors[vi + 1] = evec[1];

            
            eigenvalues[vi + 0] = evals[0];                         // save the eigenvalues to the output array
            eigenvalues[vi + 1] = evals[1];
        }
    }
}

__host__ __device__ float StickDecay(float cos_theta, float length, float sigma) {
    float c = exp(-(length * length) / (sigma * sigma));
    float radial = 1 - (cos_theta * cos_theta);
    float D = c * radial;
    return D;
}

__host__ __device__ float PlateDecay(float length, float sigma) {
    float c = 3.1415926535897932384626433832795028841971693993751058209749445923078164062
        * exp(-(length * length) / (sigma * sigma)) / 2.0f;
    //float radial = 1 - (cos_theta * cos_theta);
    //float D = c * radial;
    return c;
}

/// <summary>
/// Calculate the stick vote for the relative position (u, v) given the voter eigenvales and eigenvectors
/// </summary>
/// <param name="u">u coordinate for the relative position of the receiver</param>
/// <param name="v">v coordinate for the relative position of the receiver</param>
/// <param name="sigma">decay value (standard deviation)</param>
/// <param name="eigenvectors">array containing the largest eigenvector</param>
/// <returns></returns>
__host__ __device__  VoteContribution StickVote(float u, float v, float sigma, float* eigenvectors) {

    glm::vec2 ev(eigenvectors[0], eigenvectors[1]);         // get the eigenvector
    float length = sqrt(u * u + v * v);                     // calculate the distance between voter and votee

    glm::vec2 uv_norm = glm::vec2(u, v);                    // normalize the direction vector
    if (length != 0.0) {                                    // handle normalization if length is zero
        uv_norm /= length;
    }

    float eTv = ev[0] * uv_norm[0] + ev[1] * uv_norm[1];    // calculate the dot product between the eigenvector and direction
    float radius;
    if (eTv == 0.0)                                         // handle the radius if eTv is zero
        radius = 0.0;
    else
        radius = length / (2 * eTv);
    float d = StickDecay(eTv, length, sigma);

    float tvx, tvy;
    if (radius == 0.0) {
        tvx = ev[0];
        tvy = ev[1];
    }
    else {
        tvx = (radius * ev[0] - length * uv_norm[0]) / radius;
        tvy = (radius * ev[1] - length * uv_norm[1]) / radius;
    }

    glm::mat2 TV;
    TV[0][0] = tvx * tvx;
    TV[1][1] = tvy * tvy;
    TV[0][1] = TV[1][0] = tvx * tvy;
    VoteContribution R;
    R.votes = TV;
    R.decay = d;
    return R;
}

__host__ __device__  VoteContribution PlateVote(float u, float v, float sigma) {

    //glm::vec2 ev(eigenvectors[0], eigenvectors[1]);         // get the eigenvector
    float length = sqrt(u * u + v * v);                     // calculate the distance between voter and votee

    VoteContribution R;
    R.decay = PlateDecay(length, sigma);

    glm::mat2 TV;
    TV[0][0] = 1.0f;                                // initialize the receiver vote to I
    TV[1][1] = 1.0f;
    TV[0][1] = TV[1][0] = 0.0f;
    
    if (length >= 0) {
        float phi = atan2(v, u);
        TV[0][0] = TV[0][0] - 0.25f * (cos(2.0f * phi) + 2);
        TV[1][1] = TV[1][1] - 0.25f * (2 - cos(2.0f * phi));
        TV[0][1] = TV[0][1] - 0.25f * (sin(2.0f * phi));
        TV[1][0] = TV[0][1];
    }

    R.votes = TV;
    return R;
}


/// @brief The kernel to perform the tensor voting on the GPU
/// @param data Input tensor field
/// @param VT Output tensor field
/// @param eigenvalues Eigenvalues of the input tensor field
/// @param eigenvectors Eigenvalues of the input tensor field
/// @param sigma Sigma value for the decay function
/// @param w Window size
/// @param width Tensor field width in pixels
/// @param height Tensor field height in pixels
__global__ void kernelVote(float* VT, float* L, float* V, float sigma, int w, int s0, int s1, bool PLATE) {
    int x0 = blockDim.x * blockIdx.x + threadIdx.x;                                       // get the x and y image coordinates for the current thread
    int x1 = blockDim.y * blockIdx.y + threadIdx.y;

    if (x0 >= s0 || x1 >= s1)                                                          // if not within bounds of image, return
        return;

    glm::mat2 Votee(0.0f);

    int hw = w / 2;
    int r0, r1;
    for (int v = -hw; v < hw; v++) {                    // for each pixel in the window
        r0 = x0 + v;
        if (r0 >= 0 && r0 < s0) {
            for (int u = -hw; u < hw; u++) {

                r1 = x1 + u;
                if (r1 >= 0 && r1 < s1) {
                    // calculate the contribution of (u,v) to (x,y)   
                    VoteContribution vote = StickVote(
                        u,
                        v,
                        sigma,
                        //&L[(yr * sx + xr) * 2],
                        &V[(r0 * s1 + r1) * 2]
                    );
                    float scale = L[(r0 * s1 + r1) * 2 + 1] - L[(r0 * s1 + r1) * 2 + 0];
                    Votee = Votee + scale * vote.votes * vote.decay;

                    if (PLATE) {
                        vote = PlateVote(u, v, sigma);
                        scale = L[(r0 * s1 + r1) * 2 + 0];
                        Votee = Votee + scale * vote.votes * vote.decay;
                    }
                }
            }
        }
    }
    
    VT[4 * (x0 * s1 + x1) + 0] = Votee[0][0];
    VT[4 * (x0 * s1 + x1) + 1] = Votee[1][0];
    VT[4 * (x0 * s1 + x1) + 2] = Votee[0][1];
    VT[4 * (x0 * s1 + x1) + 3] = Votee[1][1];
}

void cudaVote2D(float* input_field, float* output_field, unsigned int s0, unsigned int s1, float sigma, unsigned int w, unsigned int device, bool PLATE, bool time) {
    
    auto start = std::chrono::high_resolution_clock::now();
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, device));
    auto end = std::chrono::high_resolution_clock::now();
    t_deviceprops = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int tensorFieldSize = 4 * s0 * s1;
    float* V = new float[s0 * s1 * 2];                              // allocate space for the eigenvectors
    float* L = new float[s0 * s1 * 2];                              // allocate space for the eigenvalues


    start = std::chrono::high_resolution_clock::now();
    cpuEigendecomposition(input_field, &V[0], &L[0], s0, s1);   // calculate the eigendecomposition of the entire field
    end = std::chrono::high_resolution_clock::now();
    t_eigendecomposition = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    // Declare GPU arrays
    float* gpuOutputField;
    float* gpuV;
    float* gpuL;

    
    // Allocate GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc(&gpuOutputField, tensorFieldSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuV, s0 * s1 * 2 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuL, s0 * s1 * 2 * sizeof(float)));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    t_devicealloc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  
    start = std::chrono::high_resolution_clock::now();
    // Copy input arrays
    HANDLE_ERROR(cudaMemcpy(gpuV, V, s0 * s1 * 2 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuL, L, s0 * s1 * 2 * sizeof(float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();

    t_host2device = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Specify the CUDA block and grid dimensions
    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    dim3 blocks(s0 / threads.x + 1, s1 / threads.y + 1);

    start = std::chrono::high_resolution_clock::now();
    kernelVote << <blocks, threads >> > (gpuOutputField, gpuL, gpuV, sigma, w, s0, s1, PLATE);              // call the CUDA kernel for voting
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    t_voting = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    // Copy the final result back from the GPU
    HANDLE_ERROR(cudaMemcpy(output_field, gpuOutputField, tensorFieldSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    t_device2host = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Free all of the GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaFree(gpuOutputField));
    HANDLE_ERROR(cudaFree(gpuV));
    HANDLE_ERROR(cudaFree(gpuL));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    t_devicefree = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

