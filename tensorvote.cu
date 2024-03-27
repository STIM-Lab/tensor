
#include <glm/glm.hpp>
#define GLM_FORCE_CUDA

#include "tensorvote.cuh"

#include <iostream>


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

__host__ __device__  void cpuEigendecomposition(float *input_field, float *eigenvectors, float *eigenvalues, unsigned int sx, unsigned int sy) {

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

__host__ __device__ float Decay(float cos_theta, float length, float sigma) {
    float c = exp(-(length * length) / (sigma * sigma));
    float radial = 1 - (cos_theta * cos_theta);
    float D = c * radial;
    return D;
}

__host__ __device__  VoteContribution Saliency(float u, float v, float sigma, float* eigenvalues, float* eigenvectors) {

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
    float d = Decay(eTv, length, sigma);

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

/// @brief The kernel to perform the tensor voting on the GPU
/// @param data Input tensor field
/// @param VT Output tensor field
/// @param eigenvalues Eigenvalues of the input tensor field
/// @param eigenvectors Eigenvalues of the input tensor field
/// @param sigma Sigma value for the decay function
/// @param w Window size
/// @param width Tensor field width in pixels
/// @param height Tensor field height in pixels
__global__ void cudaKernel(float *data, float *VT, float* eigenvalues, float* eigenvectors, float sigma, int w, int width, int height) {
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;                                       // get the x and y image coordinates for the current thread
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= height || x >= width)                                                          // if not within bounds of image, return
        return;

    float vt[4] = {0, 0, 0, 0};                                                             // initialize a tensor to zeros

    for (int u = -w; u < w; u++){                                                           // for each pixel within the window
        for (int v = -w; v < w; v++) {
            int index = ((v + y) * width + x + u);                                          // calculate a 1D index into the tensor field
            if (index < width * height && index >= 0) {                                     // DAVID: This will cause wrap-around artifacts
                glm::mat2 T(
                    data[4 * index + 0],
                    data[4 * index + 1],
                    data[4 * index + 2],
                    data[4 * index + 3]);

                // printf("T: %f %f %f %f eigenvalues: %f %f eigenvectors: (%f, %f) (%f, %f)\n", T[0][0], T[0][1], T[1][0], T[1][1], eigenvalues[2 * index], eigenvalues[2 * index + 1], eigenvectors[index], eigenvectors[index + 1], eigenvectors[index + 2], eigenvectors[index + 3]);

                VoteContribution vc = Saliency(
                    u, 
                    v, 
                    sigma, 
                    &eigenvalues[2 * index], 
                    &eigenvectors[index]
                );                             // calculate the saliency given the tensor at (u,v)
                vt[0] += vc.votes[0][0] * vc.decay;                                         // sum the tensor contribution based on the saliency
                vt[1] += vc.votes[0][1] * vc.decay;
                vt[2] += vc.votes[1][0] * vc.decay;
                vt[3] += vc.votes[1][1] * vc.decay;
            }
        }
    }

    VT[4 * (y * width + x) + 0] = vt[0];
    VT[4 * (y * width + x) + 1] = vt[1];
    VT[4 * (y * width + x) + 2] = vt[2];
    VT[4 * (y * width + x) + 3] = vt[3];

}

/// <summary>
/// 
/// </summary>
/// <param name="input_field">Pointer to the input field in CPU memory</param>
/// <param name="output_field">Pointer to the output field (destination) in CPU memory</param>
/// <param name="sx">Size of the field along the x axis</param>
/// <param name="sy">Size of the field along the y axis</param>
/// <param name="sigma">Standard deviation for the decay function</param>
/// <param name="w">Width of the window to be used</param>
/// <param name="device">ID for the CUDA device</param>
void cudaVote2D(float* input_field, float* output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w, unsigned int device) {
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, device));

    std::cout << "**********CUDA**********" << std::endl;
    //int hw = (int)(w / 2);
    int tensorFieldSize = 4 * sx * sy;
                        
    float* V = new float[sx * sy * 2];                              // allocate space for the eigenvectors
    float* L = new float[sx * sy * 2];                              // allocate space for the eigenvalues
    cpuEigendecomposition(input_field, &V[0], &L[0], sx, sy);

    float* gpuInputField;
    float* gpuOutputField;
    float* gpuV;
    float* gpuL;

    HANDLE_ERROR(cudaMalloc(&gpuInputField, tensorFieldSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuOutputField, tensorFieldSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuV, sx * sy * 2 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuL, sx * sy * 2 * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(gpuInputField, input_field, tensorFieldSize * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuV, V, sx * sy * 2 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuL, L, sx * sy * 2 * sizeof(float), cudaMemcpyHostToDevice));

    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    dim3 blocks(sx / threads.x + 1, sy / threads.y + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    // cudaKernel<<<blocks, threads>>>(gpuInputField, gpuOutputField, &V[0], &L[0], sigma, w, sx, sy);              // call the CUDA kernel for voting
    cudaKernel<<<blocks, threads>>>(gpuInputField, gpuOutputField, gpuV, gpuL, sigma, w, sx, sy);              // call the CUDA kernel for voting

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    HANDLE_ERROR(cudaMemcpy(output_field, gpuOutputField, tensorFieldSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    std::cout << "Elapsed time: " << totalTime / 1000 << " seconds" << std::endl;

    delete V;
    delete L;
}

