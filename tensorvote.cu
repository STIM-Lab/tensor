
#include <glm/glm.hpp>
#define GLM_FORCE_CUDA

#define PI 3.14159265358979323846

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

float* cudaEigenvalues2(float* tensors, unsigned int n, int device);
float* cudaEigenvectors2DPolar(float* tensors, float* evals, unsigned int n, int device);
float* cudaEigenvalues3(float* tensors, unsigned int n, int device);
float* cudaEigenvectors3DPolar(float* tensors, float* evals, unsigned int n, int device);


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
    float min = abs(a) < abs(b) ? a : b;
    float max = abs(a) > abs(b) ? a : b;
    glm::vec2 out(min, max);
    return out;
}

// small then large
__host__ __device__ glm::vec2 Eigenvector2D(glm::mat2 T, float lambda) {
/*
    //[ d    e ]
    //[ e    g ]
    float d = T[0][0];
    float e = T[0][1];
    float g = T[1][1];

    if (e != 0) {
        return glm::normalize(glm::vec2(1.0, (lambda - d) / e));
    }
    else if (g == 0) {
        if (index == 0) return glm::vec2(0.0, 1.0);
        if (index == 1) return glm::vec2(1.0, 0.0);
    }
    else {
        if (index == 0) return glm::vec2(0.0, 1.0);
        if (index == 1) return glm::vec2(1.0, 0.0);
    }
    */
    // [ a   b ]
    // [ b   d ]
    float a = T[0][0];
    float b = T[0][1];
    //float c = b;
    float d = T[1][1];

    if (b != 0) {
        return glm::normalize(glm::vec2(lambda - d, b));
    }
    else if (lambda == 0) {
        if (a < d) return glm::vec2(1.0, 0.0);
        else return glm::vec2(0.0, 1.0);
    }
    else {
        if (a < d) return glm::vec2(0.0, 1.0);
        else return glm::vec2(1.0, 0.0);
    }
}

__host__ void cpuEigendecomposition2D(float *input_field, float *eigenvectors, float *eigenvalues, unsigned int sx, unsigned int sy) {

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
            glm::vec2 evec(Eigenvector2D(T, evals[1]));              // calculate the largest (1) eigenvector

            unsigned int vi = i * 2;                                // update the vector/value index (each is 2 elements)
            eigenvectors[vi + 0] = evec[0];                         // save the eigenvectors to the output array
            eigenvectors[vi + 1] = evec[1];

            
            eigenvalues[vi + 0] = evals[0];                         // save the eigenvalues to the output array
            eigenvalues[vi + 1] = evals[1];
        }
    }
}

__host__ __device__ float decay(float term, float length, float sigma, unsigned int power = 1) {
    float c = exp(-(length * length) / (sigma * sigma));

    float tp = term;
    for (unsigned int pi = 1; pi < power; pi++)
        tp *= term;

    return c * tp;
}

__host__ __device__ float PlateDecay2D(float length, float sigma) {
    float c = PI * exp(-(length * length) / (sigma * sigma)) / 2.0f;

    return c;
}

__host__ __device__ double factorial(unsigned int n) {
    double fac = 1;
    for(unsigned int i=1; i<=n; i++)
        fac *= i;
    return fac;
}

__host__ __device__ double sticknorm2D(double sigma1, double sigma2, unsigned int p) {
    double num = PI * factorial(2 * p);
    double ex = std::pow(2, 2 * p);
    double facp = factorial(p);
    double trig_int = num / (ex * facp * facp);
    return trig_int * (sigma1 * sigma1 + sigma2 * sigma2);
}

/// <summary>
/// Calculate the stick vote for the relative position (u, v) given the voter eigenvales and eigenvectors
/// </summary>
/// <param name="u">u coordinate for the relative position of the receiver</param>
/// <param name="v">v coordinate for the relative position of the receiver</param>
/// <param name="sigma">decay value (standard deviation)</param>
/// <param name="eigenvectors">array containing the largest eigenvector</param>
/// <returns></returns>
__host__ __device__  VoteContribution2D StickVote2D(float u, float v, float sigma1, float sigma2, float* eigenvectors, unsigned int power) {

    glm::vec2 q(eigenvectors[0], eigenvectors[1]);          // get the eigenvector

    glm::vec2 d = glm::vec2(u, v);                          // normalize the direction vector
    float l = glm::length(d);                               // calculate ell (distance between voter/votee)
    //if (l == 0) d = q;                                      // assumes the voter doesn't contribute to the voter
    if(l == 0) d = glm::vec2(0, 0);                     // assumes that the voter DOES contribute to itself
    else d = glm::normalize(d);

    float qTd = glm::dot(q, d);
    
    float eta1 = 0;
    float eta2 = 0;
    if (sigma1 > 0)
        eta1 = decay(1 - qTd * qTd, l, sigma1, power);                       // calculate the decay function
    if (sigma2 > 0)
        eta2 = decay(qTd * qTd, l, sigma2, power);

    glm::mat2 R = glm::mat2(1.0f) - 2.0f * glm::outerProduct(d, d);
    glm::vec2 Rq = R * q;
    glm::mat2 RqRq = glm::outerProduct(Rq, Rq);


    VoteContribution2D S;
    S.votes = RqRq;
    S.decay = eta1 + eta2;
    return S;
}

__host__ __device__  glm::mat2 PlateVote2D(float u, float v, float sigma1, float sigma2) {

    float length = sqrt(u * u + v * v);                     // calculate the distance between voter and votee
    float l2 = length * length;
    float s12 = sigma1 * sigma1;
    float s22 = sigma2 * sigma2;
    float e1 = 0;
    if (sigma1 > 0)
        e1 = std::exp(-l2 / s12);
    float e2 = 0;
    if (sigma2 > 0)
        e2 = std::exp(-l2 / s22);

    float alpha = std::atan2(v, u);
    float two_a = 2 * alpha;
    float cos_2a = std::cos(two_a);
    float sin_2a = std::sin(two_a);
    glm::mat2 M;
    M[0][0] = cos_2a + 2;
    M[1][0] = sin_2a;
    M[0][1] = sin_2a;
    M[1][1] = 2 - cos_2a;

    glm::mat2 I(1.0f);

    float c = 1.0f / (PI * (s12 + s22));

    return c * (e1 * (I - 0.25f * M) + e2 * (0.25f * M));
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
__global__ void kernelStickVote2D(float* VT, float* L, float* V, float sigma1, float sigma2, unsigned int power, float norm, int w, int s0, int s1) {
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
                    float Vcart[2];
                    Vcart[0] = std::cos(V[(r0 * s1 + r1) * 2 + 1]);
                    Vcart[1] = std::sin(V[(r0 * s1 + r1) * 2 + 1]);
                    VoteContribution2D vote = StickVote2D(u, v, sigma1, sigma2, Vcart, power);
                    float l0 = L[(r0 * s1 + r1) * 2 + 0];
                    float l1 = L[(r0 * s1 + r1) * 2 + 1];
                    float scale = std::abs(l1) - std::abs(l0);
                    if(l1 < 0) scale = scale * (-1);
                    Votee = Votee + scale * vote.votes * norm * vote.decay;
                }
            }
        }
    }
    
    VT[4 * (x0 * s1 + x1) + 0] += Votee[0][0];
    VT[4 * (x0 * s1 + x1) + 1] += Votee[1][0];
    VT[4 * (x0 * s1 + x1) + 2] += Votee[0][1];
    VT[4 * (x0 * s1 + x1) + 3] += Votee[1][1];
}

__global__ void kernelPlateVote2D(float* VT, float* L, float* V, float sigma1, float sigma2, unsigned int power, float norm, int w, int s0, int s1) {
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
                    float l0 = L[(r0 * s1 + r1) * 2 + 0];
                    Votee = Votee + l0 * PlateVote2D(u, v, sigma1, sigma2);
                }
            }
        }
    }

    VT[4 * (x0 * s1 + x1) + 0] += Votee[0][0];
    VT[4 * (x0 * s1 + x1) + 1] += Votee[1][0];
    VT[4 * (x0 * s1 + x1) + 2] += Votee[0][1];
    VT[4 * (x0 * s1 + x1) + 3] += Votee[1][1];
}

void cudaVote2D(float* input_field, float* output_field, unsigned int s0, unsigned int s1, float sigma, float sigma2, 
    unsigned int w, unsigned int power, unsigned int device, bool STICK, bool PLATE, bool debug) {
    
    auto start = std::chrono::high_resolution_clock::now();
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, device));
    auto end = std::chrono::high_resolution_clock::now();
    float t_deviceprops = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int tensorFieldSize = 4 * s0 * s1;

    start = std::chrono::high_resolution_clock::now();
    float* L = cudaEigenvalues2(input_field, s0 * s1, device);
    float* V = cudaEigenvectors2DPolar(input_field, L, s0 * s1, device);
    end = std::chrono::high_resolution_clock::now();
    float t_eigendecomposition = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Declare GPU arrays
    float* gpuOutputField;
    float* gpuV;
    float* gpuL;

    
    // Allocate GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc(&gpuOutputField, tensorFieldSize * sizeof(float)));
    HANDLE_ERROR(cudaMemset(gpuOutputField, 0, tensorFieldSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuV, s0 * s1 * 2 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gpuL, s0 * s1 * 2 * sizeof(float)));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_devicealloc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  
    start = std::chrono::high_resolution_clock::now();
    // Copy input arrays
    HANDLE_ERROR(cudaMemcpy(gpuV, V, s0 * s1 * 2 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuL, L, s0 * s1 * 2 * sizeof(float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();

    float t_host2device = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Specify the CUDA block and grid dimensions
    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    dim3 blocks(s0 / threads.x + 1, s1 / threads.y + 1);

    float sn = 1.0 / sticknorm2D(sigma, sigma2, power);

    if (debug)
        std::cout << "Stick Area: " << sn << std::endl;

    start = std::chrono::high_resolution_clock::now();
    if(STICK)
        kernelStickVote2D << <blocks, threads >> > (gpuOutputField, gpuL, gpuV, sigma, sigma2, power, sn, w, s0, s1);              // call the CUDA kernel for voting
    if (PLATE)
        kernelPlateVote2D << <blocks, threads >> > (gpuOutputField, gpuL, gpuV, sigma, sigma2, power, sn, w, s0, s1);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_voting = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    // Copy the final result back from the GPU
    HANDLE_ERROR(cudaMemcpy(output_field, gpuOutputField, tensorFieldSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_device2host = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Free all of the GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaFree(gpuOutputField));
    HANDLE_ERROR(cudaFree(gpuV));
    HANDLE_ERROR(cudaFree(gpuL));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_devicefree = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (debug) {
        std::cout << "Eigendecomposition:  " << t_eigendecomposition << " ms" << std::endl;
        std::cout << "Voting: " << t_voting << " ms" << std::endl;
        std::cout << "cudaMemcpy (H->D):  " << t_host2device << " ms" << std::endl;
        std::cout << "cudaMemcpy (D->H):  " << t_device2host << " ms" << std::endl;
        std::cout << "cudaMalloc: " << t_devicealloc << " ms" << std::endl;
        std::cout << "cudaFree: " << t_devicefree << " ms" << std::endl;
        std::cout << "cudaDeviceProps: " << t_deviceprops << " ms" << std::endl;
    }
}



/// <summary>
/// Calculate the stick vote for the relative position (u, v) given the voter eigenvales and eigenvectors
/// </summary>
/// <param name="u">u coordinate for the relative position of the receiver</param>
/// <param name="v">v coordinate for the relative position of the receiver</param>
/// <param name="sigma">decay value (standard deviation)</param>
/// <param name="eigenvectors">array containing the largest eigenvector</param>
/// <returns></returns>
__host__ __device__  VoteContribution3D StickVote3D(float u, float v, float w, float sigma1, float sigma2, float* eigenvectors, unsigned int power) {

    glm::vec3 q(eigenvectors[0], eigenvectors[1], eigenvectors[2]);             // get the eigenvector

    glm::vec3 d = glm::vec3(u, v, w);                                           // normalize the direction vector
    float l = glm::length(d);                                                   // calculate ell (distance between voter/votee)
    if (l == 0) d = glm::vec3(0, 0, 0);                                         // assumes that the voter DOES contribute to itself
    else d = glm::normalize(d);

    float qTd = glm::dot(q, d);

    float term1 = 0;
    float term2 = 0;
    if (sigma1 > 0)
        term1 = decay(1 - qTd * qTd, l, sigma1, power);                          // calculate the decay function
    if (sigma2 > 0)
        term2 = decay(qTd * qTd, l, sigma2, power);

    glm::mat3 R = glm::mat3(1.0f) - 2.0f * glm::outerProduct(d, d);
    glm::vec3 Rq = R * q;
    glm::mat3 RqRq = glm::outerProduct(Rq, Rq);

    VoteContribution3D S;
    S.votes = RqRq;
    S.decay = term1 + term2;
    return S;
}

/// @brief The kernel to perform the 3D stick tensor voting on the GPU
/// @param VT Output tensor field
/// @param L Eigenvalues of the input tensor field
/// @param V Eigenvalues of the input tensor field
/// @param sigma1 Sigma value for the decay function
/// @param sigma2 Sigma value for the decay function
/// @param power Power used to refine the vote field
/// @param norm The decay normalization term
/// @param w Window size
/// @param s0 Tensor field width in pixels
/// @param s1 Tensor field height in pixels
/// @param s2 Tensor field depth in pixels
__global__ void kernelStickVote3D(float* VT, float* L, float* V, float sigma, float sigma2, unsigned int power, float norm, int w, int s0, int s1, int s2) {
    int x0 = blockDim.x * blockIdx.x + threadIdx.x;                                     // get the x, y, and z volume coordinates for the current thread
    int x1 = blockDim.y * blockIdx.y + threadIdx.y;
    int x2 = blockDim.z * blockIdx.z + threadIdx.z;

    if (x0 >= s0 || x1 >= s1 || x2 >= s2)                                               // if not within bounds of image, return
        return;

    glm::mat3 Votee(0.0f);
    float scale;

    int hw = int(w / 2);
    int r0, r1, r2;
    for (int w0 = -hw; w0 <= hw; w0++) {                                                 // for each pixel in the window
        r0 = x0 + w0;
        if (r0 >= 0 && r0 < s0) {
            for (int w1 = -hw; w1 <= hw; w1++) {
                r1 = x1 + w1;
                if (r1 >= 0 && r1 < s1) {
                    for (int w2 = -hw; w2 <= hw; w2++) {
                        r2 = x2 + w2;
                        if (r2 >= 0 && r2 < s2) {
                            // calculate the contribution of (w0, w1, w2) to (x,y,z)
                            float Vcart[3];
                            float theta = V[(r0 * s1 * s2 + r1 * s2 + r2) * 4 + 2];
                            float phi = V[(r0 * s1 * s2 + r1 * s2 + r2) * 4 + 3];

                            Vcart[0] = sinf(theta) * cosf(phi);
                            Vcart[1] = sinf(theta) * sinf(phi);
                            Vcart[2] = cosf(theta);
                            VoteContribution3D vote = StickVote3D(w2, w1, w0, sigma, sigma2, Vcart, power);

                            //float l0 = L[(r0 * s1 * s2 + r1 * s2 + r2) * 3 + 0];
                            float l1 = L[(r0 * s1 * s2 + r1 * s2 + r2) * 3 + 1];
                            float l2 = L[(r0 * s1 * s2 + r1 * s2 + r2) * 3 + 2];

                            scale = fabsf(l2) - fabsf(l1);
                            if (l2 < 0.0f) scale *= -1;

                            Votee = Votee + scale * vote.votes * vote.decay;
                        }
                    }
                }
            }
        }
    }

    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 0] += Votee[0][0];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 1] += Votee[0][1];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 2] += Votee[0][2];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 3] += Votee[1][0];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 4] += Votee[1][1];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 5] += Votee[1][2];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 6] += Votee[2][0];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 7] += Votee[2][1];
    VT[(x0 * s1 * s2 + x1 * s2 + x2) * 9 + 8] += Votee[2][2];
}

void cudaVote3D(float* input_field, float* output_field, unsigned int s0, unsigned int s1, unsigned int s2, float sigma, float sigma2,
    unsigned int w, unsigned int power, unsigned int device, bool STICK, bool PLATE, bool debug) {

    // Declare GPU arrays
    float* gpuOutputField;
    float* gpuV;
    float* gpuL;

    auto start = std::chrono::high_resolution_clock::now();
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, device));
    auto end = std::chrono::high_resolution_clock::now();
    float t_deviceprops = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    size_t tensorField_bytes =  sizeof(float) * 9 * s0 * s1 * s2;
    size_t evals_bytes =        sizeof(float) * 3 * s0 * s1 * s2;
    size_t evecs_bytes =        sizeof(float) * 4 * s0 * s1 * s2;

    start = std::chrono::high_resolution_clock::now();
    float* L = cudaEigenvalues3(input_field, s0 * s1 * s2, device);
    float* V = cudaEigenvectors3DPolar(input_field, L, s0 * s1 * s2, device);
    end = std::chrono::high_resolution_clock::now();
    float t_eigendecomposition = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Allocate GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc(&gpuOutputField, tensorField_bytes));
    HANDLE_ERROR(cudaMemset(gpuOutputField, 0, tensorField_bytes));
    HANDLE_ERROR(cudaMalloc(&gpuV, evecs_bytes));
    HANDLE_ERROR(cudaMalloc(&gpuL, evals_bytes));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_devicealloc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    // Copy input arrays
    HANDLE_ERROR(cudaMemcpy(gpuV, V, evecs_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuL, L, evals_bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_host2device = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Specify the CUDA block and grid dimensions
    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim, blockDim);
    dim3 blocks(s0 / threads.x + 1, s1 / threads.y + 1, s2 / threads.z + 1);

    float sn = 1.0f;          //1.0 / sticknorm2D(sigma, sigma2, power);      The function hasn't been implemented yet...

    if (debug)
        std::cout << "Stick Area: " << sn << std::endl;

    start = std::chrono::high_resolution_clock::now();
    if (STICK)
        kernelStickVote3D << <blocks, threads >> > (gpuOutputField, gpuL, gpuV, sigma, sigma2, power, sn, w, s0, s1, s2);              // call the CUDA kernel for voting
    if (PLATE)
        std::cout << "Not yet implemented." << std::endl;
        //kernelPlateVote2D << <blocks, threads >> > (gpuOutputField, gpuL, gpuV, sigma, sigma2, power, sn, w, s0, s1);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_voting = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    // Copy the final result back from the GPU
    HANDLE_ERROR(cudaMemcpy(output_field, gpuOutputField, tensorField_bytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_device2host = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Free all of the GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaFree(gpuOutputField));
    HANDLE_ERROR(cudaFree(gpuV));
    HANDLE_ERROR(cudaFree(gpuL));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float t_devicefree = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (debug) {
        std::cout << "Eigendecomposition:  " << t_eigendecomposition << " ms" << std::endl;
        std::cout << "Voting: " << t_voting << " ms" << std::endl;
        std::cout << "cudaMemcpy (H->D):  " << t_host2device << " ms" << std::endl;
        std::cout << "cudaMemcpy (D->H):  " << t_device2host << " ms" << std::endl;
        std::cout << "cudaMalloc: " << t_devicealloc << " ms" << std::endl;
        std::cout << "cudaFree: " << t_devicefree << " ms" << std::endl;
        std::cout << "cudaDeviceProps: " << t_deviceprops << " ms" << std::endl;
    }
}