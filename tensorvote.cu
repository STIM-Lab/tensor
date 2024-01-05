#include <iostream>
#include <stdio.h>

#include "tensorvote.h"

#include "tensorvote.cuh"

#include <chrono>

#ifdef __CUDACC__
#define __device__
#endif

__host__ __device__ bool NonZeroTensor(glm::mat2 T)
{
    return T[0][0] || T[0][1] || T[1][0] || T[1][1];
}

__host__ __device__ float Decay(float angle, float length, int sigma)
{
    if (length == 0)
        return 1;

    float alpha = acos(abs(cos(M_PI / 2 - angle)));

    // calculate c (see math)
    float c = (-16 * log10f(0.1) * (sigma - 1)) / (pow(M_PI, 2));

    // calculate saliency decay
    float S;
    if (alpha == 0)
        S = length;
    else
        S = (alpha * length) / sin(alpha);

    float kappa = (2 * sin(alpha)) / length;
    float S_kappa = pow(S, 2) + c * pow(kappa, 2);
    float E = -1 * S_kappa / (pow(sigma, 2));
    float d;
    float pi4 = M_PI / 4;
    if (alpha > pi4 || alpha < -pi4)
        d = 0;
    else
        d = exp(E);

    return d;
}

__host__ __device__ TensorAngleCalculation SaliencyTheta(float theta, float u, float v, int sigma = 10)
{
    float theta_cos = cos(theta);
    float theta_sin = sin(theta);

    glm::mat2 Rtheta_r(theta_cos, theta_sin, -theta_sin, theta_cos);
    glm::mat2 Rtheta_l(theta_cos, -theta_sin, theta_sin, theta_cos);

    glm::vec2 p(Rtheta_r[0][0] * u + Rtheta_r[0][1] * v, Rtheta_r[1][0] * u + Rtheta_r[1][1] * v);

    float l = sqrt(pow(p[0], 2) + pow(p[1], 2));

    float phi = atan2(p[1], p[0]);

    float decay = Decay(phi, l, sigma);

    float phi2 = 2 * phi;

    float phi2_cos = cos(phi2);
    float phi2_sin = sin(phi2);

    glm::mat2 Rphi2(phi2_cos, -phi2_sin, phi2_sin, phi2_cos);

    glm::vec2 V_source(Rphi2[0][0], Rphi2[1][0]);

    glm::vec2 V(Rtheta_l[0][0] * V_source[0] + Rtheta_l[0][1] * V_source[1], Rtheta_l[1][0] * V_source[0] + Rtheta_l[1][1] * V_source[1]);
    glm::mat2 outer = glm::outerProduct(V, V);

    TensorAngleCalculation out;

    out.votes = outer;
    out.decay = decay;

    return out;
}

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line << std::endl;
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

tira::image<glm::mat2> voteCPU(tira::image<glm::mat2> T, int sigma)
{
    int w = 6 * sigma / 2;

    int X = T.shape()[1];
    int Y = T.shape()[0];

    tira::image<glm::mat2> VT(X, Y);

    for (int x = 0; x < X; x++)
    {
        for (int y = 0; y < Y; y++)
        {
            for (int u = -w; u < w; u++)
            {
                for (int v = -w; v < w; v++)
                {
                    if (y + v >= 0 && y + v < Y && x + u >= 0 && x + u < X)
                    {
                        VoteContribution vc = Saliency(T(x, y), u, v, sigma);

                        VT(x + u, y + v) += vc.votes * vc.decay;
                    }
                }
            }
        }
    }

    return VT;
}

__global__ void voteGPU(float *data, float *VT, int sigma, int w, int width, int height)
{
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;                                       // get the x and y image coordinates for the current thread
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    
    if (y >= height || x >= width)                                                          // if not within bounds of image, return
        return;

    float vt[4] = {0, 0, 0, 0};                                                             // initialize a tensor to zeros

    for (int u = -w; u < w; u++)                                                            // for each pixel within the window
    {
        for (int v = -w; v < w; v++)
        {
            int index = ((v + y) * width + x + u);                                          // calculate a 1D index into the tensor field
            //int indexShared = ((threadIdx.y + v) * blockDim.y + threadIdx.x + u);
            if (index < width * height && index >= 0) {                                     // DAVID: This will cause wrap-around artifacts
                glm::mat2 T(
                    data[4 * index + 0],
                    data[4 * index + 1],
                    data[4 * index + 2],
                    data[4 * index + 3]);

                VoteContribution vc = Saliency(T, u, v, sigma);                             // calculate the saliency given the tensor at (u,v)
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

tira::image<glm::mat2> CPUImplementation(tira::image<glm::mat2> Tn, int sigma)
{
    std::cout << "**********CPU**********" << std::endl;

    std::chrono::high_resolution_clock::time_point start, stop;
    start = std::chrono::high_resolution_clock::now();

    tira::image<glm::mat2> T = voteCPU(Tn, sigma);

    stop = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds d;
    d = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Elapsed time: " << d.count() / 1000.0f << " seconds" << std::endl;

    return T;
}

/// <summary>
/// Calculates one iteration of tensor voting given an input tensor field Tn
/// </summary>
/// <param name="Tn">Input tensor field as a 2D image of matrices</param>
/// <param name="sigma">Standard deviation for the decay function</param>
/// <param name="w">Window size for the decay function (usually dependent on sigma)</param>
/// <returns></returns>
float *CUDAImplementation(tira::image<glm::mat2> Tn, int sigma, int w)
{
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

    std::cout << "**********CUDA**********" << std::endl;

    float width = Tn.shape()[1];
    float height = Tn.shape()[0];
    int size = 4 * width * height;

    float *data = (float *)Tn.data();

    float *inArray;
    float *outArray;

    HANDLE_ERROR(cudaMalloc(&inArray, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&outArray, size * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(inArray, data, size * sizeof(float), cudaMemcpyHostToDevice));

    size_t blockDim = sqrt(props.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);

    dim3 blocks(width / threads.x + 1, height / threads.y + 1);

    int sharedBytes = props.sharedMemPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    voteGPU<<<blocks, threads, sharedBytes>>>(inArray, outArray, sigma, w, width, height);              // call the CUDA kernel for voting

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float *gpu_out = new float[size];
    HANDLE_ERROR(cudaMemcpy(gpu_out, outArray, size * sizeof(float), cudaMemcpyDeviceToHost));

    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);

    std::cout << "Elapsed time: " << totalTime / 1000 << " seconds" << std::endl;

    return gpu_out;
}

