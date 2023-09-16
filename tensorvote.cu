#include <iostream>
#include <stdio.h>

#include "tensorvote.cuh"

#include <chrono>

#ifdef __CUDACC__
#define __device__
#endif

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
    // get index for input data
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    // if not within bounds of image, return
    if (y >= height || x >= width)
        return;

    float vt[4] = {0, 0, 0, 0};

    for (int u = -w; u < w; u++)
    {
        for (int v = -w; v < w; v++)
        {
            int index = ((v + y) * width + x + u);
            int indexShared = ((threadIdx.y + v) * blockDim.y + threadIdx.x + u);
            if (index < width * height && index >= 0)
            {
                glm::mat2 T(
                    data[4 * index + 0],
                    data[4 * index + 1],
                    data[4 * index + 2],
                    data[4 * index + 3]);

                VoteContribution vc = Saliency(T, u, v, sigma);
                vt[0] += vc.votes[0][0] * vc.decay;
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

    voteGPU<<<blocks, threads, sharedBytes>>>(inArray, outArray, sigma, w, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float *gpu_out = new float[size];
    HANDLE_ERROR(cudaMemcpy(gpu_out, outArray, size * sizeof(float), cudaMemcpyDeviceToHost));

    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);

    std::cout << "Elapsed time: " << totalTime / 1000 << " seconds" << std::endl;

    return gpu_out;
}

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        std::cout << "Need input in the form of {input npy file} {output npy file} {sigma} {windowSize (optional)}" << std::endl;
        return -1;
    }

    std::string inFile(argv[1]);
    std::string outFile(argv[2]);
    int sigma = atoi(argv[3]);
    int windowSize;

    if (argc > 4) {
        windowSize = atoi(argv[4]);
    }
    else {
        windowSize = 6 * sigma / 2;
    }

    tira::image<glm::mat2> Tn = LoadTensorField(inFile);

    // tira::image<glm::mat2> T_cpu = CPUImplementation(Tn, sigma);
    float *T_cuda = CUDAImplementation(Tn, sigma, windowSize);

    // SaveTensorField(T_cpu, "cpu_" + outFile);
    SaveTensorField(T_cuda, Tn.shape()[1], Tn.shape()[0], "cuda_" + outFile);

    return 0;
}