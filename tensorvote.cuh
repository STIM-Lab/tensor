#include <stdio.h>
#include <string>
#include <iostream>
#include <stdio.h>

#include <glm/glm.hpp>

#include <tira/field.h>
#include <tira/image.h>

#include <cuda_runtime_api.h>

#include <boost/program_options.hpp>

#include <math.h>

#define _USE_MATH_DEFINES

__host__ __device__ static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line << std::endl;
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#include <glm/glm.hpp>

struct VoteContribution
{
    glm::mat2 votes;
    float decay;
};

struct TensorAngleCalculation
{
    glm::mat2 votes;
    float decay;
};

struct multiVec2
{
    glm::vec2 x;
    glm::vec2 y;
};

__host__ __device__ glm::vec2 Eigenvalues2D(glm::mat2 T);
__host__ __device__ glm::vec2 Eigenvector2D(glm::mat2 T, glm::vec2 lambdas, unsigned int index = 1);
__host__ __device__ void cpuEigendecomposition(float *input_field, float *eigenvectors, float *eigenvalues, unsigned int sx, unsigned int sy);
__host__ __device__ VoteContribution Saliency(float u, float v, float sigma, float* eigenvalues, float* eigenvectors);
__global__ void cudaVote2D(float* input_field, float* output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w, unsigned int device);