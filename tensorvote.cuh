//#include <stdio.h>
//#include <string>
//#include <vector>

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>

//#include "tira/image.h"

//#define _USE_MATH_DEFINES
//#include <math.h>

//__host__ __device__ bool NonZeroTensor(glm::mat2 T);
//void SaveTensorField(tira::image<glm::mat2> T, std::string filename);
//void SaveTensorField(float* data, float width, float height, std::string filename);
//__host__ __device__ glm::vec2 gpuEigenvalues2D(glm::mat2 T);
//__host__ __device__ glm::vec2 gpuEigenvectors2D(glm::mat2 T, glm::vec2 lambdas, unsigned int index = 1);
//float Decay(float cos_theta, float lenght, float sigma);
//__host__ __device__ VoteContribution Saliency(float u, float v, float sigma, float* eigenvalues)

glm::vec2 Eigenvalues2D(glm::mat2 T);

void cudaVote2D(float* input_field, float* output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w, unsigned int device);