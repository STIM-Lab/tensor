#include <stdio.h>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "tira/image.h"

#define _USE_MATH_DEFINES
#include <math.h>

void SaveTensorField(tira::image<glm::mat2> T, std::string filename) {
    tira::image<float> out((float *)T.data(), T.X(), T.Y(), 4);
    out.save_npy(filename);
}

void SaveTensorField(float *data, float width, float height, std::string filename) {
    tira::image<float> out(data, width, height, 4);
    out.save_npy(filename);
}

__host__ __device__ glm::vec2 gpuEigenvalues2D(glm::mat2 T) {
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

__host__ __device__ glm::vec2 gpuEigenvectors2D(glm::mat2 T, glm::vec2 lambdas, unsigned int index = 1) {
    float d = T[0][0];
    float e = T[0][1];
    float f = e;
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

float Decay(float cos_theta, float length, float sigma) {
    float c = exp(-(length * length) / (sigma * sigma));
    float radial = 1 - (cos_theta * cos_theta);
    float D = c * radial;
    return D;
}

__host__ __device__ VoteContribution Saliency(float u, float v, float sigma, float* eigenvalues, float* eigenvectors) {

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