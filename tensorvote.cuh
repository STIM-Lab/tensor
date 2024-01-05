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

void SaveTensorField(tira::image<glm::mat2> T, std::string filename)
{
    tira::image<float> out((float *)T.data(), T.X(), T.Y(), 4);
    out.save_npy(filename);
}

void SaveTensorField(float *data, float width, float height, std::string filename)
{
    tira::image<float> out(data, width, height, 4);
    out.save_npy(filename);
}

__host__ __device__ glm::vec2 Eigenvalue2D(glm::mat2 T)
{
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

__host__ __device__ multiVec2 Eigenvector2D(glm::mat2 T, glm::vec2 lambdas)
{
    float d = T[0][0];
    float e = T[0][1];
    float f = e;
    float g = T[1][1];

    multiVec2 e_vecs;
    if (e != 0)
    {
        glm::vec2 vec0 = glm::normalize(glm::vec2(1.0, (lambdas[0] - d) / e));
        glm::vec2 vec1 = glm::normalize(glm::vec2(1.0, (lambdas[1] - d) / e));

        e_vecs.x = vec0;
        e_vecs.y = vec1;
    }
    else if (g == 0)
    {
        e_vecs.x = glm::vec2(1.0, 0.0);
        e_vecs.y = glm::vec2(1.0, 0.0);
    }
    else
    {
        e_vecs.x = glm::vec2(0.0, 1.0);
        e_vecs.y = glm::vec2(0.0, 1.0);
    }
    return e_vecs;
}



__host__ __device__ VoteContribution Saliency(glm::mat2 T, float u, float v, int sigma)
{

    if (!NonZeroTensor(T))
    {
        VoteContribution out;
        out.votes = glm::mat2(0, 0, 0, 0);
        out.decay = 0;

        return out;
    }

    glm::vec2 lambdas = Eigenvalue2D(T);

    multiVec2 e_vecs = Eigenvector2D(T, lambdas);

    glm::vec2 k1 = e_vecs.y;

    float theta = atan2(k1[1], k1[0]);

    TensorAngleCalculation st = SaliencyTheta(theta, u, v, sigma);

    float lambdaLarge = lambdas[1];
    float lambdaSmall = lambdas[0];
    float ecc = sqrt(1.0 - (pow(lambdaSmall, 2) / pow(lambdaLarge, 2)));

    if (isnan(ecc))
    {
        ecc = 0;
    }

    VoteContribution out;
    out.votes = st.votes;
    out.decay = st.decay * ecc * lambdaLarge;

    return out;
}