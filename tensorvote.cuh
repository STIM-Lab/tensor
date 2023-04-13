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

#define GLM_FORCE_CUDA
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

struct multiVec2 {
    glm::vec2 x;
    glm::vec2 y;
};

__host__ __device__ bool NonZeroTensor(glm::mat2 T)
{
    return T[0][0] || T[0][1] || T[1][0] || T[1][1];
}

tira::image<glm::mat2> LoadTensorField(std::string filename)
{
    tira::image<glm::mat2> T0;
    T0.load_npy<float>(filename);
    return T0;
}

void SaveTensorField(tira::image<glm::mat2> T, std::string filename) {
    tira::image<float> out((float*)T.data(), T.X(), T.Y(), 4);
    out.save_npy(filename);
}

void SaveTensorField(float* data, float width, float height, std::string filename) {
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

__host__ __device__ VoteContribution Saliency(glm::mat2 T, float u, float v, int sigma)
{

    if (!NonZeroTensor(T)) {
        VoteContribution out;
        out.votes = glm::mat2(0,0,0,0);
        out.decay = 0;

        return out;
    }

    glm::vec2 lambdas = Eigenvalue2D(T);

    multiVec2 e_vecs = Eigenvector2D(T, lambdas);

    glm::vec2 k1 = e_vecs.y;

    float theta = atan2(k1[1], k1[0]);

    TensorAngleCalculation st = SaliencyTheta(theta, u, v, sigma);

    float lambda1 = lambdas[1];
    float lambda2 = lambdas[0];
    float ecc = sqrt(1.0 - (pow(lambda2, 2) / pow(lambda1, 2)));

    VoteContribution out;
    out.votes = st.votes;
    out.decay = st.decay * ecc * lambda1;

    return out;
}