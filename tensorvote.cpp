#include "tensorvote.h"

#include <iostream>

#include <boost/program_options.hpp>

#include <tira/image.h>

#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>

std::string in_inputname;
std::string in_outputname;
float in_sigma;
unsigned int in_window;
int in_cuda;

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line << std::endl;
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

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

bool NonZeroTensor(glm::mat2 T)
{
    return T[0][0] || T[0][1] || T[1][0] || T[1][1];
}

// small then large
glm::vec2 Eigenvalues2D(glm::mat2 T)
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

// small then large
glm::mat2 Eigenvectors2D(glm::mat2 T, glm::vec2 lambdas)
{
    float d = T[0][0];
    float e = T[0][1];
    float f = e;
    float g = T[1][1];

    if (e != 0)
    {
        glm::vec2 vec0 = glm::normalize(glm::vec2(1.0, (lambdas[0] - d) / e));
        glm::vec2 vec1 = glm::normalize(glm::vec2(1.0, (lambdas[1] - d) / e));

        return glm::mat2(vec0, vec1);
    }
    else if (g == 0)
    {
        return glm::mat2(glm::vec2(1.0, 0.0), glm::vec2(1.0, 0.0));
    }
    else
    {
        return glm::mat2(glm::vec2(0.0, 1.0), glm::vec2(0.0, 1.0));
    }
}

void cpuEigendecomposition(float *input_field, float *eigenvectors, float *eigenvalues, unsigned int sx, unsigned int sy)
{

    unsigned int i;
    for (unsigned int yi = 0; yi < sy; yi++)
    { // for each tensor in the field
        for (unsigned int xi = 0; xi < sx; xi++)
        {
            i = (yi * sx + xi); // calculate a 1D index into the 2D image

            // DAVID: May have to transpose this
            unsigned int ti = i * 4;
            glm::mat2 T(input_field[ti + 0], input_field[ti + 1], input_field[ti + 2], input_field[ti + 3]);

            glm::vec2 evals = Eigenvalues2D(T);         // calculate the eigenvalues
            glm::mat2 evecs = Eigenvectors2D(T, evals); // calculate the eigenvectors

            eigenvectors[ti + 0] = evecs[0][0]; // save the eigenvectors to the output array
            eigenvectors[ti + 1] = evecs[0][1];
            eigenvectors[ti + 2] = evecs[1][0];
            eigenvectors[ti + 3] = evecs[1][1];

            unsigned int vi = i * 2;
            eigenvalues[vi + 0] = evals[0]; // save the eigenvalues to the output array
            eigenvalues[vi + 1] = evals[1];
        }
    }
}

float Decay(float angle, float length, int sigma)
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

TensorAngleCalculation SaliencyTheta(float theta, float u, float v, int sigma = 10)
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

VoteContribution Saliency(glm::mat2 T, float u, float v, int sigma, float *eigenvalues, float *eigenvectors)
{

    if (!NonZeroTensor(T))
    {
        VoteContribution out;
        out.votes = glm::mat2(0, 0, 0, 0);
        out.decay = 0;

        return out;
    }

    // glm::vec2 lambdas = Eigenvalue2D(T);

    // ordered as large then small
    glm::vec2 lambdas(eigenvalues[1], eigenvalues[0]);

    // multiVec2 e_vecs = Eigenvector2D(T, lambdas);
    multiVec2 e_vecs = multiVec2(glm::vec2(eigenvectors[2], eigenvectors[3]), glm::vec2(eigenvectors[0], eigenvectors[1]));

    // glm::vec2 k1 = e_vecs.y;

    glm::vec2 k1(eigenvectors[0], eigenvectors[1]);

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

void cpuVote2D(float *input_field, float *output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w)
{
    float *V = new float[sx * sy * 2 * 2];            // allocate space for the eigenvectors
    float *L = new float[sx * sy * 2];                // allocate space for the eigenvalues
    cpuEigendecomposition(input_field, V, L, sx, sy); // calculate the eigendecomposition of the entire field

    for (unsigned int yi = 0; yi < sy; yi++)
    {
        for (unsigned int xi = 0; xi < sx; xi++)
        {
            for (int u = -w; u < w; u++)
            {
                for (int v = -w; v < w; v++)
                {
                    // DO TENSOR VOTING HERE
                    // output_field[??] = ??

                    glm::mat2 T = glm::mat2(
                        input_field[(yi * sx + xi) * 4 + 0],
                        input_field[(yi * sx + xi) * 4 + 1],
                        input_field[(yi * sx + xi) * 4 + 2],
                        input_field[(yi * sx + xi) * 4 + 3]);

                    VoteContribution vote = Saliency(
                                                T,
                                                u,
                                                v,
                                                sigma,
                                                &L[(yi * sx + xi) * 2],
                                                &V[(yi * sx + xi) * 4])
                                                .votes[0][0];

                    output_field[(yi + u) * sx + (xi + v) + 0] += vote.votes[0][0] * vote.decay;
                    output_field[(yi + u) * sx + (xi + v) + 1] += vote.votes[0][1] * vote.decay;
                    output_field[(yi + u) * sx + (xi + v) + 2] += vote.votes[1][0] * vote.decay;
                    output_field[(yi + u) * sx + (xi + v) + 3] += vote.votes[1][1] * vote.decay;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{

    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("input", boost::program_options::value<std::string>(&in_inputname), "output filename for the coupled wave structure")("output", boost::program_options::value<std::string>(&in_outputname)->default_value("tv.npy"), "optional image field corresponding to the tensors")("sigma", boost::program_options::value<float>(&in_sigma)->default_value(5.0f), "order used to calculate the first derivative")("window", boost::program_options::value<unsigned int>(&in_window), "window size (6 * sigma as default)")("cuda", boost::program_options::value<int>(&in_cuda)->default_value(0), "cuda device index (-1 for CPU)")("help", "produce help message");
    boost::program_options::variables_map vm;

    boost::program_options::positional_options_description p;
    p.add("input", -1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

    boost::program_options::notify(vm);

    // if the user passes the help parameter, output command line details
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    // calculate the window size if one isn't provided
    if (!vm.count("window"))
    {
        in_window = int(3 * in_sigma);
    }

    // make sure that the selected CUDA device is valid, and switch to the CPU if it isn't
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

    tira::field<float> T(in_inputname); // load the input tensor field
    tira::field<float> Tr(T.shape());   // create a field to store the vote result

    // CPU IMPLEMENTATION
    if (in_cuda < 0)
    {
        cpuVote2D(T.data(), Tr.data(), T.shape()[0], T.shape()[1], in_sigma, in_window);
    }

    Tr.save_npy(in_outputname);

    return 0;
}