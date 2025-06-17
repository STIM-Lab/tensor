#include <glm/glm.hpp>
#define GLM_FORCE_CUDA

#define PI 3.14159265358979323846

#include <tira/eigen.cuh>

#include <tira/tensorvote.cuh>
#include <string>
#include <iostream>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

#include <chrono>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

extern float t_voting;
extern float t_device2host;
extern float t_host2device;
extern float t_eigendecomposition;
extern float t_total;
extern float t_devicealloc;
extern float t_devicefree;
extern float t_deviceprops;

void cudacaller_tensorvote2(float* input_field, float* output_field, unsigned int s0, unsigned int s1, float sigma, float sigma2,
        unsigned int w, unsigned int power, int device, bool STICK, bool PLATE, bool debug, unsigned samples) {
    tira::cuda::tensorvote2(input_field, output_field, s0, s1, sigma, sigma2,
        w, power, device, STICK, PLATE, debug, samples);
}

float* cudaEigenvalues3(float* tensors, unsigned int n, int device);
float* cudaEigenvectors3DPolar(float* tensors, float* evals, unsigned int n, int device);

struct TensorAngleCalculation2D
{
    glm::mat2 votes;
    float decay;
};

struct multiVec2
{
    glm::vec2 x;
    glm::vec2 y;
};

void save_field2D(float* field, unsigned int sx, unsigned int sy, unsigned int vals, std::string filename);

struct VoteContribution3D
{
    glm::mat3 votes;
    float decay;
};

struct TensorAngleCalculation3D
{
    glm::mat3 votes;
    float decay;
};

void save_field3D(float* field, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int vals, std::string filename);

/// <summary>
/// Calculate the stick vote for the relative position (u, v) given the voter eigenvales and eigenvectors
/// </summary>
/// <param name="u">u coordinate for the relative position of the receiver</param>
/// <param name="v">v coordinate for the relative position of the receiver</param>
/// <param name="sigma1">decay value (standard deviation)</param>
/// <param name="eigenvectors">array containing the largest eigenvector</param>
/// <returns></returns>
__host__ __device__  VoteContribution3D StickVote3D(const float u, const float v, const float w, const float sigma1, const float sigma2, const float* eigenvectors, const unsigned int power) {

    glm::vec3 q(eigenvectors[0], eigenvectors[1], eigenvectors[2]);             // get the eigenvector

    glm::vec3 d = glm::vec3(u, v, w);                                           // normalize the direction vector
    float l = sqrt(u * u + v * v + w * w);                                      // calculate ell (distance between voter/votee)
    if (l == 0) d = glm::vec3(q[0], q[1], q[2]);                                // assumes that the voter DOES contribute to itself
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

__host__ __device__  glm::mat3 PlateVote3D(float u, float v, float w, float sigma1, float sigma2, unsigned int power) {
    
    float l = sqrt(u * u + v * v + w * w);                              // calculate the distance between voter and votee
    glm::vec3 d(0.0f);
    if (l != 0) d = glm::normalize(glm::vec3(u, v, w));                 // normalize direction vector
    
    float dx = d.x;
    float dy = d.y;
    float dz = d.z;

    // calculate the length and angle of the reflected direction matrix on xy-plane 
    float alpha = sqrt(dx*dx + dy*dy);
    float a2 = alpha * alpha;
    float phi = std::atan2(dy, dx);

    // build the rotation matrix about the z-axis by phi degrees
    glm::mat3 Rz(0.0f);
    Rz[0][0] = std::cos(phi);   Rz[0][1] = -std::sin(phi);
    Rz[1][0] = std::sin(phi);   Rz[1][1] = std::cos(phi);
    Rz[2][2] = 1.0f;

    // build the rotation matrix about the z-axis by -phi degrees
    glm::mat3 Rz_rev = glm::transpose(Rz);

    // compute beta and hypergeometric integrals
    double p_d = static_cast<double>(power);
    double a2_d = static_cast<double>(a2);
    double J0 = 0.5 * PI * boost::math::hypergeometric_pFq(std::vector<double>{ -p_d, 1.5 },
        std::vector<double>{ 2.0 }, a2_d);
    double J1 = PI * boost::math::hypergeometric_pFq(std::vector<double>{ -p_d, 0.5 },
        std::vector<double>{ 1.0 }, a2_d);
    double K0 = boost::math::beta(0.5, p_d + 1.5);
    double K1 = boost::math::beta(0.5, p_d + 0.5);

    // calcualte each term
    glm::mat3 A(0.0f);
    glm::mat3 B(0.0f);

    float tmp_a2 = 1.0f - 2.0f * a2;

    A[0][0] = (tmp_a2 * tmp_a2) * static_cast<float>(J0);
    A[0][2] = -2.0f * alpha * dz * tmp_a2 * static_cast<float>(J0);
    A[2][0] = -2.0f * alpha * dz * tmp_a2 * static_cast<float>(J0);
    A[1][1] = static_cast<float>(J1 - J0);
    A[2][2] = 4.0f * a2 * (dz * dz) * static_cast<float>(J0);

    float a2p = std::pow(a2, static_cast<float>(power));

    B[0][0] = a2p * (tmp_a2 * tmp_a2) * static_cast<float>(K0);
    B[0][2] = -2.0f * alpha * a2p * dz * tmp_a2 * static_cast<float>(K0);
    B[2][0] = -2.0f * alpha * a2p * dz * tmp_a2 * static_cast<float>(K0);
    B[1][1] = a2p * static_cast<float>(K1 - K0);
    B[2][2] = 4.0f * a2p * a2 * (dz * dz) * static_cast<float>(K0);

    // rotate back to original coordinates
    glm::mat3 term_a = Rz * A * Rz_rev;
    glm::mat3 term_b = Rz * B * Rz_rev;

    // calculate exponential terms
    float e1 = 0.0f, e2 = 0.0f;
    float l2 = l * l;
    float s12 = sigma1 * sigma1;
    float s22 = sigma2 * sigma2;
    if (sigma1 > 0)
        e1 = std::exp(-l2 / s12);
    if (sigma2 > 0)
        e2 = std::exp(-l2 / s22);

    // final integral
    glm::mat3 PlateVote = (e1 * term_a) + (e2 * term_b);

    return PlateVote;
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
__global__ void kernelStickVote3D(float* VT, float* L, float* V, float sigma, float sigma2,
    unsigned int power, float norm, int w, int s0, int s1, int s2) {
    int x0 = blockDim.x * blockIdx.x + threadIdx.x;                                     // get the x, y, and z volume coordinates for the current thread
    int x1 = blockDim.y * blockIdx.y + threadIdx.y;
    int x2 = blockDim.z * blockIdx.z + threadIdx.z;

    if (x0 >= s0 || x1 >= s1 || x2 >= s2)                                               // if not within bounds of image, return
        return;

    glm::mat3 Votee(0.0f);
    int hw = int(w / 2);
    // Precompute valid ranges
    int w0_start = max(-hw, -x0);
    int w0_end = min(hw, s0 - 1 - x0);
    int w1_start = max(-hw, -x1);
    int w1_end = min(hw, s1 - 1 - x1);
    int w2_start = max(-hw, -x2);
    int w2_end = min(hw, s2 - 1 - x2);

    for (int w0 = w0_start; w0 <= w0_end; w0++) {                                                 // for each pixel in the window
        int r0 = x0 + w0;
            for (int w1 = w1_start; w1 <= w1_end; w1++) {
                int r1 = x1 + w1;
                    for (int w2 = w2_start; w2 <= w2_end; w2++) {
                        int r2 = x2 + w2;
                        size_t location_thread = r0 * s1 * s2 + r1 * s2 + r2;

                        // calculate the contribution of (w0, w1, w2) to (x,y,z)
                        float3 Vcart;
                        const float theta = V[location_thread * 4 + 2];
                        const float phi = V[location_thread * 4 + 3];

                        Vcart.x = sinf(theta) * cosf(phi);
                        Vcart.y = sinf(theta) * sinf(phi);
                        Vcart.z = cosf(theta);
                        if (Vcart.x == 0 && Vcart.y == 0 && Vcart.z == 0) Votee = glm::mat3(0);
                        else {
                            VoteContribution3D vote = StickVote3D((float)w0, (float)w1, (float)w2, sigma, sigma2, (float*)&Vcart, power);

                            //float l0 = L[(r0 * s1 * s2 + r1 * s2 + r2) * 3 + 0];
                            const float l1 = L[location_thread * 3 + 1];
                            const float l2 = L[location_thread * 3 + 2];

                            float scale = fabsf(l2) - fabsf(l1);
                            if (l2 < 0.0f) scale *= -1;

                            Votee = Votee + scale * vote.votes * vote.decay;
                            }
                        }
            }
    }
    size_t location_mat = (x0 * s1 * s2 + x1 * s2 + x2) * 9;
    VT[location_mat + 0] += Votee[0][0];
    VT[location_mat + 1] += Votee[0][1];
    VT[location_mat + 2] += Votee[0][2];
    VT[location_mat + 3] += Votee[1][0];
    VT[location_mat + 4] += Votee[1][1];
    VT[location_mat + 5] += Votee[1][2];
    VT[location_mat + 6] += Votee[2][0];
    VT[location_mat + 7] += Votee[2][1];
    VT[location_mat + 8] += Votee[2][2];
}

/// @brief The kernel to perform the 3D plate tensor voting on the GPU
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
__global__ void kernelPlateVote3D(float* VT, float* L, float sigma, float sigma2,
    unsigned int power, float norm, int w, int s0, int s1, int s2) {
    int x0 = blockDim.x * blockIdx.x + threadIdx.x;                                     // get the x, y, and z volume coordinates for the current thread
    int x1 = blockDim.y * blockIdx.y + threadIdx.y;
    int x2 = blockDim.z * blockIdx.z + threadIdx.z;

    if (x0 >= s0 || x1 >= s1 || x2 >= s2)                                               // if not within bounds of image, return
        return;

    glm::mat3 Votee(0.0f);
    int hw = int(w / 2);
    // Precompute valid ranges
    int w0_start = max(-hw, -x0);
    int w0_end = min(hw, s0 - 1 - x0);
    int w1_start = max(-hw, -x1);
    int w1_end = min(hw, s1 - 1 - x1);
    int w2_start = max(-hw, -x2);
    int w2_end = min(hw, s2 - 1 - x2);

    for (int w0 = w0_start; w0 <= w0_end; w0++) {                                                 // for each pixel in the window
        int r0 = x0 + w0;
        for (int w1 = w1_start; w1 <= w1_end; w1++) {
            int r1 = x1 + w1;
            for (int w2 = w2_start; w2 <= w2_end; w2++) {
                int r2 = x2 + w2;
                size_t location_thread = r0 * s1 * s2 + r1 * s2 + r2;

                const float l0 = L[(r0 * s1 * s2 + r1 * s2 + r2) * 3 + 0];
                const float l1 = L[location_thread * 3 + 1];

                float scale = fabsf(l1) - fabsf(l0);
                if (l1 < 0.0f) scale *= -1;
                Votee = Votee + scale * PlateVote3D((float)w0, (float)w1, (float)w2, sigma, sigma2, power);
            }
        }
    }
    size_t location_mat = (x0 * s1 * s2 + x1 * s2 + x2) * 9;
    VT[location_mat + 0] += Votee[0][0];
    VT[location_mat + 1] += Votee[0][1];
    VT[location_mat + 2] += Votee[0][2];
    VT[location_mat + 3] += Votee[1][0];
    VT[location_mat + 4] += Votee[1][1];
    VT[location_mat + 5] += Votee[1][2];
    VT[location_mat + 6] += Votee[2][0];
    VT[location_mat + 7] += Votee[2][1];
    VT[location_mat + 8] += Votee[2][2];
}


void cudaVote3D(float* input_field, float* output_field, unsigned int s0, unsigned int s1, unsigned int s2, float sigma, float sigma2,
    unsigned int w, unsigned int power, unsigned int device, bool STICK, bool PLATE, bool debug) {

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

    // Declare GPU arrays
    float* gpuOutputField;
    float* gpuV;
    float* gpuL;

    // Allocate GPU arrays
    start = std::chrono::high_resolution_clock::now();
    HANDLE_ERROR(cudaMalloc(&gpuOutputField, tensorField_bytes));
    HANDLE_ERROR(cudaMemset(gpuOutputField, 0, tensorField_bytes));
    HANDLE_ERROR(cudaMalloc(&gpuV, evecs_bytes));
    HANDLE_ERROR(cudaMalloc(&gpuL, evals_bytes));
    end = std::chrono::high_resolution_clock::now();
    float t_devicealloc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    // Copy input arrays
    HANDLE_ERROR(cudaMemcpy(gpuV, V, evecs_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuL, L, evals_bytes, cudaMemcpyHostToDevice));
    end = std::chrono::high_resolution_clock::now();
    float t_host2device = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Specify the CUDA block and grid dimensions
    size_t blockDim = sqrt(props.maxThreadsPerBlock) / 2;
    dim3 threads(blockDim, blockDim/2, blockDim/2);
    dim3 blocks(s0 / threads.x + 1, s1 / threads.y + 1, s2 / threads.z + 1);

    float sn = 1.0f;          //1.0 / sticknorm2D(sigma, sigma2, power);      The function hasn't been implemented yet...

    if (debug)
        std::cout << "Stick Area: " << sn << std::endl;

    start = std::chrono::high_resolution_clock::now();
    if (STICK)
        kernelStickVote3D << <blocks, threads >> > (gpuOutputField, gpuL, gpuV, sigma, sigma2, power, sn, w, s0, s1, s2);              // call the CUDA kernel for voting
    if (PLATE)
        kernelPlateVote3D << <blocks, threads >> > (gpuOutputField, gpuL, sigma, sigma2, power, sn, w, s0, s1, s2);
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
        //std::cout << "Eigendecomposition:  " << t_eigendecomposition << " ms" << std::endl;
        std::cout << "Voting: " << t_voting << " ms" << std::endl;
        std::cout << "cudaMemcpy (H->D):  " << t_host2device << " ms" << std::endl;
        std::cout << "cudaMemcpy (D->H):  " << t_device2host << " ms" << std::endl;
        std::cout << "cudaMalloc: " << t_devicealloc << " ms" << std::endl;
        std::cout << "cudaFree: " << t_devicefree << " ms" << std::endl;
        std::cout << "cudaDeviceProps: " << t_deviceprops << " ms" << std::endl;
    }
}