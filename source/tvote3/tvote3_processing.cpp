#include <glm/glm.hpp>
#include <tira/volume.h>
#include "tvote3.h"
#include <tira/functions/eigen.h>
#include <tira/functions/tensorvote.h>

extern TV3_UI UI;

auto tic() {
    return std::chrono::high_resolution_clock::now();
}

float duration(auto t0, auto t1) {
    std::chrono::duration< float > fs = t1 - t0;
    return fs.count();
}

void VolumeFrom_Eigenvalue(const tira::volume<float>* lambda, tira::volume<float>* scalar, const unsigned i) {
    *scalar = lambda->channel(i);
}

inline static float Timg(tira::volume<glm::mat3>* tensors, const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int u, const unsigned int v) {
    return (*tensors)(x, y, z)[static_cast<int>(u)][static_cast<int>(v)];
}

void VolumeFrom_TensorElement3D(tira::volume<glm::mat3>* tensors, tira::volume<float>* elements, const unsigned int u, const unsigned int v) {

    *elements = tira::volume<float>(tensors->X(), tensors->Y(), tensors->Z(), 1);
    for (unsigned int zi = 0; zi < tensors->Shape()[0]; zi++) {
        for (unsigned int yi = 0; yi < tensors->Shape()[1]; yi++) {
            for (unsigned int xi = 0; xi < tensors->Shape()[2]; xi++) {
                const float val = Timg(tensors, xi, yi, zi, u, v);
                (*elements)(xi, yi, zi, 0) = val;
            }
        }
    }
}

void EigenDecomposition(tira::volume<glm::mat3>* tensor, tira::volume<float>* lambda, tira::volume<glm::vec2>* theta_phi, int cuda_device) {

    auto t0 = tic();

    float* eigenvalues_raw = hsa_eigenvalues3(reinterpret_cast<float*>(tensor->Data()), tensor->Size(), cuda_device);
    float* eigenvectors_raw = hsa_eigenvectors3spherical(reinterpret_cast<float*>(tensor->Data()), eigenvalues_raw, tensor->Size(), cuda_device);
    *lambda = tira::volume<float>(eigenvalues_raw, tensor->X(), tensor->Y(), tensor->Z(), 3);
    *theta_phi = tira::volume<glm::vec2>((glm::vec2*)eigenvectors_raw, tensor->X(), tensor->Y(), tensor->Z(), 3);

    free(eigenvalues_raw);
    free(eigenvectors_raw);

    auto t1 = tic();
    UI.t_eigendecomposition = duration(t0, t1);
}

void GaussianFilter(const tira::volume<glm::mat3>* tensors_in, tira::volume<glm::mat3>* tensors_out, const float sigma, glm::vec3 pixel_size, const int cuda_device) {
    unsigned int out_x, out_y, out_z;

    glm::mat3* out_raw = hsa_gaussian3(tensors_in->ConstData(), tensors_in->X(), tensors_in->Y(), tensors_in->Z(), sigma, pixel_size, out_x, out_y, out_z, cuda_device);
    *tensors_out = tira::volume<glm::mat3>(out_raw, out_x, out_y, out_z, 1, {pixel_size[0], pixel_size[1], pixel_size[2]});
    if (cuda_device < 0) delete[] out_raw;
    else free(out_raw);
}

void TensorVote(const tira::volume<glm::mat3>* tensors_in, tira::volume<glm::mat3>* tensors_out, const float sigma, const float sigma2, 
    const unsigned int p, const bool stick, const bool plate, const int cuda_device, const unsigned samples) {
    *tensors_out = tira::volume<glm::mat3>(tensors_in->X(), tensors_in->Y(), tensors_in->Z());

    const auto w = static_cast<unsigned int>(6.0f * std::max(sigma, sigma2) + 1.0f);

    // CUDA version
    if (cuda_device >= 0) {
        hsa_tensorvote3(reinterpret_cast<const float*>(tensors_in->ConstData()), reinterpret_cast<float*>(tensors_out->Data()),
            static_cast<unsigned int>(tensors_in->Shape()[0]), static_cast<unsigned int>(tensors_in->Shape()[1]), static_cast<unsigned int>(tensors_in->Shape()[2]),
            sigma, sigma2, w, p, cuda_device, stick, plate, false, samples);
    }

    // CPU version
    else {
        float* lambdas = tira::cpu::evals3_symmetric<float>(reinterpret_cast<const float*>(tensors_in->ConstData()), tensors_in->Size());
        float* evecs   = tira::cpu::evecs3spherical_symmetric<float>(reinterpret_cast<const float*>(tensors_in->ConstData()), lambdas, tensors_in->Size());

	    const size_t n = tensors_in->Size();
	    std::vector<glm::vec3> largest_q(n);                                    // q - the stick tensor (voter) orientation (largest eigenvector)
		std::vector<glm::vec3> smallest_q(n);                                   // q - the plate tensor (voter) orientation (smallest eigenvector)

        if (stick)  largest_q.resize(n);
		if (plate)  smallest_q.resize(n);

        for (size_t i = 0; i < n; ++i) {
            if (stick)  {
                const float theta_large = evecs[i * 6 + 4];
                const float phi_large = evecs[i * 6 + 5];
				largest_q[i] = tira::spherical_to_cartesian(theta_large, phi_large);
            }
            
            if (plate) {
                const float theta_small = evecs[i * 6 + 0];
                const float phi_small = evecs[i * 6 + 1];
				smallest_q[i] = tira::spherical_to_cartesian(theta_small, phi_small);
            }
	    }
        tira::cpu::tensorvote3(tensors_out->Data(), reinterpret_cast<glm::vec3*>(lambdas), 
            stick ? largest_q.data() : nullptr, 
            plate ? smallest_q.data() : nullptr, 
            glm::vec2(sigma, sigma2), p,
            w, tensors_in->Shape()[0], tensors_in->Shape()[1], tensors_in->Shape()[2], stick, plate, samples);
        delete[] evecs;
        delete[] lambdas;
    }
}

/**
* Save CPU and GPU eigendecomposition for the same tensor field to NPY files.
*/
void SaveEigenCPUvsCUDA(tira::volume<glm::mat3>* tensor_field, const int cuda_device, const std::string& base_name) {

    // CPU eigendecomposition
	const unsigned int s0 = tensor_field->Shape()[0];
	const unsigned int s1 = tensor_field->Shape()[1];
	const unsigned int s2 = tensor_field->Shape()[2];
	const unsigned int n = static_cast<unsigned int>(tensor_field->Size());

	float* evals_cpu = hsa_eigenvalues3(reinterpret_cast<float*>(tensor_field->Data()), n, -1);
    float* evecs_cpu = hsa_eigenvectors3spherical(reinterpret_cast<float*>(tensor_field->Data()), evals_cpu, n, -1);

	tira::volume<float> lambda_cpu(evals_cpu, s0, s1, s2, 3);
    tira::volume<float> theta_cpu(s0, s1, s2, 3);
    tira::volume<float> phi_cpu(s0, s1, s2, 3);

    for (unsigned int idx=0; idx < n; ++idx) {
        theta_cpu.Data()[idx * 3 + 0] = evecs_cpu[idx * 6 + 0];
        phi_cpu.Data()[idx * 3 + 0] = evecs_cpu[idx * 6 + 1];
        theta_cpu.Data()[idx * 3 + 1] = evecs_cpu[idx * 6 + 2];
        phi_cpu.Data()[idx * 3 + 1] = evecs_cpu[idx * 6 + 3];
        theta_cpu.Data()[idx * 3 + 2] = evecs_cpu[idx * 6 + 4];
        phi_cpu.Data()[idx * 3 + 2] = evecs_cpu[idx * 6 + 5];
	}

	lambda_cpu.save_npy(base_name + "_evals_cpu.npy");
	theta_cpu.save_npy(base_name + "_theta_cpu.npy");
	phi_cpu.save_npy(base_name + "_phi_cpu.npy");

    free(evals_cpu);
	free(evecs_cpu);

	// CUDA eigendecomposition
    float* evals_cuda = hsa_eigenvalues3(reinterpret_cast<float*>(tensor_field->Data()), n, cuda_device);
	float* evecs_cuda = hsa_eigenvectors3spherical(reinterpret_cast<float*>(tensor_field->Data()), evals_cuda, n, cuda_device);

	tira::volume<float> lambda_cuda(evals_cuda, s0, s1, s2, 3);
	tira::volume<float> theta_cuda(s0, s1, s2, 3);
    tira::volume<float> phi_cuda(s0, s1, s2, 3);

    for (unsigned int idx=0; idx < n; ++idx) {
        theta_cuda.Data()[idx * 3 + 0] = evecs_cuda[idx * 6 + 0];
        phi_cuda.Data()[idx * 3 + 0] = evecs_cuda[idx * 6 + 1];
        theta_cuda.Data()[idx * 3 + 1] = evecs_cuda[idx * 6 + 2];
        phi_cuda.Data()[idx * 3 + 1] = evecs_cuda[idx * 6 + 3];
        theta_cuda.Data()[idx * 3 + 2] = evecs_cuda[idx * 6 + 4];
        phi_cuda.Data()[idx * 3 + 2] = evecs_cuda[idx * 6 + 5];
	}

	lambda_cuda.save_npy(base_name + "_evals_cuda.npy");
	theta_cuda.save_npy(base_name + "_theta_cuda.npy");
	phi_cuda.save_npy(base_name + "_phi_cuda.npy");
	free(evals_cuda);
	free(evecs_cuda);
}
