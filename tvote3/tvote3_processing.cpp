

#include "tvote3.h"
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
    for (unsigned int zi = 0; zi < tensors->shape()[0]; zi++) {
        for (unsigned int yi = 0; yi < tensors->shape()[1]; yi++) {
            for (unsigned int xi = 0; xi < tensors->shape()[2]; xi++) {
                const float val = Timg(tensors, xi, yi, zi, u, v);
                (*elements)(xi, yi, zi, 0) = val;
            }
        }
    }
}

void EigenDecomposition(tira::volume<glm::mat3>* tensor, tira::volume<float>* lambda, tira::volume<glm::vec2>* theta_phi, int cuda_device) {

    auto t0 = tic();

    float* eigenvalues_raw = hsa_eigenvalues3(reinterpret_cast<float*>(tensor->data()), tensor->size(), cuda_device);
    float* eigenvectors_raw = hsa_eigenvectors3spherical(reinterpret_cast<float*>(tensor->data()), eigenvalues_raw, tensor->size(), cuda_device);
    *lambda = tira::volume<float>(eigenvalues_raw, tensor->X(), tensor->Y(), tensor->Z(), 3);
    *theta_phi = tira::volume<glm::vec2>((glm::vec2*)eigenvectors_raw, tensor->X(), tensor->Y(), tensor->Z(), 3);

    free(eigenvalues_raw);
    free(eigenvectors_raw);

    auto t1 = tic();
    UI.t_eigendecomposition = duration(t0, t1);
}

void GaussianFilter(const tira::volume<glm::mat3>* tensors_in, tira::volume<glm::mat3>* tensors_out, const float sigma, glm::vec3 pixel_size, const int cuda_device) {
    unsigned int out_x, out_y, out_z;

    glm::mat3* out_raw = hsa_gaussian3(tensors_in->const_data(), tensors_in->X(), tensors_in->Y(), tensors_in->Z(), sigma, pixel_size, out_x, out_y, out_z, cuda_device);
    tensors_out = new tira::volume<glm::mat3>(out_raw, out_x, out_y, out_z, 1, {pixel_size[0], pixel_size[1], pixel_size[2]});

}
