#include <string>

#include <tira/image.h>
#include <glm/glm.hpp>

#include "tvote2.h"

extern TV2_UI UI;

void LoadTensorField(const std::string& filename, tira::image<glm::mat2>* tensor) {
    tensor->load_npy<float>(filename);
}

void SaveTensorField(const std::string& filename, tira::image<glm::mat2>* tensor) {
    tensor->SaveNpy(filename);
}

/// Generate an impulse tensor field with a single tensor at the center.
void GenerateImpulse(tira::image<glm::mat2>* tensor, unsigned resolution, float theta, float anisotropy) {

    *tensor = tira::image<glm::mat2>(resolution, resolution);                       // create a new tensor field
    glm::mat2 zero = glm::mat2(0.0f);                                        // fill it with zero tensors
    *tensor = zero;

    glm::mat2 P = glm::mat2(cosf(theta), sinf(theta), -sinf(theta), cosf(theta));
    glm::mat2 Pinv = glm::inverse(P);
    glm::mat2 L = glm::mat2(1.0f);
    L[1][1] = -(anisotropy * anisotropy - 1);

    (*tensor)(resolution/2, resolution/2) = P * L * Pinv;

    UI.field_loaded = true;
}
