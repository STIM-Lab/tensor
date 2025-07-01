#include "tview3.h"

glm::mat3 GenerateImpulse(glm::vec2 stick_polar, float plate_theta, glm::vec2 lambdas) {
    // create a tensor from the input data
    float sin_theta = std::sin(stick_polar.x);
    float cos_theta = std::cos(stick_polar.x);
    float sin_phi = std::sin(stick_polar.y);
    float cos_phi = std::cos(stick_polar.y);

    glm::vec3 v2(sin_phi * cos_theta, sin_phi * sin_theta, cos_phi);        // calculate the longest vector in cartesian coordinates

    glm::vec3 v0_temp(1.0f, 0.0f, 0.0f);
    glm::vec3 v1 = glm::normalize(glm::cross(v0_temp, v2));
    glm::vec3 v0 = glm::normalize(glm::cross(v1, v2));

    glm::mat3 T = lambdas[0] * glm::outerProduct(v0, v0) +
        lambdas[1] * glm::outerProduct(v1, v1) +
        glm::outerProduct(v2, v2);
    return T;
}

void GenerateImpulseField(tira::volume<glm::mat3>* tensor, unsigned resolution, glm::vec2 stick_polar, float plate_theta, glm::vec2 lambdas) {

    glm::mat3 T = GenerateImpulse(stick_polar, plate_theta, lambdas);

    *tensor = tira::volume<glm::mat3>(resolution, resolution, resolution);                  // create a new tensor field
    glm::mat3 zero = glm::mat3(0.0f);                                        // fill it with zero tensors
    *tensor = zero;

    (*tensor)(resolution/2, resolution/2, resolution/2) = T;
}
