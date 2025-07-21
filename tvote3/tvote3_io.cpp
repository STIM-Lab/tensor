#include <numbers>

#include "tvote3.h"


glm::mat3 GenerateImpulse(glm::vec2 stick_polar, float plate_theta, glm::vec2 lambdas) {

    // convert the input spherical coordinates into Cartesian eigenvectors

    // calculate the largest eigenvector from the provided spherical coordinates
    float sin_theta = std::sin(stick_polar[0]);                                  // pre-compute trigonometric values
    float cos_theta = std::cos(stick_polar[0]);
    float sin_phi = std::sin(stick_polar[1]);
    float cos_phi = std::cos(stick_polar[1]);

    glm::vec3 v2(sin_phi * cos_theta, sin_phi * sin_theta, cos_phi);      // convert the largest eigenvector to Cartesian coordinates

    // calculate a starting vector that is orthogonal to v2
    float v0s_phi = stick_polar.y + std::numbers::pi / 2.0f;
    float sin_v0s_phi = std::sin(v0s_phi);
    float cos_v0s_phi = std::cos(v0s_phi);
    glm::vec3 v0_start(sin_v0s_phi * cos_theta, sin_v0s_phi * sin_theta, cos_v0s_phi);

    // calculate v0 using the user-specified plate_theta parameter as the angle from v0_start
    glm::quat q0_rotation = glm::rotate(glm::quat_cast(glm::mat3(1.0f)), plate_theta, v2);
    glm::mat3 v0_rotation = glm::mat3_cast(q0_rotation);
    glm::vec3 v0 = v0_rotation * v0_start;

    glm::vec3 v1 = glm::normalize(glm::cross(v0, v2));

    //glm::vec3 ortho_start(1.0f, 0.0f, 0.0f);
    //glm::vec3 v1 = glm::normalize(glm::cross(v0_temp, v2));
    //glm::vec3 v0 = glm::normalize(glm::cross(v1, v2));

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
