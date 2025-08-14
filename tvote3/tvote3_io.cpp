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

    glm::mat3 V;
    V[0][0] = v0[0];    V[0][1] = v0[1];    V[0][2] = v0[2];
    V[1][0] = v1[0];    V[1][1] = v1[1];    V[1][2] = v1[2];
    V[2][0] = v2[0];    V[2][1] = v2[1];    V[2][2] = v2[2];
    glm::mat3 Vinv = glm::transpose(V);
    glm::mat3 L(1.0f);
    L[0][0] = lambdas[0];
    L[1][1] = lambdas[1];
    L[2][2] = 1.0f;
    glm::mat3 T = V * L * Vinv;
    return T;

    std::cout<<"------------------------------------------------------"<<std::endl;
    std::cout<<"impulse v0: "<<v0.x<<", "<<v0.y<<", "<<v0.z<<std::endl;
    std::cout<<"v0 . v1 = "<<glm::dot(v0, v1)<<std::endl;
    std::cout<<"v0 . v2 = "<<glm::dot(v0, v2)<<std::endl;
    std::cout<<"v1 . v2 = "<<glm::dot(v1, v2)<<std::endl;

    //glm::mat3 T = glm::outerProduct(lambdas[0] * v0, lambdas[0] * v0) +
    //              glm::outerProduct(lambdas[1] * v1, lambdas[1] * v1) +
    //                           glm::outerProduct(v2, v2);
    //return T;
}

void GenerateImpulseField(tira::volume<glm::mat3>* tensor, unsigned resolution, glm::vec2 stick_polar, float plate_theta, glm::vec2 lambdas) {

    glm::mat3 T = GenerateImpulse(stick_polar, plate_theta, lambdas);

    *tensor = tira::volume<glm::mat3>(resolution, resolution, resolution);                  // create a new tensor field
    glm::mat3 zero = glm::mat3(0.0f);                                        // fill it with zero tensors
    *tensor = zero;

    (*tensor)(resolution/2, resolution/2, resolution/2) = T;
}
