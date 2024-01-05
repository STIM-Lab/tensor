//#define GLM_FORCE_CUDA
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

struct multiVec2
{
    glm::vec2 x;
    glm::vec2 y;
};

bool NonZeroTensor(glm::mat2 T);
float Decay(float angle, float length, int sigma);
TensorAngleCalculation SaliencyTheta(float theta, float u, float v, int sigma);