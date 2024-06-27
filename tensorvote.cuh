#include <stdio.h>
#include <string>
#include <iostream>




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

void save_field(float* field, unsigned int sx, unsigned int sy, unsigned int vals, std::string filename);