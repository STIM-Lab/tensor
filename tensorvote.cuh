#include <stdio.h>
#include <string>
#include <iostream>




struct VoteContribution2D
{
    glm::mat2 votes;
    float decay;
};

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
