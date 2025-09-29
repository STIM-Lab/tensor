#include "glyphs3.h"
#include <tira/shapes.h>
#include <tira/eigen.h>


int main(int argc, char** argv) {
    tira::volume<glm::mat3> Field;
    Field.load_npy<float>("vote13.npy");
    //Field.load_npy<float>("impulse_blur_13.npy");

    float epsilon = 0.01;
    float sigma = 0.001;
    int subdiv = 1;

    tira::obj OBJ;

    for (size_t zi = 0; zi < Field.Z(); zi++) {
        for (size_t yi = 0; yi < Field.Y(); yi++) {
            for (size_t xi = 0; xi < Field.X(); xi++) {
                glm::mat3 t = Field(xi, yi, zi);

                float lambda[3];
                tira::eval3_symmetric<float>(t[0][0], t[0][1], t[1][1], t[0][2], t[1][2], t[2][2], 
                    lambda[0], lambda[1], lambda[2]);

                float evec0[3];
                float evec1[3];
                float evec2[3];
                tira::evec3_symmetric<float>(t[0][0], t[0][1], t[1][1], t[0][2], t[1][2], t[2][2], 
                    (const float*)lambda, evec0, evec1, evec2);

                tira::mtl new_material;
                new_material.Kd[0] = evec2[0];
                new_material.Kd[1] = evec2[1];
                new_material.Kd[2] = evec2[2];

                // if the largest eigenvalue is less than epsilon, skip this glyph
                if (lambda[2] < epsilon) continue;
                if (lambda[1] / lambda[2] < sigma) lambda[1] = sigma * lambda[2];
                if (lambda[0] / lambda[2] < sigma) lambda[0] = sigma * lambda[2];

                tira::tmesh sq = tira::superquadric(lambda[0], lambda[1], lambda[2], 20.0f, 3.0f, subdiv);

                glm::mat3 R;
                R[0][0] = evec2[0];
                R[1][0] = evec2[1];
                R[2][0] = evec2[2];
                R[0][1] = evec1[0];
                R[1][1] = evec1[1];
                R[2][1] = evec1[2];
                R[0][2] = evec0[0];
                R[1][2] = evec0[1];
                R[2][2] = evec0[2];

                //tira::tmesh sq_rotate = sq.Transform(glm::transpose(R), 0);
                tira::tmesh sq_rotate = sq.Transform(R, 0);
                tira::tmesh sq_translate = sq_rotate.Translate({ (float)xi, (float)yi, (float)zi }, 0);
                
                std::stringstream ss;
                ss << xi << "_" << yi << "_" << zi;
                new_material.name = ss.str();
                OBJ.AddMesh(sq_translate, ss.str(), new_material);

                std::cout << ss.str() << std::endl;
            }
        }
    }

    OBJ.Save("test.obj");

}
