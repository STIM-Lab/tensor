#include "glyphs3.h"
#include <tira/shapes.h>
#include <tira/eigen.h>

#include <boost/program_options.hpp>

std::string arg_infile;
std::string arg_outfile;
float arg_epsilon;
float arg_sigma;
unsigned arg_subdiv;
bool normalize = false;


int main(int argc, char** argv) {


    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("input", boost::program_options::value<std::string>(&arg_infile), "input filename for the tensor field (*.npy)")
        ("output", boost::program_options::value<std::string>(&arg_outfile)->default_value("a.obj"), "output file that will be saved after processing")
        ("epsilon", boost::program_options::value<float>(&arg_epsilon)->default_value(0.01f), "cull glyphs (l2 < epsilon will not render)")
        ("sigma", boost::program_options::value<float>(&arg_sigma)->default_value(0.05f), "smallest allowed ratio for l0/l2 and l1/l2")
        ("normalize", "normalize all tensors to the largest eigenvalue")
        ("subdiv", boost::program_options::value<unsigned>(&arg_subdiv)->default_value(1), "subdivision iterations for superquadrics")
        ("help", "produce help message");
    boost::program_options::variables_map vm;

    boost::program_options::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    tira::volume<glm::mat3> Field;
    if (!vm.count("input")) {
        std::cout << "ERROR: no input file provided" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("normalize"))
        normalize = true;

    Field.load_npy<float>(arg_infile);

    float epsilon = arg_epsilon;
    float sigma = arg_sigma;
    int subdiv = arg_subdiv;

    tira::obj OBJ;

    float max_l2 = 0;
    if (normalize) {
        for (size_t zi = 0; zi < Field.Z(); zi++) {
            for (size_t yi = 0; yi < Field.Y(); yi++) {
                for (size_t xi = 0; xi < Field.X(); xi++) {
                    glm::mat3 t = Field(xi, yi, zi);
                    float lambda[3];
                    tira::eval3_symmetric<float>(t[0][0], t[0][1], t[1][1], t[0][2], t[1][2], t[2][2],
                        lambda[0], lambda[1], lambda[2]);
                    if (lambda[2] > max_l2)
                        max_l2 = lambda[2];
                }
            }
        }
        std::cout << "Normalizing to l2 = " << max_l2;
    }

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
                new_material.Kd[0] = std::abs(evec2[0]);
                new_material.Kd[1] = std::abs(evec2[1]);
                new_material.Kd[2] = std::abs(evec2[2]);

                // if the largest eigenvalue is less than epsilon, skip this glyph
                if (lambda[2] < epsilon) continue;
                if (lambda[1] / lambda[2] < sigma) lambda[1] = sigma * lambda[2];
                if (lambda[0] / lambda[2] < sigma) lambda[0] = sigma * lambda[2];

                tira::tmesh sq;
                if(normalize)
                    sq = tira::superquadric(lambda[0]/max_l2, lambda[1]/max_l2, lambda[2]/max_l2, 0.5f, 3.0f, subdiv);
                else
                    sq = tira::superquadric(lambda[0], lambda[1], lambda[2], 0.5f, 3.0f, subdiv);

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

                tira::tmesh sq_rotate = sq.Transform(glm::transpose(R), 0);
                //tira::tmesh sq_rotate = sq.Transform(R, 0);
                tira::tmesh sq_translate = sq_rotate.Translate({ (float)xi, (float)yi, (float)zi }, 0);
                
                std::stringstream ss;
                ss << xi << "_" << yi << "_" << zi;
                new_material.name = ss.str();
                OBJ.AddMesh(sq_translate, ss.str(), new_material);

                std::cout << ss.str() << std::endl;
            }
        }
    }

    OBJ.Save(arg_outfile);

}
