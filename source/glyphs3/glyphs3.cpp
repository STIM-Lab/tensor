#include "../include/glyphs.h"

#include <iostream>



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


    field2glyphs(Field, arg_outfile, arg_sigma, arg_epsilon, normalize, arg_subdiv);



}
