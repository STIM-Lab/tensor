#include "tensorvote.h"

#include <iostream>

#include <boost/program_options.hpp>

#include <tira/image.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

std::string in_inputname;
std::string in_outputname;
float in_sigma;
unsigned int in_window;
int in_cuda;
std::vector<float> in_votefield;
bool debug = false;

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line << std::endl;
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/// <summary>
/// Save a field of floating point values as a NumPy file
/// </summary>
/// <param name="field"> pointer to an array of floats </param>
/// <param name="sx"> size of the array in the x dimension </param>
/// <param name="sy"> size of the array in the y dimension </param>
/// <param name="vals"> number of values at each point </param>
/// <param name="filename"> filename to save the data </param>
void save_field(float* field, unsigned int sx, unsigned int sy, unsigned int vals, std::string filename) {
    std::vector<size_t> shape({ sx, sy, vals });
    tira::field<float> O(shape, field);
    O.save_npy(filename);
}

// small then large
glm::vec2 Eigenvalues2D(glm::mat2 T)
{
    float d = T[0][0];
    float e = T[0][1];
    float f = e;
    float g = T[1][1];

    float dpg = d + g;
    float disc = sqrt((4 * e * f) + pow(d - g, 2));
    float a = (dpg + disc) / 2.0f;
    float b = (dpg - disc) / 2.0f;
    float min = a < b ? a : b;
    float max = a > b ? a : b;
    glm::vec2 out(min, max);
    return out;
}

// small then large
glm::vec2 Eigenvectors2D(glm::mat2 T, glm::vec2 lambdas, unsigned int index = 1)
{
    float d = T[0][0];
    float e = T[0][1];
    float f = e;
    float g = T[1][1];

    if (e != 0) {
        return glm::normalize(glm::vec2(1.0, (lambdas[index] - d) / e));
    }
    else if (g == 0) {
        return glm::vec2(1.0, 0.0);
    }
    else {
        return glm::vec2(0.0, 1.0);
    }
}

void cpuEigendecomposition(float *input_field, float *eigenvectors, float *eigenvalues, unsigned int sx, unsigned int sy)
{

    unsigned int i;
    for (unsigned int yi = 0; yi < sy; yi++)
    { // for each tensor in the field
        for (unsigned int xi = 0; xi < sx; xi++)
        {
            i = (yi * sx + xi); // calculate a 1D index into the 2D image

            unsigned int ti = i * 4;                                // update the tensor index (each tensor is 4 elements)
                                                                    // store the tensor as a matrix to make this more readable
            glm::mat2 T(input_field[ti + 0],
                input_field[ti + 1], 
                input_field[ti + 2], 
                input_field[ti + 3]);

            glm::vec2 evals = Eigenvalues2D(T);                     // calculate the eigenvalues
            glm::vec2 evec = Eigenvectors2D(T, evals);              // calculate the largest eigenvector

            unsigned int vi = i * 2;                                // update the vector/value index (each is 2 elements)
            eigenvectors[vi + 0] = evec[0];                         // save the eigenvectors to the output array
            eigenvectors[vi + 1] = evec[1];

            
            eigenvalues[vi + 0] = evals[0];                         // save the eigenvalues to the output array
            eigenvalues[vi + 1] = evals[1];
        }
    }
}

/// Calculate the contribution of a vote at position (u,v) relative to the current tensor T
// T is the current tensor
// (u,v) is the relative position of the votee
VoteContribution Saliency(float u, float v, int sigma, float *eigenvalues, float *eigenvectors)
{

    glm::vec2 k1(eigenvectors[0], eigenvectors[1]);

    float theta = atan2(k1[1], k1[0]);

    TensorAngleCalculation st = SaliencyTheta(theta, u, v, sigma);
 
    float lambdaLarge = eigenvalues[1];
    float lambdaSmall = eigenvalues[0];
    float ecc = sqrt(1.0 - (pow(lambdaSmall, 2) / pow(lambdaLarge, 2)));

    if (isnan(ecc))
    {
        ecc = 0;
    }

    VoteContribution out;
    out.votes = st.votes;
    out.decay = st.decay;// *ecc* lambdaLarge;

    return out;
}

void cpuVote2D(float *input_field, float *output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w)
{
    std::vector<float> V(sx * sy * 2);                          // allocate space for the eigenvectors
    std::vector<float> L(sx * sy * 2);                          // allocate space for the eigenvalues

    int hw = (int)(w / 2);                                      // calculate the half window size

    cpuEigendecomposition(input_field, &V[0], &L[0], sx, sy);   // calculate the eigendecomposition of the entire field

    if (debug) {
        save_field(&L[0], sx, sy, 2, "debug_eigenvalues.npy");
        save_field(&V[0], sx, sy, 2, "debug_eigenvector.npy");
    }

    int xr, yr;                                                 // x and y coordinates within the window
    for (unsigned int yi = 0; yi < sy; yi++) {                  // for each pixel in the image
        for (unsigned int xi = 0; xi < sx; xi++) {

            glm::mat2 T = glm::mat2(                            // retrieve the tensor for the current pixel
                input_field[(yi * sx + xi) * 4 + 0],
                input_field[(yi * sx + xi) * 4 + 1],
                input_field[(yi * sx + xi) * 4 + 2],
                input_field[(yi * sx + xi) * 4 + 3]);

            glm::mat2 Votee(0.0f);

            for (int v = -hw; v < hw; v++) {                    // for each pixel in the window
                yr = yi + v;
                if (yr >= 0 && yr < sy) {
                    for (int u = -hw; u < hw; u++) {
                        
                        xr = xi + u;
                        if (xr >= 0 && xr < sx) {
                                                                // calculate the contribution of (u,v) to (x,y)        
                            VoteContribution vote = Saliency(
                                u,
                                v,
                                sigma,
                                &L[(yr * sx + xr) * 2],
                                &V[(yr * sx + xr) * 2]);

                            Votee = Votee + vote.votes * vote.decay;
                        }
                    }
                }
            }
            output_field[(yi * sx + xi) * 4 + 0] += Votee[0][0];
            output_field[(yi * sx + xi) * 4 + 1] += Votee[0][1];
            output_field[(yi * sx + xi) * 4 + 2] += Votee[1][0];
            output_field[(yi * sx + xi) * 4 + 3] += Votee[1][1];
        }
    }
}

/// Create an empty field with a single stick tensor in the middle oriented along (x, y)
tira::field<float> StickTensor(size_t sx, size_t sy, float x, float y) {
    tira::field<float> S(std::vector<size_t>({ sx, sy, 2, 2 }));                        // allocate space for the field
    S = 0.0f;                                                                           // initialize the field to zeros

    size_t cx = sx / 2;                                             // calculate the center pixel
    size_t cy = sy / 2;

    S({ cx, cy, 0, 0 }) = x * x;
    S({ cx, cy, 1, 1 }) = y * y;
    S({ cx, cy, 1, 0 }) = x * y;
    S({ cx, cy, 0, 1 }) = y * x;
    return S;
}

int main(int argc, char *argv[])
{

    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("input", boost::program_options::value<std::string>(&in_inputname), "output filename for the coupled wave structure")
        ("output", boost::program_options::value<std::string>(&in_outputname)->default_value("tv.npy"), "optional image field corresponding to the tensors")
        ("sigma", boost::program_options::value<float>(&in_sigma)->default_value(5.0f), "order used to calculate the first derivative")
        ("window", boost::program_options::value<unsigned int>(&in_window), "window size (6 * sigma + 1 as default)")
        ("cuda", boost::program_options::value<int>(&in_cuda)->default_value(0), "cuda device index (-1 for CPU)")
        ("votefield", boost::program_options::value<std::vector<float> >(&in_votefield)->multitoken(), "generate a test field based on a 2D orientation")
        ("debug", "output debug information")
        ("help", "produce help message");
    boost::program_options::variables_map vm;

    boost::program_options::positional_options_description p;
    p.add("input", -1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

    boost::program_options::notify(vm);

    // if the user passes the help parameter, output command line details
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("debug")) debug = true;                    // activate debug output

    // calculate the window size if one isn't provided
    if (!vm.count("window")) in_window = int(6 * in_sigma + 1);


    // make sure that the selected CUDA device is valid, and switch to the CPU if it isn't
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

    tira::field<float> T;

    if (vm.count("votefield")) {
        if (in_votefield.size() == 1) in_votefield.push_back(0.0f);
        T = StickTensor(in_window, in_window, in_votefield[0], in_votefield[1]);
    }
    else {
        T.load_npy(in_inputname); // load the input tensor field
    }
    tira::field<float> Tr(T.shape());   // create a field to store the vote result

    // CPU IMPLEMENTATION
    if (in_cuda < 0) {
        cpuVote2D(T.data(), Tr.data(), T.shape()[0], T.shape()[1], in_sigma, in_window);
    }

    if (debug) T.save_npy("debug_input.npy");
    Tr.save_npy(in_outputname);

    return 0;
}