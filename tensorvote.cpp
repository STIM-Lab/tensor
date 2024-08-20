#include <boost/program_options.hpp>

#include <tira/field.h>
#include <tira/image.h>
#include <glm/glm.hpp>
#include <chrono>

#include "tensorvote.cuh"
glm::vec2 Eigenvalues2D(glm::mat2 T);
glm::vec2 Eigenvector2D(glm::mat2 T, glm::vec2 lambdas, unsigned int index = 1);
void cpuEigendecomposition(float* input_field, float* eigenvectors, float* eigenvalues, unsigned int sx, unsigned int sy);
VoteContribution StickVote(float u, float v, float sigma, float* eigenvectors);
VoteContribution PlateVote(float u, float v, float sigma);
void cudaVote2D(float* input_field, float* output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w, unsigned int device, bool PLATE = true, bool time = false);


#include <tira/field.h>
#include <tira/image.h>

std::string in_inputname;
std::string in_outputname;
float in_sigma;
unsigned int in_window;
int in_cuda;
std::vector<float> in_votefield;
bool debug = false;
bool PLATE = true;
bool in_time = false;

// timing variables
float t_voting;
float t_device2host;
float t_host2device;
float t_eigendecomposition;
float t_total;
float t_devicealloc;
float t_devicefree;
float t_deviceprops;


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

void cpuVote2D(float *input_field, float *output_field, unsigned int sx, unsigned int sy, float sigma, unsigned int w, bool PLATE = true, bool time = false) {
    std::vector<float> V(sx * sy * 2);                          // allocate space for the eigenvectors
    std::vector<float> L(sx * sy * 2);                          // allocate space for the eigenvalues

    int hw = (int)(w / 2);                                      // calculate the half window size

    auto start = std::chrono::high_resolution_clock::now();
    cpuEigendecomposition(input_field, &V[0], &L[0], sx, sy);   // calculate the eigendecomposition of the entire field
    auto end = std::chrono::high_resolution_clock::now();
    t_eigendecomposition = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (debug) {
        save_field(&L[0], sx, sy, 2, "debug_eigenvalues.npy");
        save_field(&V[0], sx, sy, 2, "debug_eigenvector.npy");
    }
    std::vector<float> debug_decay;
    if (debug) {
        debug_decay = std::vector<float>(sx * sy);
    }

    float scale;
    int xr, yr;                                                 // x and y coordinates within the window
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int yi = 0; yi < sy; yi++) {                  // for each pixel in the image
        for (unsigned int xi = 0; xi < sx; xi++) {

            glm::mat2 receiver(0.0f);                           // initialize a receiver tensor to zero
            float total_decay = 0.0f;
            

            for (int v = -hw; v < hw; v++) {                    // for each pixel in the window
                yr = yi + v;
                if (yr >= 0 && yr < sy) {                       // if the pixel is inside the image (along the y axis)
                    for (int u = -hw; u < hw; u++) {
                        
                        xr = xi + u;
                        if (xr >= 0 && xr < sx) {               // if the pixel is inside the image (along the x axis)
                                                                // calculate the saliency (vote contribution)
                            VoteContribution vote = StickVote(
                                u,
                                v,
                                sigma,
                                //&L[(yr * sx + xr) * 2],
                                &V[(yr * sx + xr) * 2]
                            );
                            scale = L[(yr * sx + xr) * 2 + 1] - L[(yr * sx + xr) * 2 + 0];
                            receiver = receiver + scale * vote.votes * vote.decay;

                            if (PLATE) {                        // apply the plate vote
                                vote = PlateVote(u, v, sigma);
                                scale = L[(yr * sx + xr) * 2 + 0];
                                receiver = receiver + scale * vote.votes * vote.decay;
                            }

                            if (debug) {
                                total_decay += vote.decay;
                            }
                        }
                    }
                }
            }
            output_field[(yi * sx + xi) * 4 + 0] += receiver[0][0];
            output_field[(yi * sx + xi) * 4 + 1] += receiver[0][1];
            output_field[(yi * sx + xi) * 4 + 2] += receiver[1][0];
            output_field[(yi * sx + xi) * 4 + 3] += receiver[1][1];
            if (debug) {
                debug_decay[yi * sx + xi] = total_decay;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();

    t_voting = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    //if (time) {
    //    std::cout << "tensor vote duration: " << duration.count() << "ms" << std::endl;
    //}

    if(debug)
        save_field(&debug_decay[0], sx, sy, 1, "debug_decay.npy");
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

int main(int argc, char *argv[]) {

    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("input", boost::program_options::value<std::string>(&in_inputname), "output filename for the coupled wave structure")
        ("output", boost::program_options::value<std::string>(&in_outputname)->default_value("result.npy"), "optional image field corresponding to the tensors")
        ("sigma", boost::program_options::value<float>(&in_sigma)->default_value(5.0f), "order used to calculate the first derivative")
        ("window", boost::program_options::value<unsigned int>(&in_window), "window size (6 * sigma + 1 as default)")
        ("cuda", boost::program_options::value<int>(&in_cuda)->default_value(0), "cuda device index (-1 for CPU)")
        ("votefield", boost::program_options::value<std::vector<float> >(&in_votefield)->multitoken(), "generate a test field based on a 2D orientation")
        ("time", "times the code")
        ("stick", "stick voting only")
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
    if (vm.count("stick")) PLATE = false;                     // if only stick voting is requested, set the boolean flag
    if (vm.count("time")) in_time = true;

    // calculate the window size if one isn't provided
    if (!vm.count("window")) in_window = int(6 * in_sigma + 1);

    tira::field<float> T;

    if (vm.count("votefield")) {
        if (in_votefield.size() == 1) in_votefield.push_back(0.0f);
        T = StickTensor(in_window, in_window, in_votefield[0], in_votefield[1]);
    }
    else {
        T.load_npy(in_inputname); // load the input tensor field
    }
    tira::field<float> Tr(T.shape());   // create a field to store the vote result

    auto start = std::chrono::high_resolution_clock::now();
    // CPU IMPLEMENTATION
    if (in_cuda < 0) {
        cpuVote2D(T.data(), Tr.data(), T.shape()[0], T.shape()[1], in_sigma, in_window, PLATE, in_time);
    }
    else {
        cudaVote2D(T.data(), Tr.data(), T.shape()[0], T.shape()[1], in_sigma, in_window, in_cuda, PLATE, in_time);
    }
    auto end = std::chrono::high_resolution_clock::now();
    t_total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (in_time) {
        std::cout << "Eigendecomposition:  " << t_eigendecomposition << " ms" << std::endl;
        std::cout << "Voting: " << t_voting << " ms" << std::endl;

        if (in_cuda >= 0) {
            std::cout << "cudaMemcpy (H->D):  " << t_host2device << " ms" << std::endl;
            std::cout << "cudaMemcpy (D->H):  " << t_device2host << " ms" << std::endl;
            std::cout << "cudaMalloc: " << t_devicealloc << " ms" << std::endl;
            std::cout << "cudaFree: " << t_devicefree << " ms" << std::endl;
            std::cout << "cudaDeviceProps: " << t_deviceprops << " ms" << std::endl;
        }
        std::cout << "Total: " << t_total << " ms" << std::endl;
    }


    if (debug) T.save_npy("debug_input.npy");
    Tr.save_npy(in_outputname);

    return 0;
}