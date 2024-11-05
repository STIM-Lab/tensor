#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

#include <glm/glm.hpp>

#include <tira/image.h>
#include <tira/volume.h>

#include <boost/program_options.hpp>

std::string in_inputname;
std::string in_outputname;
unsigned int in_order;
float in_noise;
float in_sigma;
float in_blur;
bool in_crop = false;
int in_device;						// cuda device

glm::mat2* cudaGaussianBlur(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
	unsigned int& out_width, unsigned int& out_height, int deviceID = 0);
float* cudaGaussianBlur(float* source, unsigned int width, unsigned int height, float sigma,
	unsigned int& out_width, unsigned int& out_height, int deviceID = 0);

float* cudaEigenvalues2(float* tensors, unsigned int n, int device);

/// <summary>
/// Calculate the finite difference coefficients for a set of sample points.
/// </summary>
/// <param name="derivative">Derivative value provided by these coefficients</param>
/// <param name="samples">Sample points used to evaluate the derivative</param>
/// <returns>Set of coefficients applied to each sample point.</returns>
template<typename T>
Eigen::VectorX<T> finite_difference_coefficients(unsigned int derivative, Eigen::VectorX<T> samples) {

	unsigned int N = samples.size();

	Eigen::MatrixX<T> S(N, N);
	for (unsigned int ri = 0; ri < N; ri++) {
		for (unsigned int ci = 0; ci < N; ci++) {
			S(ri, ci) = pow(samples[ci], ri);
		}
	}

	Eigen::VectorX<T> b = Eigen::VectorX<T>::Zero(N);
	b(derivative) = tgamma(derivative + 1);

	return S.colPivHouseholderQr().solve(b);
}

/// <summary>
/// Calculate a matrix of finite difference coefficients, where each row represents a different center position.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="derivative"></param>
/// <param name="order"></param>
/// <returns></returns>
template<typename T>
std::vector< std::vector<T> > finite_difference_coefficients(unsigned int derivative, unsigned int order) {

	unsigned int N = order + 1;					// calculate the number of samples required to achieve the desired order

	std::vector< std::vector<T> > Coefficients;

	Eigen::VectorX<T> Samples(N);				// allocate a vector that will be used to store sample points

	for (int ri = 0; ri < N; ri++) {			// for each shifted sample position
		for (int ci = 0; ci < N; ci++) {		// calculate the point for each sample
			Samples(ci) = -ri + ci;				// store that point in the Samples vector
		}
		std::vector<T> c(N);
		Eigen::Map< Eigen::VectorX<T> >(&c[0], N) = finite_difference_coefficients<T>(derivative, Samples);
		Coefficients.push_back(c);
	}
	return Coefficients;
}


int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("input", boost::program_options::value<std::string>(&in_inputname), "input image")
		("output", boost::program_options::value<std::string>(&in_outputname)->default_value("out.npy"), "output file storing the tensor field")
		("hessian", "calculate the Hessian")
		("negatives", "sets positive eigenvalues to zero")
		("positives", "sets negative eigenvalues to zero")
		("order", boost::program_options::value<unsigned int>(&in_order)->default_value(2), "order used to calculate the derivative")
		("blur", boost::program_options::value<float>(&in_blur)->default_value(0.0f), "sigma value for blurring the input image")
		("sigma", boost::program_options::value<float>(&in_sigma)->default_value(2.0f), "sigma value for the tensor field blur")
		("cuda", boost::program_options::value<int>(&in_device)->default_value(-1), "cuda device number (-1 for CPU)")
		//("noise", boost::program_options::value<float>(&in_noise)->default_value(0.0f), "gaussian noise standard deviation added to the field")
		
		("crop", "crop the edges of the field to fit the finite difference window")
		("help", "produce help message")
		;
	boost::program_options::variables_map vm;

	boost::program_options::positional_options_description p;
	p.add("input", 1);
	p.add("output", 1);
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("crop")) in_crop = true;

	int dim = 3;															// number of dimensions
	std::vector< tira::field<float> > D;									// vector stores the derivatives

	if (in_inputname.rfind("*") == std::string::npos)
		dim = 2;

	if (dim == 2) {
		tira::image<float> I(in_inputname);												// load the input image
		tira::image<float> grey = I.channel(0);											// get the first channel if this is a color image
		grey = grey / 255.0f;

		if (in_blur > 0) {
			unsigned int raw_width;
			unsigned int raw_height;
			float* raw_field = cudaGaussianBlur(grey.data(), grey.X(), grey.Y(), in_blur, raw_width, raw_height);
			grey = tira::image<float>(raw_field, raw_width, raw_height);
			free(raw_field);
			grey.cmap().save("grey.bmp");
		}
		
		grey = grey.border(in_order, 0);

		tira::image<glm::mat2> T;
		std::vector<size_t> field_shape;

		// Evaluate the Hessian at each pixel
		if (vm.count("hessian")) {

			tira::image<float> D2x2 = grey.derivative(1, 2, in_order);			// calculate the derivative along the x axis
			tira::image<float> D2y2 = grey.derivative(0, 2, in_order);			// calculate the derivative along the y axis
			tira::image<float> Dx = grey.derivative(1, 1, in_order);
			tira::image<float> Dxy = Dx.derivative(0, 1, in_order);


			//field_shape = { Dx, D[0].shape()[1], (size_t)dim, (size_t)dim };
			T = tira::image<glm::mat2>(Dx.X(), Dx.Y());

			std::cout << "Generating tensor field...";
			// build the tensor field
			for (size_t yi = 0; yi < T.Y(); yi++) {
				for (size_t xi = 0; xi < T.X(); xi++) {
					T(xi, yi)[0][0] = D2x2(xi, yi);
					T(xi, yi)[1][1] = D2y2(xi, yi);
					T(xi, yi)[0][1] = Dxy(xi, yi);
					T(xi, yi)[1][0] = Dxy(xi, yi);
				}
			}
			std::cout << "done." << std::endl;

		}
		// Evaluate the structure tensor at each pixel
		else {
			tira::image<float> Dx = grey.derivative(1, 1, in_order);			// calculate the derivative along the x axis
			tira::image<float> Dy = grey.derivative(0, 1, in_order);			// calculate the derivative along the y axis

			Dx = Dx.border_remove(in_order);
			Dy = Dy.border_remove(in_order);
			D.push_back(Dx);
			D.push_back(Dy);

			T = tira::image<glm::mat2>(Dx.X(), Dx.Y());

			std::cout << "Generating tensor field...";
			// build the tensor field
			for (size_t yi = 0; yi < T.Y(); yi++) {
				for (size_t xi = 0; xi < T.X(); xi++) {
					T(xi, yi)[0][0] = std::pow(Dx(xi, yi), 2);
					T(xi, yi)[1][1] = std::pow(Dy(xi, yi), 2);
					T(xi, yi)[0][1] = Dx(xi, yi) * Dy(xi, yi);
					T(xi, yi)[1][0] = T(xi, yi)[0][1];
				}
			}
			std::cout << "done." << std::endl;
		}

		if (in_sigma > 0) {
			unsigned int raw_width;
			unsigned int raw_height;
			glm::mat2* raw_field = cudaGaussianBlur(T.data(), T.X(), T.Y(), in_sigma, raw_width, raw_height);
			T = tira::image<glm::mat2>(raw_field, raw_width, raw_height);
			free(raw_field);
		}

		if(vm.count("negatives") || vm.count("positives")) {
			bool keep_positives = true;
			if(vm.count("negatives")) keep_positives = false;

			float* evals = cudaEigenvalues2((float*)T.data(), T.X() * T.Y(), in_device);
			for (size_t yi = 0; yi < T.Y(); yi++) {
				for (size_t xi = 0; xi < T.X(); xi++) {
					size_t i = yi * T.X() + xi;
					if(evals[i * 2 + 1] < 0) {
						if(keep_positives) {
							T(xi, yi) = 0;
						}
					}
					else {
						if(!keep_positives) {
							T(xi, yi) = 0;
						}
					}
				}
			}
		}

		// save the tensor field to an output file
		tira::field<float> Tout({ T.shape()[0], T.shape()[1], 2, 2 }, (float*)T.data());
		Tout.save_npy(in_outputname);
	}

	else if (dim == 3) {
		std::cout << "Loading 3D images..." << std::endl;

		tira::volume<float> V(in_inputname);									// load input volume
		tira::volume<float> grey = V.channel(0);								// get the first channel if this is a color image

		grey = grey.border(in_order, 0);
		tira::volume<glm::mat3> T;
		std::vector<size_t> field_shape;

		// Evaluate the structure tensor at each pixel
		tira::volume<float> Dx = grey.gradient_dx();			// calculate the derivative along the x axis
		tira::volume<float> Dy = grey.gradient_dy();			// calculate the derivative along the y axis
		tira::volume<float> Dz = grey.gradient_dz();			// calculate the derivative along the y axis

		T = tira::volume<glm::mat3>(Dx.X(), Dx.Y(), Dx.Z());

		std::cout << "Generating tensor field...";
		// build the tensor field
		for (size_t zi = 0; zi < T.Z(); zi++) {
			for (size_t yi = 0; yi < T.Y(); yi++) {
				for (size_t xi = 0; xi < T.X(); xi++) {
					T(xi, yi, zi)[0][0] = std::pow(Dx(xi, yi, zi), 2);					// Dxx
					T(xi, yi, zi)[1][1] = std::pow(Dy(xi, yi, zi), 2);					// Dyy
					T(xi, yi, zi)[2][2] = std::pow(Dz(xi, yi, zi), 2);					// Dyy
					T(xi, yi, zi)[0][1] = Dx(xi, yi, zi) * Dy(xi, yi, zi);				// Dxy
					T(xi, yi, zi)[1][0] = T(xi, yi, zi)[0][1];							// Dyx
					T(xi, yi, zi)[0][2] = Dx(xi, yi, zi) * Dz(xi, yi, zi);				// Dxz
					T(xi, yi, zi)[2][0] = T(xi, yi, zi)[0][2];							// Dzx
					T(xi, yi, zi)[1][2] = Dy(xi, yi, zi) * Dz(xi, yi, zi);				// Dyz
					T(xi, yi, zi)[2][1] = T(xi, yi, zi)[1][2];							// Dzy
				}
			}
		}
		std::cout << " done." << std::endl;

		// save the tensor field to an output file
		tira::field<float> Tout({ T.shape()[0], T.shape()[1], T.shape()[2], 3, 3 }, (float*)T.data());
		Tout.save_npy(in_outputname);

	}

	return 0;
	
}