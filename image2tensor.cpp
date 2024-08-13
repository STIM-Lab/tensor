#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

#include <tira/image.h>
#include <tira/volume.h>

#include <boost/program_options.hpp>

std::string in_inputname;
std::string in_outputname;
unsigned int in_order;
unsigned int in_derivative;
float in_noise;
float in_sigma;
bool in_crop = false;


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
		("derivative", boost::program_options::value<unsigned int>(&in_derivative)->default_value(1), "output file storing the tensor field")
		("order", boost::program_options::value<unsigned int>(&in_order)->default_value(6), "order used to calculate the first derivative")
		("blur", boost::program_options::value<float>(&in_sigma)->default_value(0.0f), "sigma value for a Gaussian blur")
		("noise", boost::program_options::value<float>(&in_noise)->default_value(0.0f), "gaussian noise standard deviation added to the field")
		("crop", "crop the edges of the field to fit the finite difference window")
		("help", "produce help message")
		;
	boost::program_options::variables_map vm;

	boost::program_options::positional_options_description p;
	p.add("input", -1);
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
		tira::field<float> Dx = grey.derivative(1, in_derivative, in_order);			// calculate the derivative along the x axis	
		tira::field<float> Dy = grey.derivative(0, in_derivative, in_order);			// calculate the derivative along the y axis

		D.push_back(Dx);
		D.push_back(Dy);

		std::vector<size_t> field_shape = { D[0].shape()[0], D[0].shape()[1], (size_t)dim, (size_t)dim };
		tira::field<float> ST(field_shape);

		std::cout << "Generating tensor field...";
		// build the tensor field
		for (size_t x0 = 0; x0 < field_shape[0]; x0++) {
			for (size_t x1 = 0; x1 < field_shape[1]; x1++) {
				ST({ x0, x1, 0, 0 }) = D[0]({ x0, x1 }) * D[0]({ x0, x1 });
				ST({ x0, x1, 1, 1 }) = D[1]({ x0, x1 }) * D[1]({ x0, x1 });
				ST({ x0, x1, 0, 1 }) = D[0]({ x0, x1 }) * D[1]({ x0, x1 });
				ST({ x0, x1, 1, 0 }) = ST({ x0, x1, 0, 1 });
			}
		}
		std::cout << "done." << std::endl;

		if (in_sigma > 0) {
			std::cout << "Blurring tensor field (axis 0)...";
			size_t window = (int)(in_sigma + 1);
			float kernel_value = 1.0 / window;

			std::vector<size_t> kernel_size1 = {window, 1, 1, 1};
			tira::field<float> K1(kernel_size1);
			

			for (size_t yi = 0; yi < window; yi++) {
				//for (size_t xi = 0; xi < window; xi++) {
				K1({ yi, 0, 0, 0 }) = kernel_value;
				//K1({ yi, 0, 0, 1 }) = kernel_value;
				//K1({ yi, 0, 1, 0 }) = kernel_value;
				//K1({ yi, 0, 1, 1 }) = kernel_value;
				//}
			}
			ST = ST.convolve(K1);
			std::cout << "done." << std::endl;

			std::cout << "Blurring tensor field (axis 1)...";
			std::vector<size_t> kernel_size2 = {1, window, 1, 1};
			tira::field<float> K2(kernel_size2);

			for (size_t xi = 0; xi < window; xi++) {
				K2({ 0, xi, 0, 0 }) = kernel_value;
			}
			ST = ST.convolve(K2);

			std::cout << "done." << std::endl;
		}

		// add noise to the final tensor field
		if (in_noise != 0) {
			std::cout << "Adding noise...";
			std::random_device rd{};
			std::mt19937 gen{ rd() };
			std::normal_distribution d{ 0.0, (double)in_noise };

			for (size_t x0 = 0; x0 < field_shape[0]; x0++) {
				for (size_t x1 = 0; x1 < field_shape[1]; x1++) {
					ST({ x0, x1, 0, 0 }) = ST({ x0, x1, 0, 0 }) + abs((float)d(gen));
					ST({ x0, x1, 1, 1 }) = ST({ x0, x1, 1, 1 }) + abs((float)d(gen));
					ST({ x0, x1, 0, 1 }) = ST({ x0, x1, 0, 1 }) + (float)d(gen);
					ST({ x0, x1, 1, 0 }) = ST({ x0, x1, 0, 1 });
				}
			}
			std::cout << "done." << std::endl;
		}

		if (in_crop) {
			std::cout << "Cropping...";
			size_t window_width = (in_order + in_derivative) / 2;
			std::vector<size_t> min_crop = { window_width, window_width, 0, 0 };
			std::vector<size_t> max_crop = { ST.shape()[0] - window_width, ST.shape()[1] - window_width, 2, 2};
			tira::field<float> C = ST.crop(min_crop, max_crop);
			C.save_npy(in_outputname);
			std::cout << "done." << std::endl;
		}
		else
			ST.save_npy(in_outputname);
	}
	else if (dim == 3) {
		std::cout << "Doesn't support 3D images yet" << std::endl;
		return 0;
		/*tira::volume<float> I(in_inputname);
		//tira::volume<float> grey = I.channel(0);
		tira::field<float> Dx = I.derivative(2, in_derivative, in_order);
		tira::field<float> Dy = I.derivative(1, in_derivative, in_order);
		tira::field<float> Dz = I.derivative(0, in_derivative, in_order);

		D.push_back(Dx);
		D.push_back(Dy);
		D.push_back(Dz);
		*/
	}

	return 0;
	
}