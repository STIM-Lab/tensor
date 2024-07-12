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
		("noise", boost::program_options::value<float>(&in_noise)->default_value(0.0f), "gaussian noise standard deviation added to the field")
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
	}
	else if (dim == 3) {
		tira::volume<float> I(in_inputname);
		//tira::volume<float> grey = I.channel(0);
		tira::field<float> Dx = I.derivative(2, in_derivative, in_order);
		tira::field<float> Dy = I.derivative(1, in_derivative, in_order);
		tira::field<float> Dz = I.derivative(0, in_derivative, in_order);

		D.push_back(Dx);
		D.push_back(Dy);
		D.push_back(Dz);
	}

	std::vector<size_t> tensor_shape = D[0].shape();						// get the shape of the tensor field
	tensor_shape[dim] = dim;												// push two additional dimensions representing the square matrix
	tensor_shape.push_back(dim);

	tira::field<float> ST(tensor_shape);

	std::vector<size_t> st_coord, i_coord;
	for (tira::field<float>::iterator i = D[0].begin(); i != D[0].end(); i++) {
		i_coord = i.coord();
		st_coord = i_coord;
		st_coord.resize(dim + 2);
		for (unsigned int d0 = 0; d0 < dim; d0++) {
			st_coord[dim + 0] = d0;
			for (unsigned int d1 = 0; d1 < dim; d1++) {
				st_coord[dim + 1] = d1;
				ST(st_coord) = D[d0](i_coord) * D[d1](i_coord);
			}
		}
	}

	if (in_noise > 0) {
		std::random_device rd{};
		std::mt19937 gen{ rd() };

		// values near the mean are the most likely
		// standard deviation affects the dispersion of generated values from the mean
		std::normal_distribution d{ 0.0, (double)in_noise };



		for (tira::volume<float>::iterator i = ST.begin(); i != ST.end(); i++)
			*i = *i + (float)d(gen);
	}

	ST.save_npy(in_outputname);
}