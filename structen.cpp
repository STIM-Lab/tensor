#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>

#include <tira/image.h>


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

	unsigned int d = 1;						// derivative to calculate
	unsigned int O = 2;
	std::string fname = "plane.bmp";

	tira::image<float> I(fname);
	/*tira::image<float> I(255, 255, 1);
	for (int i = 0; i < I.X(); i++) {
		for (int j = 0; j < I.Y(); j++) {
			I(i, j, 0) = j;
		}
	}*/
	tira::image<float> grey = I.channel(0);

	auto clock_start = std::chrono::system_clock::now();
	
	tira::image<float> D = grey.derivative(0, 1, 6);

	auto clock_now = std::chrono::system_clock::now();
	float currentTime = float(std::chrono::duration_cast <std::chrono::microseconds> (clock_now - clock_start).count());
	std::cout << "Elapsed Time: " << currentTime / 1000000 << " S \n";

	tira::image<unsigned char> color = D.cmap();
	color.save("test.bmp");

	/*std::vector<size_t> shape = {10};
	tira::field<float> T(shape);
	T({ 0 }) = 0;
	T({ 1 }) = 1;
	T({ 2 }) = 2;
	T({ 3 }) = 3;
	T({ 4 }) = 4;
	T({ 5 }) = 5;
	T({ 6 }) = 4;
	T({ 7 }) = 3;
	T({ 8 }) = 2;
	T({ 9 }) = 1;

	tira::image<float> D = T.derivative(0, 2, 4);
	//tira::image<float>* Dimg = dynamic_cast<tira::image<float>*>(D);
	//tira::image<unsigned char> color = D.cmap(0, 1);
	*/
}