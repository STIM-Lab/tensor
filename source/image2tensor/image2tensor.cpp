//#include <Eigen/Dense>
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
float in_dx, in_dy, in_dz;
std::vector<unsigned int> in_crop_loc, in_crop_len;
int in_device;						// cuda device

glm::mat2* GaussianBlur2D(glm::mat2* source, unsigned int width, unsigned int height, float sigma,
	unsigned int& out_width, unsigned int& out_height, int deviceID);

float* GaussianBlur2D(float* source, unsigned int width, unsigned int height, float sigma,
	unsigned int& out_width, unsigned int& out_height, int deviceID);

glm::mat3* GaussianBlur3D(glm::mat3* source, unsigned int width, unsigned int height, unsigned int depth,
	float sigma_w, float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height, 
	unsigned int& out_depth, int deviceID);

float* GaussianBlur3D(float* source, unsigned int width, unsigned int height, unsigned int depth, 
	float sigma_w, float sigma_h, float sigma_d, unsigned int& out_width, unsigned int& out_height, 
	unsigned int& out_depth, int deviceID);

float* EigenValues2(float* tensors, unsigned int n, int device);


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
		("cuda", boost::program_options::value<int>(&in_device)->default_value(0), "cuda device number (-1 for CPU)")
		("dx", boost::program_options::value<float>(&in_dx)->default_value(1.0f), "size of X pixels")
		("dy", boost::program_options::value<float>(&in_dy)->default_value(1.0f), "size of Y pixels")
		("dz", boost::program_options::value<float>(&in_dz)->default_value(1.0f), "size of Z pixels")

		("crop_loc", boost::program_options::value<std::vector<unsigned int>>()->multitoken(), "crop location (two values -> 2D (w, h), three values -> 3D (w, h, d)")
		("crop_len", boost::program_options::value<std::vector<unsigned int>>()->multitoken(), "crop length (width, height, depth")
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

	int dim = 2;															// number of dimensions
	std::vector< tira::field<float> > D;									// vector stores the derivatives

	// If the user specify a wildcard or a .npy file, assume this is a 3D volume
	if (in_inputname.rfind("*") != std::string::npos || in_inputname.ends_with(".npy"))
		dim = 3;

	/**
	 * If the input image is 2D:
	 * (1) the image is loaded and cropped based on user-specified parameters
	 * (2) the input may also undergo a preliminary blur to remove noise
	 * (3) a border is added to ensure that the output image is the same size as the input image
	 * The tensor field is then computed based on whether the user requests the structure tensor (default) or the Hessian.
	*/
	if (dim == 2) {
		tira::image<float> I(in_inputname);												// load the input image
		tira::image<float> grey = I.channel(0);												// get the first channel if this is a color image
		grey = grey / 255.0f;

		/////////////////////////////////////////////////////////////////////
		// Crop the image if necessary
		/////////////////////////////////////////////////////////////////////
		std::vector<unsigned int> crop_loc, crop_len;
		if (!vm["crop_loc"].empty() && (crop_loc = vm["crop_loc"].as<std::vector<unsigned int> >()).size() == 2) {
			if (!vm["crop_len"].empty() && (crop_len = vm["crop_len"].as<std::vector<unsigned int> >()).size() == 2) {
				grey = grey.crop(crop_loc[0], crop_loc[1], crop_len[0], crop_len[1]);
			}
			else {										// no crop length specified -> crop from location to the end of the image
				grey = grey.crop(crop_loc[0], crop_loc[1], grey.X() - crop_loc[0], grey.Y() - crop_loc[1]);
			}
		}
		else if (!vm["crop_loc"].empty())
			std::cout << "Wrong number of inputs for crop location, input two." << std::endl;

		/////////////////////////////////////////////////////////////////////
		/// Apply an initial blur if the user reqests it
		/////////////////////////////////////////////////////////////////////
		if (in_blur > 0) {
			unsigned int raw_width;
			unsigned int raw_height;
			float* raw_field = GaussianBlur2D(grey.Data(), grey.X(), grey.Y(), in_blur, raw_width, raw_height, in_device);
			grey = tira::image<float>(raw_field, raw_width, raw_height);
			free(raw_field);
			grey.cmap().save("grey.bmp");
		}

		/////////////////////////////////////////////////////////////////////
		/// Add a border around the image to ensure that the output is the same size
		/////////////////////////////////////////////////////////////////////
		grey = grey.border(in_order, 0);

		/////////////////////////////////////////////////////////////////////
		/// Generate the tensor field
		/////////////////////////////////////////////////////////////////////
		tira::image<glm::mat2> T;
		std::vector<size_t> field_shape;

		/**
		 * If the user specifies that the Hessian matrix should be calculated, compute the first and second order
		 * derivatives and build the tensor field.
		*/
		if (vm.count("hessian")) {

			tira::image<float> D2x2 = grey.Derivative(1, 2, in_order);			// calculate the derivative along the x axis
			tira::image<float> D2y2 = grey.Derivative(0, 2, in_order);			// calculate the derivative along the y axis
			tira::image<float> Dx = grey.Derivative(1, 1, in_order);
			tira::image<float> Dxy = Dx.Derivative(0, 1, in_order);


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
		/**
		 * Otherwise calculate the structure tensor (which is the default)
		*/
		else {
			tira::image<float> Dx = grey.Derivative(1, 1, in_order);			// calculate the derivative along the x axis
			tira::image<float> Dy = grey.Derivative(0, 1, in_order);			// calculate the derivative along the y axis

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

		/**
		 * Blur the output field based on user parameters. Generally this will be a small blur value to remove
		 * noise introduces by taking the derivative using finite difference methods.
		 */
		if (in_sigma > 0) {
			unsigned int raw_width;
			unsigned int raw_height;
			glm::mat2* raw_field = GaussianBlur2D(T.Data(), T.X(), T.Y(), in_sigma, raw_width, raw_height, in_device);
			T = tira::image<glm::mat2>(raw_field, raw_width, raw_height);
			free(raw_field);
		}

		/**
		 * The user can specify whether or not negative or positive eigenvalues are kept for the Hessian field.
		 * If one of the values are specified, the others are set to zero.
		 */
		if(vm.count("negatives") || vm.count("positives")) {
			bool keep_positives = true;
			if(vm.count("negatives")) keep_positives = false;

			float* evals = EigenValues2((float*)T.Data(), T.X() * T.Y(), in_device);
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
			delete[] evals;
		}

		// save the tensor field to an output file
		tira::field<float> Tout({ T.Shape()[0], T.Shape()[1], 2, 2 }, (float*)T.Data());
		Tout.SaveNpy(in_outputname);
	}

	else if (dim == 3) {
		std::cout << "Loading 3D images..." << std::endl;

		tira::volume<float> V(in_inputname);									// load input volume
		tira::volume<float> grey = V.channel(0);								// get the first channel if this is a colored volume

		// Crop the volume from the specified location
		std::vector<unsigned int> crop_loc, crop_len;
		if (!vm["crop_loc"].empty() && (crop_loc = vm["crop_loc"].as<std::vector<unsigned int> >()).size() == 3) {
			if (!vm["crop_len"].empty() && (crop_len = vm["crop_len"].as<std::vector<unsigned int> >()).size() == 3) {
				grey = grey.crop(crop_loc[0], crop_loc[1], crop_loc[2], crop_len[0], crop_len[1], crop_len[2]);
			}
			else {										// no crop length specified -> crop from location to the end of the volume
				grey = grey.crop(crop_loc[0], crop_loc[1], crop_loc[2], grey.X() - crop_loc[0], grey.Y() - crop_loc[1], grey.Z() - crop_loc[2]);
			}
		}
		else if (!vm["crop_loc"].empty())
			std::cout << "Wrong number of inputs for crop location, input three." << std::endl;

		// Apply an initial blur if the user requests it
		if (in_blur > 0) {
			unsigned int raw_width;
			unsigned int raw_height;
			unsigned int raw_depth;
			float* raw_field = GaussianBlur3D(grey.Data(), grey.X(), grey.Y(), grey.Z(), in_blur/in_dx, 
				in_blur/in_dy, in_blur/in_dz, raw_width, raw_height, raw_depth, in_device);
			grey = tira::volume<float>(raw_field, raw_width, raw_height, raw_depth);
			free(raw_field);
		}

		grey = grey.border(in_order, 0);
		// Generate the tensor field
		tira::volume<glm::mat3> T;
		std::vector<size_t> field_shape;

		// Calculate the structure tensor
		tira::volume<float> Dx = grey.derivative(2, 1, in_order);			// calculate the derivative along the x axis
		tira::volume<float> Dy = grey.derivative(1, 1, in_order);			// calculate the derivative along the y axis
		tira::volume<float> Dz = grey.derivative(0, 1, in_order);			// calculate the derivative along the z axis
		Dx = Dx.border_remove(in_order);
		Dy = Dy.border_remove(in_order);
		Dz = Dz.border_remove(in_order);

		T = tira::volume<glm::mat3>(Dx.X(), Dx.Y(), Dx.Z());
		std::cout << "Generating tensor field..." << std::endl;
		// build the tensor field
		for (size_t zi = 0; zi < T.Z(); zi++) {
			for (size_t yi = 0; yi < T.Y(); yi++) {
				for (size_t xi = 0; xi < T.X(); xi++) {
					T(xi, yi, zi)[0][0] = Dx(xi, yi, zi) * Dx(xi, yi, zi);				// Dxx
					T(xi, yi, zi)[1][1] = Dy(xi, yi, zi) * Dy(xi, yi, zi);				// Dyy
					T(xi, yi, zi)[2][2] = Dz(xi, yi, zi) * Dz(xi, yi, zi);				// Dzz
					T(xi, yi, zi)[0][1] = Dx(xi, yi, zi) * Dy(xi, yi, zi);				// Dxy
					T(xi, yi, zi)[1][0] = T(xi, yi, zi)[0][1];							// Dyx
					T(xi, yi, zi)[0][2] = Dx(xi, yi, zi) * Dz(xi, yi, zi);				// Dxz
					T(xi, yi, zi)[2][0] = T(xi, yi, zi)[0][2];							// Dzx
					T(xi, yi, zi)[1][2] = Dy(xi, yi, zi) * Dz(xi, yi, zi);				// Dyz
					T(xi, yi, zi)[2][1] = T(xi, yi, zi)[1][2];							// Dzy
				}
			}
		}
		std::cout << "Done" << std::endl;

		if (in_sigma > 0) {
			unsigned int raw_width;
			unsigned int raw_height;
			unsigned int raw_depth;
			glm::mat3* raw_field = GaussianBlur3D(T.Data(), T.X(), T.Y(), T.Z(), in_sigma / in_dx,
				in_sigma / in_dy, in_sigma / in_dz, raw_width, raw_height, raw_depth, in_device);
			T = tira::volume<glm::mat3>(raw_field, raw_width, raw_height, raw_depth);
			free(raw_field);
		}

		// save the tensor field to an output file
		tira::field<float> Tout({ T.Shape()[0], T.Shape()[1], T.Shape()[2], 3, 3 }, (float*)T.Data());
		Tout.SaveNpy(in_outputname);
	}

	return 0;
	
}