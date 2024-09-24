#pragma once

#include <iomanip>
#include <filesystem>
#include <tira/volume.h>



namespace tira {

	template <class T>
	class volumetric : public volume<T> {

	protected:

		std::vector<double> _spacing;
		std::vector<double> _size;

		void init(size_t x, size_t y, size_t z, size_t c, std::vector<double> spacing) {
			volume<T>::init(x, y, z, c);
			_spacing = { spacing[0], spacing[1], spacing[2] };

			// _size corresponds to the total size of the volume (sx, sy, sz)
			_size = { x * spacing[0], y * spacing[1], z * spacing[2] };

		}
	public:
		volumetric() : volume<T>() {}

		volumetric(size_t x, size_t y, size_t z, size_t c = 1) {
			init(x, y, z, c);
		}

		volumetric(std::string filename) {
			volume<T>::load(filename);
		}

		~volumetric() {}

		inline size_t Z() const { return volume<T>::Z(); }
		inline size_t Y() const { return volume<T>::Y(); }
		inline size_t X() const { return volume<T>::X(); }
		inline size_t C() const { return volume<T>::C(); }

		inline double dx() const { return _spacing[0]; }
		inline double dy() const { return _spacing[1]; }
		inline double dz() const { return _spacing[2]; }

		inline double sx() const { return _size[0]; }
		inline double sy() const { return _size[1]; }
		inline double sz() const { return _size[2]; }

		void set_spacing(double x, double y, double z) {
			std::vector<double> spacing = { x, y, z };
			_spacing = spacing;
			std::vector<double> size = { _spacing[0] * volume<T>::X(), _spacing[1] * volume<T>::Y(), _spacing[2] * volume<T>::Z() };
			_size = size;
		}

		template<typename D = T>
		void load_npy(std::string filename) {
			volume<T>::template load_npy<D>(filename);
			std::vector<double> spacing = { 1.0, 1.0, 1.0 };
			_spacing = spacing;
			std::vector<double> size = { _spacing[0] * volume<T>::X(), _spacing[1] * volume<T>::Y(), _spacing[2] * volume<T>::Z() };
			_size = size;
		}
	};
}
