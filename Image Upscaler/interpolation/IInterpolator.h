#pragma once
#include "../core/Image.h"

class Interpolator {	
	public:
		virtual Pixel interpolate(const Image& image, float x, float y) = 0;
};