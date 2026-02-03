#pragma once
#include "../core/Image.h"

class IInterpolator {	
	public:
		virtual Pixel interpolate(Image& image, float x, float y) = 0;
};