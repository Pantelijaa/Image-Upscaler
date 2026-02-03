#pragma once

#include "Image.h"

class Scaler {
	public:
		static Image upscale(const Image& src, int nw, int nh, Interpolator* it);
}