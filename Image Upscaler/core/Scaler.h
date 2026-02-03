#pragma once

#include "Image.h"
#include "../interpolation/IInterpolator.h"

class Scaler {
public:
	static Image upscale(Image& src, int nw, int nh, IInterpolator& it);
};