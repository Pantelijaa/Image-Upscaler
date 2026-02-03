#pragma once
#include "IInterpolator.h"

class Bilinear : public IInterpolator {
	public:
		Pixel interpolate(Image& image, float x, float y) override;
};