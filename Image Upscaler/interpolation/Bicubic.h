#pragma once
#include "IInterpolator.h"

class Bicubic : public IInterpolator {
public:
	Pixel interpolate(Image& image, float x, float y) override;
};