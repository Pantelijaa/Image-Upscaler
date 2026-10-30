#pragma once
#include "IInterpolator.h"

class Bilinear : public IInterpolator {
	public:
		int radius() const override { return 1; }
		double weight(double t) const override;
		Pixel interpolate(Image& image, float x, float y) override;
};
