#pragma once
#include "IInterpolator.h"

class Bicubic : public IInterpolator {
public:
	int radius() const override { return 2; }
	double weight(double t) const override { return cubic_weight(t); }
	Pixel interpolate(Image& image, float x, float y) override;
private:
	static double cubic_weight(double x);
};
