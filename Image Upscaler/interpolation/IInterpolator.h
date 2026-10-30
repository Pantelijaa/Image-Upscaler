#pragma once
#include "../core/Image.h"

class IInterpolator {
	public:
		virtual ~IInterpolator() = default;
		virtual int radius() const = 0;
		virtual double weight(double t) const = 0;
		virtual Pixel interpolate(Image& image, float x, float y) = 0;
};
