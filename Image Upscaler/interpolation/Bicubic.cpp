#include "Bicubic.h"

Pixel Bicubic::interpolate(Image& img, float x, float y) {
	return Pixel{ 0, 0, 0 };
}

double Bicubic::cubic_weight(double x) {
	constexpr double a = -0.5;
	double abs_x = std::abs(x);
	if (abs_x <= 1.0) {
		return (a + 2.0) * abs_x * abs_x * abs_x
			    - (a + 3.0) * abs_x * abs_x
			    + 1.0;
	}
	else if (abs_x < 2.0) {
		return a * abs_x * abs_x * abs_x
				- 5.0 * a * abs_x * abs_x
				+ 8.0 * a * abs_x
				- 4.0 * a;
	}
	return 0.0;
}