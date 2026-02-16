#include "Bicubic.h"

/**
 * Given a position (x, y) in the source image, this function computes the interpolated pixel value using bicubic interpolation.
 * Sample a 4×4 neighborhood at offsets m,n ∈ {-1, 0, 1, 2}:
 * f(x, y) = Σ(i=-1..2) Σ(j=-1..2) P(ix+j, iy+i) · W(dx - j) · W(dy - i)
 * where P(i, j) is the pixel value at (i, j) and W(t) is the Keys cubic kernel
 */
Pixel Bicubic::interpolate(Image& img, float x, float y) {
	int ix = static_cast<int>(std::floor(x));
	int iy = static_cast<int>(std::floor(y));
	int img_width = img.getWidth();
	int img_height = img.getHeight();
	double dx = x - ix;
	double dy = y - iy;
	double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;

	for (int i = -1; i <= 2; i++) {
		double wy = cubic_weight(dy - i);
		int sy = std::clamp(iy + i, 0, img_height - 1);
		for (int j = -1; j <= 2; j++) {
			double wx = cubic_weight(dx - j);
			int sx = std::clamp(ix + j, 0, img_width - 1);

			Pixel p = img.at(sx, sy);
			double weight = wx * wy;
			sum_r += p.r * weight;
			sum_g += p.g * weight;
			sum_b += p.b * weight;
		}
	}

	Pixel result;
	result.r = static_cast<unsigned char>(std::clamp(sum_r, 0.0, 255.0));
	result.g = static_cast<unsigned char>(std::clamp(sum_g, 0.0, 255.0));
	result.b = static_cast<unsigned char>(std::clamp(sum_b, 0.0, 255.0));

	return result;
}

/**
 *  W(t) is cubic kernel:
 *
 *  W(t) = (a+2)|t|³ - (a+3)|t|² + 1,        |t| ≤ 1
 *  W(t) = a|t|³ - 5a|t|² + 8a|t| - 4a,  1 < |t| < 2
 *  W(t) = 0,                                 t| ≥ 2.
 */
double Bicubic::cubic_weight(double x) {
	constexpr double a = -0.5; // Catmull-Rom spline
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