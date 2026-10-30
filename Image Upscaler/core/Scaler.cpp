#include "Scaler.h"

Image Scaler::upscale(Image& src, int nw, int nh, IInterpolator& it) {
	Image dst(nw, nh);
	
	float xr = static_cast<float>(src.getWidth()) / nw;
	float yr = static_cast<float>(src.getHeight()) / nh;

	for (int y = 0; y < nh; ++y) {
		for (int x = 0; x < nw; ++x) {
			float sx = (std::max)(0.0f, (x + 0.5f) * xr - 0.5f);
			float sy = (std::max)(0.0f, (y + 0.5f) * yr - 0.5f);
			dst.at(x, y) = it.interpolate(src, sx, sy);
		}
	}
	return dst;
}