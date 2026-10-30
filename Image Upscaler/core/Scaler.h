#pragma once

#include <vector>
#include "Image.h"
#include "../interpolation/IInterpolator.h"

class Scaler {
public:
	static Image upscale(Image& src, int nw, int nh, IInterpolator& it);

	static Image upscalePerPixel(Image& src, int nw, int nh, IInterpolator& it);

private:
	struct Taps {
		int n = 0;
		std::vector<int> index;
		std::vector<float> weight;
	};
	static Taps compute_taps(int src_size, int dst_size, const IInterpolator& it);
};
