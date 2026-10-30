#include "Scaler.h"
#include <algorithm>
#include <cmath>

// Pixel-centre alignment: destination pixel d samples the source at (d + 0.5) * ratio - 0.5
static inline float source_coord(int d, float ratio) {
	return (std::max)(0.0f, (d + 0.5f) * ratio - 0.5f);
}

static inline unsigned char to_byte(float v) {
	return static_cast<unsigned char>(std::clamp(std::round(v), 0.0f, 255.0f));
}

Scaler::Taps Scaler::compute_taps(int src_size, int dst_size, const IInterpolator& it) {
	const int r = it.radius();
	Taps t;
	t.n = 2 * r;
	t.index.resize(static_cast<size_t>(dst_size) * t.n);
	t.weight.resize(static_cast<size_t>(dst_size) * t.n);

	const float ratio = static_cast<float>(src_size) / dst_size;
	for (int d = 0; d < dst_size; ++d) {
		float s = source_coord(d, ratio);
		int is = static_cast<int>(std::floor(s));
		float frac = s - is;
		for (int k = 0; k < t.n; ++k) {
			int offset = k - (r - 1);  // bilinear: 0, 1   bicubic: -1, 0, 1, 2
			t.index[d * t.n + k] = std::clamp(is + offset, 0, src_size - 1);
			t.weight[d * t.n + k] = static_cast<float>(it.weight(frac - offset));
		}
	}
	return t;
}

Image Scaler::upscale(Image& src, int nw, int nh, IInterpolator& it) {
	const int sw = src.getWidth();
	const int sh = src.getHeight();
	const Taps tx = compute_taps(sw, nw, it);
	const Taps ty = compute_taps(sh, nh, it);
	const std::vector<Pixel>& in = src.getData();

	// Pass 1: horizontal, every source row -> destination width (kept as float)
	std::vector<float> tmp(static_cast<size_t>(sh) * nw * 3);
	#pragma omp parallel for schedule(static)
	for (int y = 0; y < sh; ++y) {
		const Pixel* row = &in[static_cast<size_t>(y) * sw];
		float* out = &tmp[static_cast<size_t>(y) * nw * 3];
		for (int x = 0; x < nw; ++x) {
			const int* idx = &tx.index[static_cast<size_t>(x) * tx.n];
			const float* w = &tx.weight[static_cast<size_t>(x) * tx.n];
			float r = 0.0f, g = 0.0f, b = 0.0f;
			for (int k = 0; k < tx.n; ++k) {
				const Pixel& p = row[idx[k]];
				r += w[k] * p.r;
				g += w[k] * p.g;
				b += w[k] * p.b;
			}
			out[x * 3 + 0] = r;
			out[x * 3 + 1] = g;
			out[x * 3 + 2] = b;
		}
	}

	// Pass 2: vertical, destination rows from the horizontal result
	Image dst(nw, nh);
	#pragma omp parallel for schedule(static)
	for (int y = 0; y < nh; ++y) {
		const int* idx = &ty.index[static_cast<size_t>(y) * ty.n];
		const float* w = &ty.weight[static_cast<size_t>(y) * ty.n];
		for (int x = 0; x < nw; ++x) {
			float r = 0.0f, g = 0.0f, b = 0.0f;
			for (int k = 0; k < ty.n; ++k) {
				const float* p = &tmp[(static_cast<size_t>(idx[k]) * nw + x) * 3];
				r += w[k] * p[0];
				g += w[k] * p[1];
				b += w[k] * p[2];
			}
			dst.at(x, y) = { to_byte(r), to_byte(g), to_byte(b) };
		}
	}
	return dst;
}

Image Scaler::upscalePerPixel(Image& src, int nw, int nh, IInterpolator& it) {
	Image dst(nw, nh);
	float xr = static_cast<float>(src.getWidth()) / nw;
	float yr = static_cast<float>(src.getHeight()) / nh;

	for (int y = 0; y < nh; ++y) {
		for (int x = 0; x < nw; ++x) {
			dst.at(x, y) = it.interpolate(src, source_coord(x, xr), source_coord(y, yr));
		}
	}
	return dst;
}
