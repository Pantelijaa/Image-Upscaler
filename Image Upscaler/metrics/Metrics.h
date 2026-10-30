#pragma once
#include <algorithm>
#include <vector>
#include "../core/Image.h"

// PSNR and SSIM on the luminance (Y) channel
class Metrics {
public:
	static double calculatePSNR(const Image& img1, const Image& img2, int shave = 0);
	static double calculateSSIM(const Image& img1, const Image& img2, int shave = 0);
private:
	struct Channel {
		int width = 0;
		int height = 0;
		std::vector<float> data;
	};

	static Channel luminance(const Image& img, int shave);
	static std::vector<float> create_gaussian_kernel_1d(int size, float sigma);
	static double calculate_ssim_single_channel(const float* ch1, const float* ch2, int width, int height, const std::vector<float>& kernel1d);
	static void convolve_channel(const float* input, float* output, int width, int height, const std::vector<float>& kernel1d, float* temp_buf);
};
