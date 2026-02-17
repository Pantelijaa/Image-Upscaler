#pragma once
#include <algorithm>
#include "../core/Image.h"

class Metrics {
public:
	static double calculatePSNR(const Image& img1, const Image& img2);
	static double calculateSSIM(const Image& img1, const Image& img2);
private:
	static double calculateMSE(const Image& img1, const Image& img2);
	static std::vector<float> create_gaussian_kernel_1d(int size, float sigma);
	static double calculate_ssim_rgb(const std::vector<Pixel>& data1, const std::vector<Pixel>& data2, int width, int height, const std::vector<float>& kernel1d);
	static double calculate_ssim_single_channel(const float* ch1, const float* ch2, int width, int height, const std::vector<float>& kernel1d);
	static void convolve_channel(const float* input, float* output, int width, int height, const std::vector<float>& kernel1d, float* temp_buf);

};