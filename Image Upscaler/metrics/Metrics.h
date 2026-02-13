#pragma once
#include <algorithm>
#include "../core/Image.h"

class Metrics {
public:
	static double calculatePSNR(const Image& img1, const Image& img2);
	static double calculateSSIM(const Image& img1, const Image& img2);
private:
	static double calculateMSE(const Image& img1, const Image& img2);
	static std::vector<double> create_gaussian_kernel_1d(int size, double sigma);
	static double calculate_ssim_rgb(const Image& img1, const Image& img2, int width, int height, const std::vector<double>& kernel1d);
	static double calculate_ssim_single_channel(const std::vector<double>& ch1, const std::vector<double>& ch2, int width, int height, const std::vector<double>& kernel1d);
	static std::vector<double> convolve_channel(const std::vector<double>& channel, int width, int height, const std::vector<double>& kernel1d);
};