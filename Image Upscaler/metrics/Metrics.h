#pragma once
#include "../core/Image.h"

class Metrics {
public:
	static double calculatePSNR(const Image& img1, const Image& img2);
	static double calculateSSIM(const Image& img1, const Image& img2);
private:
	static double calculateMSE(const Image& img1, const Image& img2);
	static std::vector<std::vector<double>> create_gaussian_kernel(int size, double sigma);

};