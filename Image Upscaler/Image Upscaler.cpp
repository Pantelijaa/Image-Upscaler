// Image Upscaler.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <chrono>
#include "core/Image.h"
#include "core/Scaler.h"
#include "interpolation/IInterpolator.h"
#include "interpolation/Bilinear.h"
#include "metrics/Metrics.h"
const std::string PATH_TO_DATA = "../../../../data/";

int main()
{
	Image img;
	img.loadFromFile(PATH_TO_DATA + "/originals/input.jpg"); // PROMENITI NEKAKO U CMAKE

	IInterpolator* it = new Bilinear();
	Image upscaledImg = Scaler::upscale(img, img.getWidth() * 2, img.getHeight() * 2, *it);
	upscaledImg.saveToFile(PATH_TO_DATA + "/results/output_bilinear.png");
	std::cout << "Hello CMake." << std::endl;
	std::cout << "PSNR: " << Metrics::calculatePSNR(upscaledImg, upscaledImg) << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << "SSIM: " << Metrics::calculateSSIM(upscaledImg, upscaledImg) << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto ms = duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Time: " << ms << " ms\n";
	
	return 0;
}
