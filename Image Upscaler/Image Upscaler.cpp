// Image Upscaler.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <chrono>
#include <map>
#include <filesystem>
#include "core/Image.h"
#include "core/Scaler.h"
#include "interpolation/IInterpolator.h"
#include "interpolation/Bilinear.h"
#include "metrics/Metrics.h"
const std::string PATH_TO_DATA = "../../../../data/";
const std::string PATH_TO_ORIGINALS = PATH_TO_DATA + "originals/";
const std::string PATH_TO_RESULTS = PATH_TO_DATA + "results/";
const std::string PATH_TO_DOWNSCALED = PATH_TO_DATA + "downscaled/";
const std::string PATH_TO_DOWNSCALED_2x = PATH_TO_DOWNSCALED + "2x/";
const std::string PATH_TO_DOWNSCALED_4x = PATH_TO_DOWNSCALED + "4x/";
static std::string generate_key(std::string key);

int main()
{
	std::map<std::string, Image> original_images;


	IInterpolator* it = new Bilinear();
	for (const auto& entry : std::filesystem::directory_iterator(PATH_TO_ORIGINALS)) {
		std::cout << entry.path() << std::endl;
		Image img;
		img.loadFromFile(entry.path().string());
		original_images[generate_key(entry.path().filename().string())] = img;
	}

	for (const auto& entry : std::filesystem::directory_iterator(PATH_TO_DOWNSCALED_2x)) {
		std::string key = generate_key(entry.path().filename().string());
		auto found = original_images.find(key);
		if (found == original_images.end()) {
			std::cerr << "Error: Original image not found for " << key << std::endl;
			continue;
		}
		std::cout << entry.path() << std::endl;
		Image img;
		img.loadFromFile(entry.path().string());
		Image upscaledImg = Scaler::upscale(img, img.getWidth() * 2, img.getHeight() * 2, *it);
		std::string filename = entry.path().filename().string();
		std::cout << filename << "PSNR: " << Metrics::calculatePSNR(found->second, upscaledImg) << std::endl;
		//std::cout << filename << "vs" << "SSIM: " << Metrics::calculateSSIM(*iter, upscaledImg) << std::endl
		upscaledImg.saveToFile(PATH_TO_RESULTS + "upscaled_2x_" + filename);
	}

	for (const auto& entry : std::filesystem::directory_iterator(PATH_TO_DOWNSCALED_4x)) {
		std::string key = generate_key(entry.path().filename().string());
		auto found = original_images.find(key);
		if (found == original_images.end()) {
			std::cerr << "Error: Original image not found for " << key << std::endl;
			continue;
		}
		std::cout << entry.path() << std::endl;
		Image img;
		img.loadFromFile(entry.path().string());
		Image upscaledImg = Scaler::upscale(img, img.getWidth() * 4, img.getHeight() * 4, *it);
		std::string filename = entry.path().filename().string();
		std::cout << filename << "PSNR: " << Metrics::calculatePSNR(found->second, upscaledImg) << std::endl;
		std::cout << filename << "SSIM: " << Metrics::calculateSSIM(found->second, upscaledImg) << std::endl;
		upscaledImg.saveToFile(PATH_TO_RESULTS + "upscaled_4x_" + filename);
	}

	delete it;
	
	return 0;
}

static std::string generate_key(std::string key) {
	auto pos = key.find_last_of("_");
	return key.substr(0, pos);
}
