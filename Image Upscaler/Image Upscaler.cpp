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
#include "interpolation/Bicubic.h"
#include "metrics/Metrics.h"
const std::string PATH_TO_DATA = "../../../../data/";
const std::string PATH_TO_RESULTS = "../../../../results/";
const std::string PATH_TO_ORIGINALS = PATH_TO_DATA + "originals/";
const std::string PATH_TO_DOWNSCALED = PATH_TO_DATA + "downscaled/";
const std::string PATH_TO_DOWNSCALED_2x = PATH_TO_DOWNSCALED + "2x/";
const std::string PATH_TO_DOWNSCALED_4x = PATH_TO_DOWNSCALED + "4x/";


static std::string generate_key(std::string key);

struct MetricResult {
	std::string filename;
	std::string method;
	std::string scale;
	double psnr;
	double ssim;
	long long time_ms;
};

struct IneterpolatorInfo {
	std::string name;
	IInterpolator& interpolator;
};

struct ScaleConfig {
	std::string label;
	std::string path;
	int factor;
};

static void run_upscale(
	const std::filesystem::path& downscaled_dir,
	const std::string& scale_label,
	int scale_factor,
	IInterpolator& interpolator,
	const std::string& method_name,
	std::map<std::string, Image>& original_images,
	std::vector<MetricResult>& results)
{
	std::string output_dir = PATH_TO_RESULTS + method_name + scale_label + "/";
	std::filesystem::create_directories(output_dir);

	for (const auto& entry : std::filesystem::directory_iterator(downscaled_dir)) {
		std::string filename = entry.path().filename().string();
		std::string key = generate_key(filename);
		auto found = original_images.find(key);
		if (found == original_images.end()) {
			std::cerr << "Original not found for: " << key << std::endl;
			continue;
		}
		Image img;
		img.loadFromFile(entry.path().string());
		std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
		Image upscaledImg = Scaler::upscale(img, img.getWidth() * scale_factor, img.getHeight() * scale_factor, interpolator);
		std::chrono::steady_clock::time_point  end = std::chrono::high_resolution_clock::now();
		long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		
		double psnr = Metrics::calculatePSNR(found->second, upscaledImg);
		double ssim = Metrics::calculateSSIM(found->second, upscaledImg);

		std::cout << "[" << method_name << " " << scale_label << "] " << filename
			<< " - PSNR" << psnr << "dB"
			<< ", SSIM: " << ssim
			<< ", Time: " << duration_ms << "ms\n";

		results.push_back({ filename, method_name, scale_label, psnr, ssim, duration_ms });
		upscaledImg.saveToFile(output_dir + "upscaled_" + scale_label + "-" + filename);
	}
}

int main()
{
	std::map<std::string, Image> original_images;

	for (const auto& entry : std::filesystem::directory_iterator(PATH_TO_ORIGINALS)) {
		std::string key = generate_key(entry.path().filename().string());
		std::cout << "Loading original: " << key << std::endl;
		Image img;
		img.loadFromFile(entry.path().string());
		original_images[key] = img;
	}

	// Interpolators
	Bilinear bilinear;
	Bicubic bicubic;

	std::vector<IneterpolatorInfo> interpolation_methods = {
		{"Bilinear", bilinear},
		{"Bicubic", bicubic}
	};

	std::vector<ScaleConfig> scales = {
		{ "2x", PATH_TO_DOWNSCALED_2x, 2 },
		{ "4x", PATH_TO_DOWNSCALED_4x, 4 }
	};
	
	std::vector<MetricResult> all_results;

	for (auto& method : interpolation_methods) {
		for (auto& scale : scales) {
			std::cout << "\n===" << method.name << " " << scale.label << "===\n";
			run_upscale(scale.path, scale.label, scale.factor, method.interpolator, method.name, original_images, all_results);
		}
	}

	return 0;
}

static std::string generate_key(std::string key) {
	auto end_pos = key.find_last_of("_");
	auto start_pos = key.find_last_of("/") + 1;
	return key.substr(start_pos, end_pos);
}
