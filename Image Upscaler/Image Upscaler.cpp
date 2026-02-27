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
#include "srcnn/SRCNNUpscaler.h"

const std::string PATH_TO_DATA = "../../../../data/";
const std::string PATH_TO_RESULTS = "../../../../results/";
const std::string PATH_TO_ORIGINALS = PATH_TO_DATA + "originals/";
const std::string PATH_TO_DOWNSCALED = PATH_TO_DATA + "downscaled/";
const std::string PATH_TO_DOWNSCALED_2x = PATH_TO_DOWNSCALED + "2x/";
const std::string PATH_TO_DOWNSCALED_4x = PATH_TO_DOWNSCALED + "4x/";
const std::string PATH_TO_ONNX_MODELS = "../../../../models/onnx/";


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

// Compute averages per (method, scale)
struct Accumulator {
	double psnr_sum = 0.0;
	double ssim_sum = 0.0;
	long long time_sum = 0;
	int count = 0;
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
			<< " - PSNR " << psnr << "dB"
			<< ", SSIM: " << ssim
			<< ", Time: " << duration_ms << "ms\n";

		results.push_back({ filename, method_name, scale_label, psnr, ssim, duration_ms });
		upscaledImg.saveToFile(output_dir + "upscaled_" + scale_label + "-" + filename);
	}
}

static void run_srcnn_upscale(
	const std::filesystem::path& downscaled_dir,
	const std::string& scale_label,
	int scale_factor,
	SRCNNUpscaler& srcnn,
	std::map<std::string, Image>& original_images,
	std::vector<MetricResult>& results)
{
	std::string method_name = srcnn.get_model_name();
	std::string output_dir = PATH_TO_RESULTS + method_name + "/" + scale_label + "/";
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
		auto start = std::chrono::high_resolution_clock::now();
		Image upscaledImg = srcnn.upscale(img, scale_factor);
		auto end = std::chrono::high_resolution_clock::now();
		long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		double psnr = Metrics::calculatePSNR(found->second, upscaledImg);
		double ssim = Metrics::calculateSSIM(found->second, upscaledImg);

		std::cout << "[" << method_name << " " << scale_label << "] " << filename
			<< " - PSNR: " << psnr << "dB"
			<< ", SSIM: " << ssim
			<< ", Time: " << duration_ms << "ms\n";

		results.push_back({ filename, method_name, scale_label, psnr, ssim, duration_ms });
		upscaledImg.saveToFile(output_dir + "upscaled_" + scale_label + "-" + filename);
	}
}

static std::string generate_key(std::string key) {
	auto end_pos = key.find_last_of("_");
	auto start_pos = key.find_last_of("/") + 1;
	return key.substr(start_pos, end_pos);
}

static void print_summary(const std::vector<MetricResult>& results) {
	constexpr int col_file = 25;
	constexpr int col_method = 20;
	constexpr int col_scale = 7;
	constexpr int col_psnr = 12;
	constexpr int col_ssim = 10;
	constexpr int col_time = 10;
	constexpr int total_width = col_file + col_method + col_scale + col_psnr + col_ssim + col_time + 7;

	auto separator = [&]() {
		std::cout << std::string(total_width, '-') << "\n";
		};

	std::cout << "\n\n";
	separator();
	std::cout << "  SUMMARY OF RESULTS\n";
	separator();

	std::cout << "| " << std::left
		<< std::setw(col_file) << "Filename"
		<< std::setw(col_method) << "Method"
		<< std::setw(col_scale) << "Scale"
		<< std::setw(col_psnr) << "PSNR (dB)"
		<< std::setw(col_ssim) << "SSIM"
		<< std::setw(col_time) << "Time (ms)"
		<< " |\n";
	separator();

	for (const auto& r : results) {
		std::cout << "| " << std::left
			<< std::setw(col_file) << r.filename
			<< std::setw(col_method) << r.method
			<< std::setw(col_scale) << r.scale
			<< std::setw(col_psnr) << std::fixed << std::setprecision(2) << r.psnr
			<< std::setw(col_ssim) << std::fixed << std::setprecision(4) << r.ssim
			<< std::setw(col_time) << r.time_ms
			<< " |\n";
	}

	separator();

	std::map<std::string, Accumulator> averages;
	for (const auto& r : results) {
		std::string group = r.method + " | " + r.scale;
		auto& acc = averages[group];
		acc.psnr_sum += r.psnr;
		acc.ssim_sum += r.ssim;
		acc.time_sum += r.time_ms;
		acc.count++;
	}

	std::cout << "\n";
	separator();
	std::cout << "  AVERAGES PER METHOD\n";
	separator();

	std::cout << "| " << std::left
		<< std::setw(col_file + col_method) << "Method / Scale"
		<< std::setw(col_scale) << "Count"
		<< std::setw(col_psnr) << "Avg PSNR"
		<< std::setw(col_ssim) << "Avg SSIM"
		<< std::setw(col_time) << "Avg Time"
		<< " |\n";
	separator();

	for (const auto& [group, acc] : averages) {
		double avg_psnr = acc.psnr_sum / acc.count;
		double avg_ssim = acc.ssim_sum / acc.count;
		long long avg_time = acc.time_sum / acc.count;

		std::cout << "| " << std::left
			<< std::setw(col_file + col_method) << group
			<< std::setw(col_scale) << acc.count
			<< std::setw(col_psnr) << std::fixed << std::setprecision(2) << avg_psnr
			<< std::setw(col_ssim) << std::fixed << std::setprecision(4) << avg_ssim
			<< std::setw(col_time) << avg_time
			<< " |\n";
	}

	separator();
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

	if (std::filesystem::exists(PATH_TO_ONNX_MODELS)) {
		std::vector<std::string> onnx_files;
		for (const auto& entry : std::filesystem::directory_iterator(PATH_TO_ONNX_MODELS)) {
			if (entry.path().extension() == ".onnx") {
				onnx_files.push_back(entry.path().string());
			}
		}
		std::sort(onnx_files.begin(), onnx_files.end());

		for (const auto& onnx_path : onnx_files) {
			try {
				SRCNNUpscaler srcnn(onnx_path);
				std::cout << "\n=== SRCNN [" << srcnn.get_model_name() << "] ===\n";
				for (auto& scale : scales) {
					run_srcnn_upscale(scale.path, scale.label, scale.factor, srcnn, original_images, all_results);
				}
			} 
			catch (const std::exception& e) {
				std::cerr << "Failed to load ONNX model " << onnx_path << ": " << e.what() << std::endl;
			}
		}

	} else {
		std::cout << "\nNo ONNX models found in " << PATH_TO_ONNX_MODELS << std::endl;
	}

	print_summary(all_results);

	return 0;
}
