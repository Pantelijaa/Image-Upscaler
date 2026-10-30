// Image Upscaler.cpp : Defines the entry point for the application.
//
// Benchmarks every upscaling method (Bilinear, Bicubic, every ONNX model in
// models/onnx) on every test set found in data/test, at every scale factor in
// SCALES. Expected layout, as produced by CNN/prepare_data.py:
//
//   data/test/<Set>/HR/<name>.png
//   data/test/<Set>/LR_x<scale>/<name>.png
//
// Upscaled images go to results/<Set>/<Method>/x<scale>/<name>.png, per-image
// metrics to results/results.csv, and a per-(set, scale, method) average table
// is printed at the end.

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>
#include <chrono>
#include <map>
#include <memory>
#include <tuple>
#include <filesystem>
#include "core/Image.h"
#include "core/Scaler.h"
#include "interpolation/IInterpolator.h"
#include "interpolation/Bilinear.h"
#include "interpolation/Bicubic.h"
#include "metrics/Metrics.h"
#include "srcnn/SRCNNUpscaler.h"

namespace fs = std::filesystem;

const std::string PATH_TO_DATA = "../../../../data/";
const std::string PATH_TO_RESULTS = "../../../../results/";
const std::string PATH_TO_TEST = PATH_TO_DATA + "test/";
const std::string PATH_TO_ONNX_MODELS = "../../../../models/onnx/";

const std::vector<int> SCALES = { 2, 4 };

struct MetricResult {
	std::string dataset;
	int scale;
	std::string method;
	std::string filename;
	double psnr;
	double ssim;
	long long time_ms;
};

// An upscaling method: a name and a function that enlarges an image by a factor.
struct Method {
	std::string name;
	std::function<Image(Image&, int)> upscale;
};

// Compute averages per (dataset, scale, method)
struct Accumulator {
	double psnr_sum = 0.0;
	double ssim_sum = 0.0;
	long long time_sum = 0;
	int count = 0;
};

static bool is_image_file(const fs::path& p) {
	std::string ext = p.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

static std::vector<fs::path> list_images(const fs::path& dir) {
	std::vector<fs::path> files;
	for (const auto& entry : fs::directory_iterator(dir)) {
		if (entry.is_regular_file() && is_image_file(entry.path())) {
			files.push_back(entry.path());
		}
	}
	std::sort(files.begin(), files.end());
	return files;
}

static void run_method(
	const std::string& dataset,
	int scale,
	const fs::path& lr_dir,
	std::map<std::string, Image>& hr_images,
	const Method& method,
	std::vector<MetricResult>& results)
{
	std::string output_dir = PATH_TO_RESULTS + dataset + "/" + method.name + "/x" + std::to_string(scale) + "/";
	fs::create_directories(output_dir);

	std::cout << "\n=== " << dataset << " | x" << scale << " | " << method.name << " ===\n";

	for (const auto& lr_path : list_images(lr_dir)) {
		std::string filename = lr_path.filename().string();
		auto found = hr_images.find(filename);
		if (found == hr_images.end()) {
			std::cerr << "HR image not found for: " << filename << std::endl;
			continue;
		}

		Image img;
		if (!img.loadFromFile(lr_path.string())) {
			std::cerr << "Failed to load: " << lr_path << std::endl;
			continue;
		}

		auto start = std::chrono::steady_clock::now();
		Image upscaled = method.upscale(img, scale);
		auto end = std::chrono::steady_clock::now();
		long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		if (upscaled.getWidth() != found->second.getWidth() || upscaled.getHeight() != found->second.getHeight()) {
			std::cerr << "Size mismatch for " << filename << ": upscaled "
				<< upscaled.getWidth() << "x" << upscaled.getHeight() << ", HR "
				<< found->second.getWidth() << "x" << found->second.getHeight() << std::endl;
			continue;
		}

		// Y channel, `scale` border pixels shaved: the convention in SR papers
		double psnr = Metrics::calculatePSNR(found->second, upscaled, scale);
		double ssim = Metrics::calculateSSIM(found->second, upscaled, scale);

		std::cout << "  " << std::left << std::setw(28) << filename
			<< " PSNR " << std::fixed << std::setprecision(2) << psnr << " dB"
			<< "  SSIM " << std::setprecision(4) << ssim
			<< "  " << duration_ms << " ms\n";

		results.push_back({ dataset, scale, method.name, filename, psnr, ssim, duration_ms });
		upscaled.saveToFile(output_dir + fs::path(filename).stem().string() + ".png");
	}
}

static void write_csv(const std::vector<MetricResult>& results, const std::string& path) {
	std::ofstream csv(path);
	csv << "dataset,scale,method,image,psnr,ssim,time_ms\n";
	for (const auto& r : results) {
		csv << r.dataset << "," << r.scale << "," << r.method << "," << r.filename << ","
			<< std::fixed << std::setprecision(4) << r.psnr << "," << r.ssim << "," << r.time_ms << "\n";
	}
	std::cout << "\nPer-image results written to " << path << "\n";
}

static void print_summary(const std::vector<MetricResult>& results) {
	constexpr int col_dataset = 14;
	constexpr int col_scale = 7;
	constexpr int col_method = 20;
	constexpr int col_count = 7;
	constexpr int col_psnr = 12;
	constexpr int col_ssim = 10;
	constexpr int col_time = 12;
	constexpr int total_width = col_dataset + col_scale + col_method + col_count + col_psnr + col_ssim + col_time + 4;

	auto separator = [&]() {
		std::cout << std::string(total_width, '-') << "\n";
	};

	// Keyed by (dataset, scale, method) so the table comes out grouped and sorted.
	std::map<std::tuple<std::string, int, std::string>, Accumulator> averages;
	for (const auto& r : results) {
		auto& acc = averages[{ r.dataset, r.scale, r.method }];
		acc.psnr_sum += r.psnr;
		acc.ssim_sum += r.ssim;
		acc.time_sum += r.time_ms;
		acc.count++;
	}

	std::cout << "\n\n";
	separator();
	std::cout << "  AVERAGES PER DATASET / SCALE / METHOD\n";
	separator();
	std::cout << "| " << std::left
		<< std::setw(col_dataset) << "Dataset"
		<< std::setw(col_scale) << "Scale"
		<< std::setw(col_method) << "Method"
		<< std::setw(col_count) << "Count"
		<< std::setw(col_psnr) << "Avg PSNR"
		<< std::setw(col_ssim) << "Avg SSIM"
		<< std::setw(col_time) << "Avg Time"
		<< " |\n";
	separator();

	std::string last_dataset;
	for (const auto& [key, acc] : averages) {
		const auto& [dataset, scale, method] = key;
		if (!last_dataset.empty() && dataset != last_dataset) {
			separator();
		}
		last_dataset = dataset;

		std::cout << "| " << std::left
			<< std::setw(col_dataset) << dataset
			<< std::setw(col_scale) << ("x" + std::to_string(scale))
			<< std::setw(col_method) << method
			<< std::setw(col_count) << acc.count
			<< std::setw(col_psnr) << std::fixed << std::setprecision(2) << acc.psnr_sum / acc.count
			<< std::setw(col_ssim) << std::fixed << std::setprecision(4) << acc.ssim_sum / acc.count
			<< std::setw(col_time) << acc.time_sum / acc.count
			<< " |\n";
	}
	separator();
}

// Usage: CMakeTarget.exe [Set ...]   e.g. `CMakeTarget.exe Set5 Set14` runs only those test sets
int main(int argc, char* argv[])
{
	std::vector<std::string> only_datasets(argv + 1, argv + argc);

	// --- Methods -----------------------------------------------------------
	Bilinear bilinear;
	Bicubic bicubic;

	std::vector<Method> methods = {
		{ "Bilinear", [&](Image& img, int s) { return Scaler::upscale(img, img.getWidth() * s, img.getHeight() * s, bilinear); } },
		{ "Bicubic",  [&](Image& img, int s) { return Scaler::upscale(img, img.getWidth() * s, img.getHeight() * s, bicubic); } },
	};

	std::vector<std::unique_ptr<SRCNNUpscaler>> cnn_models;
	if (fs::exists(PATH_TO_ONNX_MODELS)) {
		std::vector<fs::path> onnx_files;
		for (const auto& entry : fs::directory_iterator(PATH_TO_ONNX_MODELS)) {
			if (entry.path().extension() == ".onnx") {
				onnx_files.push_back(entry.path());
			}
		}
		std::sort(onnx_files.begin(), onnx_files.end());

		for (const auto& onnx_path : onnx_files) {
			try {
				cnn_models.push_back(std::make_unique<SRCNNUpscaler>(onnx_path.string()));
				SRCNNUpscaler* model = cnn_models.back().get();
				methods.push_back({ model->get_model_name(), [model](Image& img, int s) { return model->upscale(img, s); } });
				std::cout << "Loaded ONNX model: " << model->get_model_name() << "\n";
			}
			catch (const std::exception& e) {
				std::cerr << "Failed to load ONNX model " << onnx_path << ": " << e.what() << std::endl;
			}
		}
	}
	if (cnn_models.empty()) {
		std::cout << "No ONNX models found in " << PATH_TO_ONNX_MODELS << "\n";
	}

	// --- Test sets ---------------------------------------------------------
	if (!fs::exists(PATH_TO_TEST)) {
		std::cerr << "Test data not found at " << PATH_TO_TEST
			<< ". Build the prepare_data CMake target (or run `python -I CNN/prepare_data.py`) first." << std::endl;
		return 1;
	}

	std::vector<fs::path> datasets;
	for (const auto& entry : fs::directory_iterator(PATH_TO_TEST)) {
		if (!entry.is_directory() || !fs::exists(entry.path() / "HR")) {
			continue;
		}
		std::string name = entry.path().filename().string();
		if (!only_datasets.empty() && std::find(only_datasets.begin(), only_datasets.end(), name) == only_datasets.end()) {
			continue;
		}
		datasets.push_back(entry.path());
	}
	std::sort(datasets.begin(), datasets.end());

	std::vector<MetricResult> all_results;

	for (const auto& dataset_dir : datasets) {
		std::string dataset = dataset_dir.filename().string();

		std::map<std::string, Image> hr_images;
		for (const auto& hr_path : list_images(dataset_dir / "HR")) {
			Image img;
			if (img.loadFromFile(hr_path.string())) {
				hr_images[hr_path.filename().string()] = img;
			}
		}
		std::cout << "\n##### " << dataset << ": " << hr_images.size() << " HR images #####\n";

		for (int scale : SCALES) {
			fs::path lr_dir = dataset_dir / ("LR_x" + std::to_string(scale));
			if (!fs::exists(lr_dir)) {
				std::cerr << "Missing " << lr_dir << ", skipping scale x" << scale << std::endl;
				continue;
			}
			for (const auto& method : methods) {
				run_method(dataset, scale, lr_dir, hr_images, method, all_results);
			}
		}
	}

	fs::create_directories(PATH_TO_RESULTS);
	write_csv(all_results, PATH_TO_RESULTS + "results.csv");
	print_summary(all_results);

	return 0;
}
