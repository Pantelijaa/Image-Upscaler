#include <algorithm>
#include "SRCNNUpscaler.h"
#include "../core/Scaler.h"
#include "../interpolation/Bicubic.h"

SRCNNUpscaler::SRCNNUpscaler(const std::string& onnx_path)
	: env(ORT_LOGGING_LEVEL_WARNING, "SRCNN"),
	session(nullptr)
{
	if (!std::filesystem::exists(onnx_path)) {
		throw std::runtime_error("ONNX file not found: " + onnx_path);
	}

	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads((std::max)(1u, std::thread::hardware_concurrency()));
	opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

	std::wstring wide_path(onnx_path.begin(), onnx_path.end());
	session = Ort::Session(env, wide_path.c_str(), opts);

	model_name = std::filesystem::path(onnx_path).stem().string();
}

cv::Mat SRCNNUpscaler::inference(const cv::Mat& y_channel) {
	const int64_t h = y_channel.rows;
	const int64_t w = y_channel.cols;
	std::array<int64_t, 4> input_shape = { 1, 1, h, w };

	// Create input tensor that wraps the cv::Mat data directly (zero-copy)
	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		mem_info,
		const_cast<float*>(y_channel.ptr<float>()),
		h * w,
		input_shape.data(),
		input_shape.size()
	);

	const char* input_names[] = { "input" };
	const char* output_names[] = { "output" };

	auto output_tensors = session.Run(
		Ort::RunOptions{ nullptr },
		input_names, &input_tensor, 1,
		output_names, 1
	);

	float* out_ptr = output_tensors[0].GetTensorMutableData<float>();
	auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int out_h = static_cast<int>(out_shape[2]);
	int out_w = static_cast<int>(out_shape[3]);

	// Copy from ORT output buffer into a cv::Mat
	cv::Mat result(out_h, out_w, CV_32F);
	std::memcpy(result.ptr<float>(), out_ptr, out_h * out_w * sizeof(float));
	return result;
}

cv::Mat SRCNNUpscaler::image_to_mat(const Image& img) {
	int w = img.getWidth();
	int h = img.getHeight();
	const auto& pixels = img.getData();

	cv::Mat mat(h, w, CV_8UC3);
	auto* dst = mat.ptr<uint8_t>();
	for (int i = 0; i < w * h; ++i) {
		dst[i * 3 + 0] = pixels[i].b;
		dst[i * 3 + 1] = pixels[i].g;
		dst[i * 3 + 2] = pixels[i].r;
	}
	return mat;
}

Image SRCNNUpscaler::mat_to_image(const cv::Mat& bgr) {
	int w = bgr.cols;
	int h = bgr.rows;
	Image result(w, h);
	const auto* src = bgr.ptr<uint8_t>();
	for (int i = 0; i < w * h; ++i) {
		result.at(i % w, i / w) = {
			src[i * 3 + 2],  // R
			src[i * 3 + 1],  // G
			src[i * 3 + 0]   // B
		};
	}
	return result;
}

Image SRCNNUpscaler::upscale(Image& src, int scale_factor) {
	int target_width = src.getWidth() * scale_factor;
	int target_height = src.getHeight() * scale_factor;

	cv::Mat src_mat = image_to_mat(src);

	cv::Mat upscaled_mat;
	cv::resize(src_mat, upscaled_mat, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);

	cv::Mat ycrcb;
	cv::cvtColor(upscaled_mat, ycrcb, cv::COLOR_BGR2YCrCb);

	std::vector<cv::Mat> channels;
	cv::split(ycrcb, channels);

	cv::Mat y_float;
	channels[0].convertTo(y_float, CV_32F, 1.0 / 255.0);

	// Ensure contiguous memory for ONNX Runtime
	if (!y_float.isContinuous()) {
		y_float = y_float.clone();
	}

	cv::Mat sr_y = inference(y_float);
	(cv::min)((cv::max)(sr_y, 0.0f), 1.0f, sr_y);

	cv::Mat sr_y_8u;
	sr_y.convertTo(sr_y_8u, CV_8U, 255.0);
	channels[0] = sr_y_8u;

	cv::Mat merged, result_bgr;
	cv::merge(channels, merged);
	cv::cvtColor(merged, result_bgr, cv::COLOR_YCrCb2BGR);

	return mat_to_image(result_bgr);
}
