#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include "../core/Image.h"

class SRCNNUpscaler {
public:
	explicit SRCNNUpscaler(const std::string& onnx_path);

	Image upscale(Image& src, int scale_factor);

	const std::string& get_model_name() const { return model_name; }

private:
	Ort::Env env;
	Ort::Session session;
	Ort::AllocatorWithDefaultOptions allocator;
	std::string model_name;

	cv::Mat inference(const cv::Mat& y_channel);

	static cv::Mat image_to_mat(const Image& img);
	static Image mat_to_image(const cv::Mat& bgr);
};
