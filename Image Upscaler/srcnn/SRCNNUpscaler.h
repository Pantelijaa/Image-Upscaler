#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "../core/Image.h"

class SRCNNUpscaler {
public:
	explicit SRCNNUpscaler(const std::string& onnx_path);

	/**
	 * Upscale an image by a given scale factor using the SRCNN model
	 */
	Image upscale(Image& src, int scale_factor);

	const std::string& get_model_name() const { return model_name; }

private:
	cv::dnn::Net net;
	std::string model_name;

	/**
	 * Run ONNX model on a single-channel input float Mat
	 * Input/Output: CV_32F, shape H x W
	 */
	cv::Mat inference(const cv::Mat& y_channel);
};
