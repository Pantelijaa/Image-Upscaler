#include <algorithm>
#include "SRCNNUpscaler.h"
#include "../core/Scaler.h"
#include "../interpolation/Bicubic.h"

SRCNNUpscaler::SRCNNUpscaler(const std::string& onnx_path) {
	if (!std::filesystem::exists(onnx_path)) {
		throw std::runtime_error("ONNX file not found: " + onnx_path);
	}
	try {
		net = cv::dnn::readNetFromONNX(onnx_path);
	}
	catch (const cv::Exception& e) {
		throw std::runtime_error("OpenCV failed to load ONNX model: " + std::string(e.what()));
	}
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	model_name = std::filesystem::path(onnx_path).stem().string();
}

cv::Mat SRCNNUpscaler::inference(const cv::Mat& y_channel) {
	cv::Mat blob = cv::dnn::blobFromImage(y_channel, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
	net.setInput(blob);
	cv::Mat output = net.forward();

	int out_width = output.size[3];
	int out_height = output.size[2];
	return cv::Mat(out_height, out_width, CV_32F, output.ptr<float>()).clone();
	
}

Image SRCNNUpscaler::upscale(Image& src, int scale_factor) {
	int src_width = src.getWidth();
	int src_height = src.getHeight();
	int target_width = src_width * scale_factor;
	int target_height = src_height * scale_factor;

	std::vector<Pixel> pixels = src.getData();
	cv::Mat src_mat(src_height, src_width, CV_8UC3);
	for (int i = 0; i < src_width * src_height; ++i) {
		src_mat.data[i * 3 + 0] = pixels[i].b;
		src_mat.data[i * 3 + 1] = pixels[i].g;
		src_mat.data[i * 3 + 2] = pixels[i].r;
	}

	cv::Mat upscaled_mat;
	cv::resize(src_mat, upscaled_mat, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
	
	cv::Mat ycrcb;
	cv::cvtColor(upscaled_mat, ycrcb, cv::COLOR_BGR2YCrCb);
	
	std::vector<cv::Mat> channels;
	cv::split(ycrcb, channels);

	cv::Mat y_float;
	channels[0].convertTo(y_float, CV_32F, 1.0 / 255.0);

	cv::Mat sr_y = inference(y_float);
	sr_y = (cv::min)((cv::max)(sr_y, 0.0f), 1.0f);

	cv::Mat sr_y_8u;
	sr_y.convertTo(sr_y_8u, CV_8U, 255.0);
	channels[0] = sr_y_8u;

	cv::Mat merged, result_bgr;
	cv::merge(channels, merged);
	cv::cvtColor(merged, result_bgr, cv::COLOR_YCrCb2BGR);
	
	Image result(target_width, target_height);
	for (int i = 0; i < target_width * target_height; ++i) {
		result.at(i % target_width, i / target_width) = {
			result_bgr.data[i * 3 + 2],  // R
			result_bgr.data[i * 3 + 1],  // G
			result_bgr.data[i * 3 + 0]   // B
		};
	}

	return result;
}
