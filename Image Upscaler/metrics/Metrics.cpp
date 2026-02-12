#include "Metrics.h"

double Metrics::calculatePSNR(const Image& img1, const Image& img2) {
	double mse = calculateMSE(img1, img2);
	if (mse == 0) {
		return INFINITY; // Images are identical
	}
	return 10 * log10((255 * 255) / mse);
}

/**
RGB SSIM:
SSIM formula: SSIM(x, y) = ((2 * μx * μy + C1) * (2 * σxy + C2)) / ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))
*/
double Metrics::calculateSSIM(const Image& img1, const Image& img2) {
	const double C1 = 6.5025;
	const double C2 = 58.5225;
	double meanA = 0.0;
	double meanB = 0.0;

	if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
		std::cerr << "Error: Images must be of the same dimensions for SSIM calculation." << std::endl;
		return -1.0;
	}

	int width = img1.getWidth();
	int height = img1.getHeight();
	std::vector<Pixel> data1 = img1.getData();
	std::vector<Pixel> data2 = img2.getData();

	std::vector<std::vector<double>> kernel = create_gaussian_kernel(11, 1.5);

	// μx, μy
	Image mu_x = convolution(img1, width, height, kernel);
	Image mu_y = convolution(img2, width, height, kernel);

	// σx^2, σy^2, σxy
	



}

double Metrics::calculateMSE(const Image& img1, const Image& img2) {
	double sum = 0.0;
	int n = img1.getWidth() * img1.getHeight() * 3;

	std::vector<Pixel> data1 = img1.getData();
	std::vector<Pixel> data2 = img2.getData();

	for (int i = 0; i < data1.size(); i++) {
		sum += (data1[i].r - data2[i].r, 2);
		sum += (data1[i].g - data2[i].g, 2);
		sum += (data1[i].b - data2[i].b, 2);
	}

	return sum / n;
}
/**
 Mathematical formula for Gaussian kernel:
 G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
 (1 / (2 * pi * sigma^2)) is the normalization factor to ensure the sum of the kernel is 1.
*/
std::vector<std::vector<double>> Metrics::create_gaussian_kernel(int size, double sigma) {
	std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
	const int half = size / 2;
	double sum = 0.0;

	// Calculate values
	for (int y = -half; y <= half; y++) {
		for (int x = -half; x <= half; x++) {
			double value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
			kernel[y + half][x + half] = value;
			sum += value;
		}
	}

	// Normalize
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			kernel[y][x] /= sum;
		}
	}

	return kernel;
}

Image Metrics::convolution(const Image& img, int width, int height, const std::vector<std::vector<double>>& kernel) {
	int kernel_size = kernel.size();
	int half = kernel_size / 2;

	Image result(width, height);
	std::vector<Pixel> data = img.getData();
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double acc_r = 0.0, acc_g = 0.0, acc_b = 0.0;
			for (int ky = 0; ky < kernel_size; ky++) {
				for (int kx = 0; kx < kernel_size; kx++) {
					int img_y = std::clamp(y + ky - half, 0, height - 1);
					int img_x = std::clamp(x + kx - half, 0, width - 1);
					Pixel& p = data[img_y * width + img_x];
					acc_r += p.r * kernel[ky][kx];
					acc_g += p.g * kernel[ky][kx];
					acc_b += p.b * kernel[ky][kx];
				}
			}
			result.at(x, y) = { static_cast<unsigned char>(std::clamp(acc_r, 0.0, 255.0)),
								static_cast<unsigned char>(std::clamp(acc_g, 0.0, 255.0)),
								static_cast<unsigned char>(std::clamp(acc_b, 0.0, 255.0)) };
		}
	}
	return result;
}