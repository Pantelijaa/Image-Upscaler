#include "Metrics.h"

double Metrics::calculatePSNR(const Image& img1, const Image& img2) {
	double mse = calculateMSE(img1, img2);
	if (mse == 0) {
		return INFINITY; // Images are identical
	}
	return 10 * log10((255 * 255) / mse);
}

/**
Single channel SSIM:
SSIM formula: SSIM(x, y) = ((2 * μx * μy + C1) * (2 * σxy + C2)) / ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))
*/
double Metrics::calculateSSIM(const Image& img1, const Image& img2) {
	if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
		std::cerr << "Error: Images must be of the same dimensions for SSIM calculation." << std::endl;
		return -1.0;
	}

	int width = img1.getWidth();
	int height = img1.getHeight();
	int N = width * height;
	std::vector<Pixel> data1 = img1.getData();
	std::vector<Pixel> data2 = img2.getData();

	std::vector<double> kernel1d = create_gaussian_kernel_1d(11, 1.5);

	if (img1.isGrayScale()) {
		std::vector<double> gray1(N), gray2(N);
		for (int i = 0; i < N; i++) {
			gray1[i] = data1[i].r;
			gray2[i] = data2[i].r;
		}
		return calculate_ssim_single_channel(gray1, gray2, width, height, kernel1d);
	}
	else {
		return calculate_ssim_rgb(img1, img2, width, height, kernel1d);
	}
}

double Metrics::calculateMSE(const Image& img1, const Image& img2) {
	double sum = 0.0;
	int n = img1.getWidth() * img1.getHeight() * 3;

	std::vector<Pixel> data1 = img1.getData();
	std::vector<Pixel> data2 = img2.getData();

	for (int i = 0; i < data1.size(); i++) {
		sum += std::pow(data1[i].r - data2[i].r, 2);
		sum += std::pow(data1[i].g - data2[i].g, 2);
		sum += std::pow(data1[i].b - data2[i].b, 2);
	}

	return sum / n;
}

double Metrics::calculate_ssim_single_channel(const std::vector<double>& ch1, const std::vector<double>& ch2, int width, int height, const std::vector<double>& kernel1d) {
	const double C1 = 6.5025;
	const double C2 = 58.5225;
	int N = width * height;
	double ssim_sum = 0.0;

	// X^2, Y^2, XY
	std::vector<double> ch1_sq(N), ch2_sq(N), ch12(N);
	for (int i = 0; i < N; i++) {
		ch1_sq[i] = ch1[i] * ch1[i];
		ch2_sq[i] = ch2[i] * ch2[i];
		ch12[i] = ch1[i] * ch2[i];
	}

	// μx, μy
	std::vector<double> mu_1 = convolve_channel(ch1, width, height, kernel1d);
	std::vector<double> mu_2 = convolve_channel(ch2, width, height, kernel1d);

	//  E[x²], E[y²], E[xy]
	std::vector<double> sigma_1_sq = convolve_channel(ch1_sq, width, height, kernel1d);
	std::vector<double> sigma_2_sq = convolve_channel(ch2_sq, width, height, kernel1d);
	std::vector<double> sigma_12 = convolve_channel(ch12, width, height, kernel1d);

	for (int i = 0; i < N; i++) {
		double mux = mu_1[i];
		double muy = mu_2[i];

		// σ² = E[x²] - μ²
		double sigmax2 = sigma_1_sq[i] - mux * mux;
		double sigmay2 = sigma_2_sq[i] - muy * muy;
		double sigmaxy = sigma_12[i] - mux * muy;

		double numerator = (2 * mux * muy + C1) * (2 * sigmaxy + C2);
		double denominator = (mux * mux + muy * muy + C1) * (sigmax2 + sigmay2 + C2);

		ssim_sum += numerator / denominator;
	}

	return ssim_sum / N;
}

double Metrics::calculate_ssim_rgb(const Image& img1, const Image& img2, int width, int height, const std::vector<double>& kernel1d) {
	int N = width * height;
	std::vector<Pixel> data1 = img1.getData();
	std::vector<Pixel> data2 = img2.getData();

	std::vector<double> r1(N), g1(N), b1(N);
	std::vector<double> r2(N), g2(N), b2(N);
	for (int i = 0; i < N; i++) {
		r1[i] = data1[i].r; g1[i] = data1[i].g; b1[i] = data1[i].b;
		r2[i] = data2[i].r; g2[i] = data2[i].g; b2[i] = data2[i].b;
	}

	double ssim_r = calculate_ssim_single_channel(r1, r2, width, height, kernel1d);
	double ssim_g = calculate_ssim_single_channel(g1, g2, width, height, kernel1d);
	double ssim_b = calculate_ssim_single_channel(b1, b2, width, height, kernel1d);

	return (ssim_r + ssim_g + ssim_b) / 3.0;
}

/**
 1D Gaussian kernel for separable convolution.
*/
std::vector<double> Metrics::create_gaussian_kernel_1d(int size, double sigma) {
	std::vector<double> kernel(size);
	int half = size / 2;
	double sum = 0.0;

	for (int x = -half; x <= half; x++) {
		double value = std::exp(-(x * x) / (2 * sigma * sigma));
		kernel[x + half] = value;
		sum += value;
	}

	for (int i = 0; i < size; i++) {
		kernel[i] /= sum;
	}

	return kernel;
}

/**
 Separable 2D Gaussian convolution via two 1D passes:
   1) Horizontal pass — convolve each row
   2) Vertical pass   — convolve each column
 Complexity: O(N * K * 2) instead of O(N * K^2)
*/
std::vector<double> Metrics::convolve_channel(const std::vector<double>& channel, int width, int height, const std::vector<double>& kernel1d) {
	int ksize = static_cast<int>(kernel1d.size());
	int half = ksize / 2;
	int N = width * height;

	// Pass 1: horizontal
	std::vector<double> temp(N, 0.0);
	for (int y = 0; y < height; y++) {
		int row = y * width;
		for (int x = 0; x < width; x++) {
			double acc = 0.0;
			for (int k = 0; k < ksize; k++) {
				int ix = std::clamp(x + k - half, 0, width - 1);
				acc += kernel1d[k] * channel[row + ix];
			}
			temp[row + x] = acc;
		}
	}

	// Pass 2: vertical
	std::vector<double> result(N, 0.0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double acc = 0.0;
			for (int k = 0; k < ksize; k++) {
				int iy = std::clamp(y + k - half, 0, height - 1);
				acc += kernel1d[k] * temp[iy * width + x];
			}
			result[y * width + x] = acc;
		}
	}

	return result;
}