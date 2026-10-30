#include "Metrics.h"
#include <cmath>
#include <iostream>


Metrics::Channel Metrics::luminance(const Image& img, int shave) {
	Channel ch;
	ch.width = img.getWidth() - 2 * shave;
	ch.height = img.getHeight() - 2 * shave;
	ch.data.resize(static_cast<size_t>(ch.width) * ch.height);

	const std::vector<Pixel>& px = img.getData();
	const int src_w = img.getWidth();
	const bool gray = img.isGrayScale();

	#pragma omp parallel for schedule(static)
	for (int y = 0; y < ch.height; y++) {
		for (int x = 0; x < ch.width; x++) {
			const Pixel& p = px[(y + shave) * src_w + (x + shave)];
			ch.data[y * ch.width + x] = gray
				? static_cast<float>(p.r)
				: 16.0f + (65.481f * p.r + 128.553f * p.g + 24.966f * p.b) / 255.0f;
		}
	}
	return ch;
}

double Metrics::calculatePSNR(const Image& img1, const Image& img2, int shave) {
	if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
		std::cerr << "Error: Images must be of the same dimensions for PSNR calculation." << std::endl;
		return -1.0;
	}
	Channel a = luminance(img1, shave);
	Channel b = luminance(img2, shave);

	double sum = 0.0;
	for (size_t i = 0; i < a.data.size(); i++) {
		double d = a.data[i] - b.data[i];
		sum += d * d;
	}
	double mse = sum / a.data.size();
	if (mse == 0) {
		return INFINITY;
	}
	return 10 * log10((255.0 * 255.0) / mse);
}

/**
SSIM formula: SSIM(x, y) = ((2 * μx * μy + C1) * (2 * σxy + C2)) / ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))
*/
double Metrics::calculateSSIM(const Image& img1, const Image& img2, int shave) {
	if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
		std::cerr << "Error: Images must be of the same dimensions for SSIM calculation." << std::endl;
		return -1.0;
	}
	Channel a = luminance(img1, shave);
	Channel b = luminance(img2, shave);
	std::vector<float> kernel1d = create_gaussian_kernel_1d(11, 1.5f);
	return calculate_ssim_single_channel(a.data.data(), b.data.data(), a.width, a.height, kernel1d);
}


double Metrics::calculate_ssim_single_channel(const float* ch1, const float* ch2, int width, int height, const std::vector<float>& kernel1d) {
	constexpr float C1 = 6.5025f;   // (0.01 * 255)^2
	constexpr float C2 = 58.5225f;  // (0.03 * 255)^2
	const int N = width * height;
	const int half = static_cast<int>(kernel1d.size()) / 2;
	if (width <= 2 * half || height <= 2 * half) {
		std::cerr << "Error: Image too small for SSIM window." << std::endl;
		return -1.0;
	}

	// X^2, Y^2, XY
	std::vector<float> ch1_sq(N), ch2_sq(N), ch12(N);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < N; i++) {
		float a = ch1[i];
		float b = ch2[i];
		ch1_sq[i] = a * a;
		ch2_sq[i] = b * b;
		ch12[i] = a * b;
	}

	// μx, μy and E[x²], E[y²], E[xy]
	std::vector<float> mu_1(N), mu_2(N);
	std::vector<float> sigma_1_sq(N), sigma_2_sq(N), sigma_12(N);
	std::vector<float> temp1(N), temp2(N);

	convolve_channel(ch1, mu_1.data(), width, height, kernel1d, temp1.data());
	convolve_channel(ch2, mu_2.data(), width, height, kernel1d, temp2.data());
	convolve_channel(ch1_sq.data(), sigma_1_sq.data(), width, height, kernel1d, temp1.data());
	convolve_channel(ch2_sq.data(), sigma_2_sq.data(), width, height, kernel1d, temp2.data());
	convolve_channel(ch12.data(), sigma_12.data(), width, height, kernel1d, temp1.data());

	double ssim_sum = 0.0;
	#pragma omp parallel for reduction(+:ssim_sum) schedule(static)
	for (int y = half; y < height - half; y++) {
		for (int x = half; x < width - half; x++) {
			int i = y * width + x;
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
	}

	const double valid = static_cast<double>(width - 2 * half) * (height - 2 * half);
	return ssim_sum / valid;
}

/**
 1D Gaussian kernel for separable convolution.
*/
std::vector<float> Metrics::create_gaussian_kernel_1d(int size, float sigma) {
	std::vector<float> kernel(size);
	int half = size / 2;
	float sum = 0.0f;

	for (int x = -half; x <= half; x++) {
		float value = std::exp(-(x * x) / (2 * sigma * sigma));
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
 Borders are clamped; callers only read the interior where the window fits.
*/
void Metrics::convolve_channel(const float* input, float* output, int width, int height, const std::vector<float>& kernel1d, float* temp) {
	int ksize = static_cast<int>(kernel1d.size());
	int half = ksize / 2;
	const float* kern = kernel1d.data();

	// Pass 1: horizontal convolution → temp
	#pragma omp parallel for schedule(static)
	for (int y = 0; y < height; y++) {
		int row = y * width;

		for (int x = 0; x < half && x < width; x++) {
			float acc = 0.0f;
			for (int k = 0; k < ksize; k++) {
				int ix = x + k - half;
				if (ix < 0) ix = 0;
				acc += kern[k] * input[row + ix];
			}
			temp[row + x] = acc;
		}

		for (int x = half; x < width - half; x++) {
			float acc = 0.0f;
			int base = row + x - half;
			for (int k = 0; k < ksize; k++) {
				acc += kern[k] * input[base + k];
			}
			temp[row + x] = acc;
		}

		for (int x = (std::max)(half, width - half); x < width; x++) {
			float acc = 0.0f;
			for (int k = 0; k < ksize; k++) {
				int ix = x + k - half;
				if (ix >= width) ix = width - 1;
				acc += kern[k] * input[row + ix];
			}
			temp[row + x] = acc;
		}
	}

	// Pass 2: vertical convolution → output
	#pragma omp parallel for schedule(static)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float acc = 0.0f;
			if (y >= half && y < height - half) {
				// Interior — no bounds check
				int base = (y - half) * width + x;
				for (int k = 0; k < ksize; k++) {
					acc += kern[k] * temp[base];
					base += width;
				}
			}
			else {
				// Border rows
				for (int k = 0; k < ksize; k++) {
					int iy = y + k - half;
					if (iy < 0) iy = 0;
					else if (iy >= height) iy = height - 1;
					acc += kern[k] * temp[iy * width + x];
				}
			}
			output[y * width + x] = acc;
		}
	}
}
