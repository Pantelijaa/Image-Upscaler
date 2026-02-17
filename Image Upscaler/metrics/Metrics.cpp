#include "Metrics.h"

double Metrics::calculatePSNR(const Image& img1, const Image& img2) {
	if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
		std::cerr << "Error: Images must be of the same dimensions for PSNR calculation." << std::endl;
		return -1.0;
	}
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
	const std::vector<Pixel> data1 = img1.getData();
	const std::vector<Pixel> data2 = img2.getData();

	std::vector<float> kernel1d = create_gaussian_kernel_1d(11, 1.5);

	if (img1.isGrayScale()) {
		std::vector<float> gray1(N), gray2(N);
		for (int i = 0; i < N; i++) {
			gray1[i] = data1[i].r;
			gray2[i] = data2[i].r;
		}
		return calculate_ssim_single_channel(gray1.data(), gray2.data(), width, height, kernel1d);
	}
	else {
		return calculate_ssim_rgb(data1, data2, width, height, kernel1d);
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

double Metrics::calculate_ssim_single_channel(const float* ch1, const float* ch2, int width, int height, const std::vector<float>& kernel1d) {
	constexpr float C1 = 6.5025f;
	constexpr float C2 = 58.5225f;
	int N = width * height;
	double ssim_sum = 0.0;

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

	// μx, μy
	std::vector<float> mu_1(N), mu_2(N);
	std::vector<float> sigma_1_sq(N), sigma_2_sq(N), sigma_12(N);
	//  E[x²], E[y²], E[xy]
	std::vector<float> temp1(N), temp2(N);

	convolve_channel(ch1, mu_1.data(), width, height, kernel1d, temp1.data());
	convolve_channel(ch2, mu_2.data(), width, height, kernel1d, temp2.data());
	convolve_channel(ch1_sq.data(), sigma_1_sq.data(), width, height, kernel1d, temp1.data());
	convolve_channel(ch2_sq.data(), sigma_2_sq.data(), width, height, kernel1d, temp2.data());
	convolve_channel(ch12.data(), sigma_12.data(), width, height, kernel1d, temp1.data());
	#pragma omp parallel for reduction(+:ssim_sum) schedule(static)
	for (int i = 0; i < N; i++) {
		double mux = mu_1[i];
		double muy = mu_2[i];
		float mux2 = mux * mux;
		float muy2 = muy * muy;

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

double Metrics::calculate_ssim_rgb(const std::vector<Pixel>& data1, const std::vector<Pixel>& data2, int width, int height, const std::vector<float>& kernel1d) {
	int N = width * height;

	std::vector<float> r1(N), g1(N), b1(N);
	std::vector<float> r2(N), g2(N), b2(N);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < N; i++) {
		const Pixel& p1 = data1[i];
		const Pixel& p2 = data2[i];
		r1[i] = p1.r; g1[i] = p1.g; b1[i] = p1.b;
		r2[i] = p2.r; g2[i] = p2.g; b2[i] = p2.b;
	}

	double ssim_r = calculate_ssim_single_channel(r1.data(), r2.data(), width, height, kernel1d);
	double ssim_g = calculate_ssim_single_channel(g1.data(), g2.data(), width, height, kernel1d);
	double ssim_b = calculate_ssim_single_channel(b1.data(), b2.data(), width, height, kernel1d);

	return (ssim_r + ssim_g + ssim_b) / 3.0;
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