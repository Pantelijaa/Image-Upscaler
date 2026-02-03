#include "Metrics.h"

Metrcis::calculatePSNR(const Image& img1, const Image& img2) {
	double mse = calculateMSE(img1, img2);
	if (mse == 0) {
		return INFINITY; // Images are identical
	}
	return 10 * log10((255 * 255) / mse);
}


Metrcis::calculateMSE(const Image& img1, const Image& img2) {
	double sum = 0.0;
	int n = img1.width * img1.height * 3;

	for (int i = 0; i < img1.data.size(); i++) {
		sum += (img1.data[i].r - img2.data[i].r, 2);
		sum += (img1.data[i].g - img2.data[i].g, 2);
		sum += (img1.data[i].b - img2.data[i].b, 2);
	}

	return sum / n;
}