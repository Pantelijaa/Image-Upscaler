#include "Bilinear.h"

Pixel Bilinear::interpolate(Image& img, float x, float y) {
	int x1 = floor(x);
	int y1 = floor(y);
	int x2 = min(x1 + 1, img.getWidth() - 1);
	int y2 = min(y1 + 1, img.getHeight() - 1);
	float dx = x - x1;
	float dy = y - y1;

	Pixel p11 = img.at(x1, y1);
	Pixel p21 = img.at(x2, y1);
	Pixel p12 = img.at(x1, y2);
	Pixel p22 = img.at(x2, y2);

	Pixel result;
	result.r = static_cast<unsigned char>(
		(p11.r * (1 - dx) * (1 - dy)) +
		(p21.r * dx * (1 - dy)) +
		(p12.r * (1 - dx) * dy) +
		(p22.r * dx * dy));
	result.g = static_cast<unsigned char>(
		(p11.g * (1 - dx) * (1 - dy)) +
		(p21.g * dx * (1 - dy)) +
		(p12.g * (1 - dx) * dy) +
		(p22.g * dx * dy));
	result.b = static_cast<unsigned char>(
		(p11.b * (1 - dx) * (1 - dy)) +
		(p21.b * dx * (1 - dy)) +
		(p12.b * (1 - dx) * dy) +
		(p22.b * dx * dy));
	return result;
}