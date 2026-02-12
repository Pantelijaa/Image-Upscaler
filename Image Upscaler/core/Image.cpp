#include "Image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../includes/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../includes/stb_image_write.h"

Image::Image(int w, int h) : width(w), height(h), channels(3) {
	data.resize(w * h);
}

Pixel& Image::at(int x, int y) {
	return data[y * width + x];
}

bool Image::loadFromFile(const std::string& filename) {
	unsigned char* imgData = stbi_load(filename.c_str(), &width, &height, &channels, 3);
	std::cout << std::filesystem::current_path() << std::endl; // error debugging
	if (!imgData)
		return false;
	data.resize(width * height);
	for(int i = 0; i < width * height; ++i) {
		data[i].r = imgData[i * 3];
		data[i].g = imgData[i * 3 + 1];
		data[i].b = imgData[i * 3 + 2];
	}
	stbi_image_free(imgData);

	return true;
}

bool Image::saveToFile(const std::string& filename) {
	std::vector <unsigned char> rawData(width * height * 3);

	for (int i = 0; i < width * height; ++i) {
		rawData[i * 3] = data[i].r;
		rawData[i * 3 + 1] = data[i].g;
		rawData[i * 3 + 2] = data[i].b;
	}

	std::string ext = filename.substr(filename.find_last_of('.') + 1);

	if (ext == "png") {
		return stbi_write_png(filename.c_str(), width, height, 3, rawData.data(), width * 3);
	}
	else if (ext == "jpg" || ext == "jpeg") {
		return stbi_write_jpg(filename.c_str(), width, height, 3, rawData.data(), 90);
	}

	return stbi_write_png(filename.c_str(), width, height, 3, rawData.data(), width * 3);
}

int Image::getWidth() const {
	return width;
}

int Image::getHeight() const {
	return height;
}

std::vector<Pixel> Image::getData() const {
	return data;
}
