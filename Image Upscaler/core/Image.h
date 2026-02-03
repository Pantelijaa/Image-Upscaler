#pragma once

#include <vector>
#include <string>
#include <windows.h>

struct Pixel {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

class Image {	
	private:
		int width;
		int height;
		int channels; // RGB
		std::vector<Pixel> data;
	public:
	Image(int w=0, int h=0);
	Pixel& at(int x, int y);
	bool loadFromFile(const std::string& filename);
	bool saveToFile(const std::string& filename);

	int getWidth() const;
	int getHeight() const;

};