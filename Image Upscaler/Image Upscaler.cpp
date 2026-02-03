// Image Upscaler.cpp : Defines the entry point for the application.
//

#include <iostream>
#include "core/Image.h"

using namespace std;

int main()
{
	Image img;
	img.loadFromFile("../../../../data/input.jpg"); // PROMENITI NEKAKO U CMAKE
	cout << "Hello CMake." << endl;
	return 0;
}
