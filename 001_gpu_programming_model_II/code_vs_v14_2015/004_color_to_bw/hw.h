#pragma once

// include cuda for use in cpp files
#include <cuda.h> // defines the public host functions and types for the CUDA driver API
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// use opencv for image load and save operations.
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

// utility librarys
#include <string>
#include <iostream>

size_t numRows();
size_t numCols();

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string &filename);

void postProcess(const std::string &output_file, uchar4 *d_rgbaImage, unsigned char *d_greyImage);

