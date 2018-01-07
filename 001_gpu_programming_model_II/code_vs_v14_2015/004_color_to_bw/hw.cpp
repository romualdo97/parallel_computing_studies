// use of cude.h and cuda_runtime.h headers: https://stackoverflow.com/questions/6302695/difference-between-cuda-h-cuda-runtime-h-cuda-runtime-api-h
// For writing host code to be compiled with the host compiler which
// includes CUDA API calls
#include <cuda.h> // defines the public host functions and types for the CUDA driver API
#include <cuda_runtime_api.h>
#include <cuda_runtime.h> // defines everything cuda_runtime_api.h does, as well as built-in type definitions and function overlays for the CUDA language extensions and device intrinsic functions.

// use opencv for image load and save operations.
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

// utility librarys
#include <string>
#include <iostream>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...

//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string &filename)
{
	//make sure the context initializes ok
	cudaFree(0);

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		std::cerr << "Couldn´t open file: " << filename << std::endl;
		exit(1);
	}

	// smart opencv version of shared_ptr
	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	// allocate memory on the device for both input and output
	cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
	cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
	cudaMemset(d_greyImage, 0, sizeof(unsigned char) * numPixels); //make sure no memory is left laying around

	// copy input array to the gpu
	cudaMemcpy(*d_greyImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file)
{
	const int numPixels = numRows() * numCols();
	// copy the output back to the host
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

	// output the image
	cv::imwrite(output_file.c_str(), imageGrey);

	// cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}