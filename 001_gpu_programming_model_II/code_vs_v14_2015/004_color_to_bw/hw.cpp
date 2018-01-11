#include "hw.h"

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
	std::cout << "\nLOADING IMAGE:" << std::endl;
	cudaError_t cudaStatus;

	//make sure the context initializes ok
	cudaFree(0);

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		std::cerr << "\t- Couldn´t open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	// Allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	// This shouldn't ever happen given the way the images are created
	// at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous())
	{
		std::cerr << "\t- Images aren`t continuos!! Exiting." << std::endl;
		exit(1);
	}

	// smart opencv version of shared_ptr
	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	// allocate memory on the device for both input and output
	cudaStatus = cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "\t- cudaMalloc failed for d_rgbaImage" << std::endl;
		return;
	}

	cudaStatus = cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "\t- cudaMalloc failed for d_greyImage" << std::endl;
		return;
	}

	//make sure no memory is left laying around
	//cudaStatus = cudaMemset(d_greyImage, 0, sizeof(unsigned char) * numPixels); //make sure no memory is left laying around
	/*cudaStatus = cudaMemset(&d_greyImage, 0, sizeof(unsigned char) * numPixels);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "\t- cudaMemset failed for d_greyImage: " << cudaGetLastError() << std::endl;
	}*/

	// copy input array to the gpu
	cudaStatus = cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "\t- Failed to copy image data from HOST to DEVICE" << cudaGetLastError() << std::endl;
		return;
	}

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;

	std::cout << "\t- Image " << filename << " of dimensions " << image.cols << "x" << image.rows << " loaded succesfully" << std::endl;
}

void postProcess(const std::string &output_file, uchar4 *d_rgbaImage, unsigned char *d_greyImage)
{
	std::cout << "\nSAVING IMAGE:" << std::endl;

	cudaError_t cudaStatus;
	const int numPixels = numRows() * numCols();

	// copy the output back to the host
	cudaStatus = cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	//cudaStatus = cudaMemcpy(imageRGBA.ptr<uchar4>(0), d_rgbaImage__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "\t- Failed to copy image data from DEVICE to HOST" << std::endl;
		return;
	}
	//imageGrey = imageRGBA;
	// output the image
	cv::imwrite(output_file.c_str(), imageGrey);
	//cv::imwrite(output_file.c_str(), imageRGBA);

	// cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
	std::cout << "\t- Image saved succesfully" << std::endl;
}