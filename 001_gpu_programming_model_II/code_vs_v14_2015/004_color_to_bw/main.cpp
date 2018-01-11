size_t numRows();
size_t numCols();

// include definitions of last declared functions
#include "hw.h"
#include "kernel.cuh"

int main(int argc, char **argv)
{
	uchar4			*h_rgbaImage, *d_rgbaImage;
	unsigned char	*h_greyImage, *d_greyImage;

	std::string input_file;
	std::string output_file;
	if (argc == 3)
	{
		input_file = std::string(argv[1]);
		output_file = std::string(argv[2]);
	}
	else
	{
		std::cerr << "Usage: hw [input_file] [output_file]" << std::endl;
		exit(1);
	}

	// load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	// kernel call: Process image
	your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());

	// check results and output the grey image
	postProcess(output_file, d_rgbaImage, d_greyImage);

	return 0;
}