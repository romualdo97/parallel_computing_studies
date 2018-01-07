// include definitions of the crucial functions for this homework
#include "hw.cpp"
#include "kernel.cu"

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

	// check results and output the grey image
	postProcess(output_file);

	return 0;
}