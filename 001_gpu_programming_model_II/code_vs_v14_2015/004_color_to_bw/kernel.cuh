#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void your_rgba_to_greyscale(uchar4 const * const h_rgbaImage, uchar4 * const d_rgbaImage,
	unsigned char * const d_greyImage, size_t numRows, size_t numCols);