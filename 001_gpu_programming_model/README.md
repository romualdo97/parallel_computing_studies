# Todo list
> 1) Develop exercises from `Introduction to CUDA C - GPU Technology Conference` slides

# Basic CUDA program structure

A basic CUDA program structure consists basically of three parts

**PART 1 - Allocates `DEVICE` memory:** Here we allocate memory for later usage in kernel computations.

	cudaMalloc((void**) &dev_a, size);

**PART 2 - Pass data from `HOST`to `DEVICE` memory:** Assign values to some previously allocated device memory.

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

**PART 3 -	Launch kernel:** From CPU execute the kernels.

	kernel<<<1, 1>>>(dev_a);

**PART 4 -	Get the results:** request to GPU for corresponding results in kernel computations.

	cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);

**PART 4 -	De-allocate memory:** don't forget to free the memory at the end of program execution.

	cudaFree(dev_a);

# Student notes

The class notes here presented are not of my property except the exercices development and some additional notes added to professor slides, the class notes slides are here just for future fast reference purpouse and were extracted from Intro to Parallel Programming in collaboration with Nvidia© that is free available at Udacity©.
The main purpouse of this repo is to mantain an organized register of exercises source code and explain core functionality of algorithms involved using this readme.