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

# Main difference between blocks and threads

Unlike parallel blocks, parallel **threads have mechanism to communicate and synchronize**, see `003_synchronization_example` for a demonstration or read `Introduction to CUDA C slides` for a deeper explanation of the sychronization example.

# Race conditions

Las condiciones de carrera ocurren cuando dos o mas "procesos" intentan interactuar con un **recurso compartido**, y una forma de evitar comportamientos inesperados es  haciendo que las operaciones sobre dicho recurso compartido sean **operaciones atómicas.**

> [Segun wikipedia](https://es.wikipedia.org/wiki/Condici%C3%B3n_de_carrera)
> Condición de carrera o Condición de Secuencia (del inglés race condition) es una expresión usada en electrónica y en programación. 
> Cuando la salida o estado de un proceso es dependiente de una secuencia de eventos que se ejecutan en orden arbitrario y van a trabajar sobre un mismo recurso compartido, se puede producir un bug cuando dichos eventos no "llegan" (se ejecutan) en el orden que el programador esperaba. El término se origina por la similitud de dos procesos "compitiendo" en carrera por llegar antes que el otro, de manera que el estado y la salida del sistema dependerán de cuál "llegó" antes, pudiendo provocarse inconsistencias y comportamientos impredecibles y no compatibles con un sistema determinista.

en el kernel `multiblock_dot_kernel`ubicado en `code_vs_v14_2015 / 003_synchronization_example` se puede ver un ejemplo de condición de carrera, que se intentará explicar a continuación.


	__global__ void multiblock_dot_kernel(int *r, int *a, int *b)

En la dirección de memoria representada por el puntero `int *r`se guardará el resultado del kernel.

	if (0 == threadIdx.x)
	
las siguientes operaciones se realizaran en el **thread 0** de cada **block**

	atomicAdd(r, sum); // "read-modify-write" uninterrumpible when atomic

Dado que varios **blocks** están realizando **operaciones (read-modify-write)** en el valor ubicado en la dirección `r`, podemos considerar que **el `valor` en la dirección `r` es un recurso compartido entre los distintos`blocks`**, por esta razón es necesario hacer que el valor en `sum` se adicione al `valor` en la dirección `r` mediante una operación de suma atómica.

# Student notes

The class notes here presented are not of my property except the exercices development and some additional notes added to professor slides, the class notes slides are here just for future fast reference purpouse and were extracted from Intro to Parallel Programming in collaboration with Nvidia© that is free available at Udacity©.
The main purpouse of this repo is to mantain an organized register of exercises source code and explain core functionality of algorithms involved using this readme.