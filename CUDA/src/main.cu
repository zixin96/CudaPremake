#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N (33 * 1024)

// In order to process even larger vector, each thread needs to process more than one element

__global__ void add(int* a, int* b, int* c)
{
	// each parallel thread needs to start on a different data index
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		// each thread increments their indices by the total number of threads to ensure we don't miss any elements and don't add a pair twice
		tid += blockDim.x * gridDim.x;
	}
}

int main(void)
{
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the CPU
	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	c = (int*)malloc(N * sizeof(int));

	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// fill the arrays 'a' and 'b' on the CPU
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	// copy the arrays 'a' and 'b' to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice));

	// to ensure that we never lanuch too many blocks/threads, 
	// we will fix the number of blocks and blocks to some reasonable small value
	add<<<128, 128>>>(dev_a, dev_b, dev_c);

	// copy the array 'c' back from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost));

	// verify that the GPU did the work we requested
	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if ((a[i] + b[i]) != c[i])
		{
			printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if (success) printf("We did it!\n");

	// free the memory we allocated on the GPU
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	// free the memory we allocated on the CPU
	free(a);
	free(b);
	free(c);

	return 0;
}
