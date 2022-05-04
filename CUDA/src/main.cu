#include <corecrt_math.h>

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f
// number of random spheres in the scene
#define SPHERES 100
#define DIM 1024


struct Sphere
{
	// sphere's color represented by RGB
	float r{}, b{}, g{};
	float radius{};
	// sphere's center coordinate
	float x{}, y{}, z{};
	// Given a ray shot from the pixel at (ox, oy), this function computes whether the ray intersects the sphere
	// and if so, return the distance from the camera where the ray hits the sphere
	// See scheme at Figure 6.1, page 97
	__device__ float hit(float ox, float oy, float* n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius)
		{
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

// the array of spheres is stored in constant memory
// We no longer need to cudaMalloc and cudaFree the memory, but we do need to commit to a size for this array at compile time
__constant__ Sphere s[SPHERES];

// !!! Performance with using constant memory in raytraceKernel:
// 1. a single read from constant memory can be broadcast to other "nearby" threads, effectively saving up to 15 reads
// 2. constant memory is cached, so consecutive reads of the same address will not incur any additional memory traffic

// 1. 
// Why do we mean by "nearby" threads?
// a warp refers to a collection of 32 threads that are "woven together" and get executed in lockstep
// at every line in your program, each thread in a warp executes the same instruction on different data
// When it comes to handling constant memory, NVIDIA hardware can broadcast a single memory read to each half-warp (a group of 16 threads)
//  If every thread in a half-warp requests data from the same address in constant memory,
// your GPU will generate only a single read request and subsequently broadcast the data to every thread
// If you are reading a lot of data from constant memory, you will generate only 1/16 (roughly 6 percent) of the memory traffic as you would when using global memory.
// this is a 94% reduction in bandwidth when reading from constant memory!

// 2.
// since the memory is read-only, the hardware can cache the constant data on GPU
// So after the first read from an address in constant memory, other half-warps 
// requesting the same address, and therefore hitting the constant cache, will
// generate no additional memory traffic

// For our raytraceKernel, every thread in the launch reads the data corresponding to the first sphere so the thread can test its ray for intersection
// Since we store the spheres in constant memory, the hardware needs to make only a single request for this data. After caching the data,
// every other thread avoids generating memory traffic as a result of:
// 1. it receives the data in a half-warp broadcast
// 2. it retrieves the data from the constant memory cache

// Potential downside of using constant memory:
// the half-warp broadcast can slow performance to a crawl when all 16 threads read different addresses

// the trade-off to allowing the broadcast of a single read to 16 threads is that the 16 threads are allowed to place only a single read request at a time.
// thus, if all 16 threads in a half-warp need different data from constant memory, the 16 different reads get serialized, effectively taking 16 times the amount of time to place the request
// If the threads were reading from conventional global memory, the request could be issued at the same time. In this case, reading from constant memory would probably be slower than using global memory 

__global__ void raytraceKernel(unsigned char* ptr)
{
	// Given our configuration, each thread is generating one pixel for our output (one camera ray per thread)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// shift our (x,y) image coordinate so that the z-axis runs through the center of the image
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	float r = 0, g = 0, b = 0;
	// store the closest intersection, if existed
	float maxz = -INF;

	// each ray needs to check each sphere for intersection
	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		// does this ray hit sphere i? 
		float t = s[i].hit(ox, oy, &n);
		// find the closest hit
		if (t > maxz)
		{
			// record hit's color and depth
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}




int main(void)
{
	// capture the start time
	// to measure the time a GPU spends on a task, we will use the CUDA event API
	// An event in CUDA is essentially a GPU time stamp that is recorded at a userspecified point in time

	// create start event
	cudaEvent_t start;
	HANDLE_ERROR(cudaEventCreate(&start));
	// make a record of the current time (for start event)
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// to time a block of code, we also need a stop event
	cudaEvent_t stop;
	HANDLE_ERROR(cudaEventCreate(&stop));


	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	// allocate memory on the GPU for the output bitmap
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	// allocate CPU-side sphere data
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		// randomly generate the center coordinate, color, and radius for our spheres
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(2000.0f) - 500;
		temp_s[i].y = rnd(2000.0f) - 500;
		temp_s[i].z = rnd(000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}
	// this special version of cudaMemcpy() is used when we copy from host memory to constant memory on the GPU
	HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s,
		sizeof(Sphere) * SPHERES));

	// CPU-side spheres are no longer needed 
	free(temp_s);

	// generate a bitmap from our sphere data
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	// !!!
	// some of the calls we make in CUDA C are actually asynchronous
	// For example, when we launch raytraceKernel, the GPU begins executing our code, but the CPU continues executing the next line of our program before the GPU finishes
	raytraceKernel << <grids, threads >> > (dev_bitmap);

	// copy our bitmap back from the GPU for display
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
		bitmap.image_size(),
		cudaMemcpyDeviceToHost));

	// make a record of the current time (for stop event)
	HANDLE_ERROR(cudaEventRecord(stop, 0));

	// Before reading the elapsed time from the CUDA events, we need to make sure stop event contains the correct time value. 
	// since the calls we make in CUDA C are asynchronous, we need another synchronization point for the event API to work

	// Calls to cudaEventRecord() will be placed into the GPU's pending queue of work.
	// As result, our event won't actually be recorded until the GPU finishes everything in front of cudaEventRecord() in the GPU queue

	// In this case, assume that when we reach "HANDLE_ERROR(cudaEventSynchronize(stop))", raytraceKernel is not yet finished executing. Thus, the queue looks like this: FRONT[raytraceKernel, cudaMemcpy, cudaEventRecord]BACK
	// What cudaEventSynchronize does is to instruct CPU to wait until GPU has completed [raytraceKernel, cudaMemcpy, cudaEventRecord]. This is when stop event contains the correct time.
	// So after this call, we can safely read the value from the stop event and copy it back to CPU

	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	// free our memory
	HANDLE_ERROR(cudaFree(dev_bitmap));

	bitmap.display_and_exit();
}
