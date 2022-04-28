#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DIM 1000

struct cuComplex
{
    float r;
    float i;
    // note: here we need to declare ctor as __device__ as well
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void)
    {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr)
{
    // Note: unlike in CPU version, we don't have for loop to generate pixel indices anymore/
    // in GPU, we compute pixel indices through blockIdx

    // map from blockIdx to pixel position
    // (x,y) ranges from (0, 0) and (DIM - 1, DIM - 1)
    int x = blockIdx.x;
    int y = blockIdx.y;
    // offset ranges from 0 to (DIM * DIM - 1)
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(void)
{
    CPUBitmap bitmap(DIM, DIM);
    // declare a pointer to hold a copy of the data on the device
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    // specify 2D grid of blocks because our problem is 2D
    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
                            bitmap.image_size(),
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_bitmap));

    bitmap.display_and_exit();
}
