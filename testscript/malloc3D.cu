#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <driver_functions.h>

__global__ void myKernel(cudaPitchedPtr devPitchedPtr, cudaExtent extent)
{
    float * devPtr = (float *)devPitchedPtr.ptr;
    float *sliceHead, *rowHead;
        // 可以定义为 char * 作面、行迁移的时候直接加减字节数，取行内元素的时候再换回 float *

    for (int z = 0; z < extent.depth; z++)
    {
        sliceHead = (float *)((char *)devPtr + z * devPitchedPtr.pitch * extent.height);
        for (int y = 0; y < extent.height; y++)
        {
            rowHead = (float*)((char *)sliceHead + y * devPitchedPtr.pitch);
            for (int x = 0; x < extent.width / sizeof(float); x++)// extent 存储的是行有效字节数，要除以元素大小 
            {
                printf("\t%f",rowHead[x]);// 逐个打印并自增 1
                rowHead[x]++;
            }
            printf("\n");
        }
        printf("\n");
    }
}
 
int main()
{
    size_t width = 2;
    size_t height = 3;
    size_t depth = 4;
    float *h_data;

    cudaPitchedPtr d_data;
    cudaExtent extent;
    cudaMemcpy3DParms cpyParm;

    h_data = (float *)malloc(sizeof(float) * width * height * depth);
    for (int i = 0; i < width * height * depth; i++)
        h_data[i] = (float)i;

    printf("\n\tAlloc memory.");

    //三维数组申请的方式，不管是几维的，首先保证的是行对齐

    extent = make_cudaExtent(sizeof(float) * width, height, depth);
    cudaMalloc3D(&d_data, extent);
                                
    printf("\n\tCopy to Device.\n");
    cpyParm = {0};
    cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_data, sizeof(float) * width, width, height);
    cpyParm.dstPtr = d_data;
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParm);

    myKernel << <1, 1 >> > (d_data, extent);
    cudaDeviceSynchronize();

    printf("\n\tCopy back to Host.\n");
    cpyParm = { 0 };
    cpyParm.srcPtr = d_data;
    cpyParm.dstPtr = make_cudaPitchedPtr((void*)h_data, sizeof(float) * width, width, height);
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyDeviceToHost;
    // memcpy3D 也是和 cpy 2D不一样的地方
    cudaMemcpy3D(&cpyParm);

    for (int i = 0; i < width*height*depth; i++)
    {
        printf("\t%f", h_data[i]);
        if ((i + 1) % width == 0)
            printf("\n");
        if ((i + 1) % (width*height) == 0)
            printf("\n");
    }               

    free(h_data);
    cudaFree(d_data.ptr);
    // getchar();
    return 0;
}