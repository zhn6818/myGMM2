#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_runtime_api.h"
#include "comman.h"

int main(int argc, char **argv)
{
    std::cout << "this is a test programme" << std::endl;

    testArray();

    const int size = 4096;
    float *data = new float[size];
    float valueInfo = 300;
    for (int i = 0; i < size; i++)
    {
        data[i] = valueInfo;
    }
    float *dev_data;
    cudaMalloc((void **)&dev_data, size * sizeof(float));

    cudaMemcpy(dev_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    const int filersize = 256;
    float *data_filter = new float[size];
    float valueInfo_filter = 299;
    for (int i = 0; i < size; i++)
    {
        data_filter[i] = valueInfo_filter;
    }
    float *dev_data_filter;
    cudaMalloc((void **)&dev_data_filter, filersize * sizeof(float));

    cudaMemcpy(dev_data_filter, data_filter, filersize * sizeof(float), cudaMemcpyHostToDevice);

    cv::cuda::GpuMat img = cv::cuda::GpuMat(64, 64, CV_32FC1, dev_data);
    cv::cuda::GpuMat imgfilter = cv::cuda::GpuMat(16, 16, CV_32FC1, dev_data_filter);

    cv::cuda::GpuMat result = cv::cuda::GpuMat(img.rows, img.cols, CV_32FC1, cv::Scalar::all(0));

    diffsquare(img, imgfilter, result);

    cv::Mat img_cpu;
    result.download(img_cpu);
    for (int i = 0; i < img_cpu.rows; i++)
    {
        for (int j = 0; j < img_cpu.cols; j++)
        {
            std::cout << (int)(img_cpu.at<float>(i, j)) << std::endl;
        }
    }
    return 0;
}

// int main(int argc, char **argv)
// {
// std::cout << "this is a test programme" << std::endl;

// cv::Mat img = cv::Mat(5, 5, CV_32FC1, cv::Scalar::all(0));

// int value = 0;
// for (int i = 0; i < img.rows; i++)
// {
//     for (int j = 0; j < img.cols; j++)
//     {
//         img.at<float>(i, j) = value++;
//     }
// }

// uchar *dd = (img.data);

// float *x = (float *)(dd);
// for (int i = 0; i < value; i++)
// {
//     std::cout << x[i] << std::endl;
// }

// uchar *y = (uchar *)(x);

// cv::Mat test = cv::Mat(1, 25, CV_32FC1, y);
// return 0;
// }
