#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_runtime_api.h"

int main(int argc, char **argv)
{
    std::cout << "this is a test programme" << std::endl;
    const int size = 25;
    float *data = new float[size];
    float valueInfo = 1234;
    for (int i = 0; i < size; i++)
    {
        data[i] = valueInfo++;
    }
    float *dev_data;
    cudaMalloc((void **)&dev_data, size * sizeof(float));

    cudaMemcpy(dev_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cv::cuda::GpuMat img = cv::cuda::GpuMat(5, 5, CV_32FC1, dev_data);

    cv::Mat img_cpu;
    img.download(img_cpu);
    for (int i = 0; i < img_cpu.rows; i++)
    {
        for (int j = 0; j < img_cpu.cols; j++)
        {
            std::cout << img_cpu.at<float>(i, j) << std::endl;
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
