#pragma once
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpuGmm.h"

struct A
{
    float *a;
    float *dev_a;
    __device__ void sort(int size)
    {
        int w;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size - 1; j++)
            {
                if (dev_a[j] < dev_a[j + 1])
                {
                    w = dev_a[j];
                    dev_a[j] = dev_a[j + 1];
                    dev_a[j + 1] = w;
                }
            }
        }
    }
    __device__ float GetValue(int num)
    {
        float value = 0;
        for (int i = 0; i < num; i++)
        {
            value += dev_a[i];
        }
        return value;
    }
};

inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void InitNode(cv::cuda::GpuMat &tmpImg, float *nodeP, double cov);

void processNode(cv::cuda::GpuMat &tmpImg, cv::cuda::GpuMat &outImg, float *nodeP, double cov, double alpha, double alpha_bar, double prune, double cfbar);

void GetNode(cv::cuda::GpuMat &imgGmm, float *nodeP);

void processDiff(cv::cuda::GpuMat &img1, cv::cuda::GpuMat &img2, cv::cuda::GpuMat &src, cv::cuda::GpuMat &result);

void caculateSim(cv::Mat &img1, cv::Mat &img2, cv::Mat &result, int binSize);

void diffsquare(cv::cuda::GpuMat &src, cv::cuda::GpuMat &filter, cv::cuda::GpuMat &result);

void testArray();

// void test(uchar *input1, uchar *input2, int imgWidth, int imgHeight, int channels, cv::Mat& tmp);
