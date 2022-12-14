#pragma once
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpuGmm.h"

inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }



void InitNode(cv::cuda::GpuMat &tmpImg, float *nodeP, double cov);

void processNode(cv::cuda::GpuMat &tmpImg, cv::cuda::GpuMat &outImg, float *nodeP, double cov, double alpha, double alpha_bar, double prune, double cfbar);


void test(uchar *input1, uchar *input2, int imgWidth, int imgHeight, int channels, cv::Mat& tmp);

