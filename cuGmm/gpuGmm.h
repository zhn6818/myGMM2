#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "cuda_runtime_api.h"
#define MaxSize 4
#define DEBUGINFO 0
#define ArraySize (MaxSize + 1)

struct GaussianGpu
{
    float covariance; // 4
    float weight;     // 4
    float mean[3];    // 12
    GaussianGpu()
    {
        mean[0] = 0.0;
        mean[1] = 0.0;
        mean[2] = 0.0;
        covariance = 0.0;
        weight = 0.0;
    }
};

struct NodePixelGpu
{
    float realSize;           // 4
    GaussianGpu gaussian[10]; // 20 -> 1
    NodePixelGpu()
    {
        realSize = 0;
    }
};
class GpuGmm
{
public:
    double alpha;
    double cT;
    double covariance0;
    double cf;
    double cfbar;
    double temp_thr;
    double prune;
    double alpha_bar;
    NodePixelGpu *node;

    float *devArray;
    cv::cuda::GpuMat tmpImg;
    cv::cuda::GpuMat outImg;
    cv::cuda::GpuMat GmmImg;
    cv::Mat imgGmm;
    GpuGmm(double LearningRate);
    ~GpuGmm();

    void initial(cv::Mat &orig_img);
    void process(cv::Mat &orig_img, cv::Mat &bin_img);
    cv::Mat getBackImg();
};
