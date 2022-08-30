#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "cuda_runtime_api.h"
#define MaxSize 4
#define DEBUGINFO   0
#define ArraySize (MaxSize + 1)

struct Gaussian
{
    float covariance; // 4
    float weight;     // 4
    float mean[3];    // 12
    Gaussian()
    {
        mean[0] = 0.0;
        mean[1] = 0.0;
        mean[2] = 0.0;
        covariance = 0.0;
        weight = 0.0;
    }
};

struct NodePixel
{
    float realSize;         // 4
    Gaussian gaussian[5]; // 20 -> 1
    NodePixel()
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
    NodePixel *node;

    float* devArray;
    cv::cuda::GpuMat tmpImg;
    cv::cuda::GpuMat outImg;

    GpuGmm(double LearningRate);
    ~GpuGmm();

    void initial(cv::Mat &orig_img);
    void process(cv::Mat &orig_img, cv::Mat &bin_img);
};
