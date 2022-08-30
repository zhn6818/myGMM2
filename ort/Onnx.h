//
// Created by 张海宁 on 2020/8/19.
//

#ifndef MACOPENCV_ONNX_H
#define MACOPENCV_ONNX_H

#include <cassert>
#include <vector>
#include <cassert>
#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>

#include <core/session/onnxruntime_cxx_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
#include <core/session/onnxruntime_c_api.h>

// #include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <array>
#include "UTIL.h"

using namespace std;
using namespace cv;

const int WIDTH = 544;
const int HEIGHT = 544;
const int CHANNEL = 3;

// typedef std::array<float, 2 * HEIGHT * WIDTH> ResultType;
typedef std::array<float, 1 * WIDTH * HEIGHT * CHANNEL> InputType;

class MyOnnxPack
{
private:
    Ort::Env *env;
    Ort::SessionOptions *sessionOptions;
    Ort::Session *session;
    Ort::AllocatorWithDefaultOptions *allocator;
    std::string strModelPath;
    std::string strTestImg;

    size_t numInputNodes;
    size_t numOutputNodes;

    InputType input_image;
    // ResultType result;

    std::vector<std::string> vecImglist;

    std::vector<float> input_tensor_value;
    std::vector<float> output_tensor_value;
    std::vector<const char *> output_node_names;
    std::vector<const char *> input_nodes_names;

    std::vector<int64_t> input_node_dims;
    std::array<int64_t, 4> out_shape;

    cv::Mat mPredict;
    cv::Mat img;

public:
    MyOnnxPack(std::string modelpath, std::string testImg);
    MyOnnxPack(std::string modelpath, std::vector<std::string> &);
    void InitializeOnnxEnv();
    void normalize(cv::Mat &img);
    void PrintfInputInfo();
    void Mat2ChannelLast(cv::Mat &, float *);
    void Mat2ChannelFirst(cv::Mat &, float *);
    std::vector<const char *> GetOutPutName();
    // cv::Mat *GetImgFromVector(ResultType &);
    void ReadImg();
    void ReadImg(std::string path);
    void InferenceImg();
    void InferenceVecImg();
    std::string ConstructFilePath();
    ~MyOnnxPack();
    const float MEANS[3] = {123.68, 116.78, 103.94};
    const float STD[3] = {58.40, 57.12, 57.38};
    using DataOutputType = std::pair<float *, std::vector<int64_t>>;
};

#endif // MACOPENCV_ONNX_H
