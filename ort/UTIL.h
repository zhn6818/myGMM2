//
// Created by 张海宁 on 2020/8/19.
//

#ifndef MACOPENCV_UTIL_H
#define MACOPENCV_UTIL_H


#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <opencv2/opencv.hpp>


namespace HN_UTIL{
    static const std::string class_names[] = { "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" };
#ifdef _WIN32
    static const string prefix = "\\";
#else
    static const std::string prefix = "/";
#endif
    static const char* POSFIX = ";jpg;png;bmp;jpeg;gif;";
    typedef std::vector<std::string> VecString;
    const std::string ConcatString(const std::string &path, const std::string &file);
    bool IsSupportPos(const std::string& posfix,const std::string& support);
    std::string GetFilePosfix(const char* path);
    VecString GetListFiles(const std::string& path, const std::string& exten = "*", bool isContainPath = true);
    //TF_Tensor* Mat2Tensor(cv::Mat &img, float normal = 1 / 255.0);
    int ArgMax(const std::vector<float> result);
}

#endif //MACOPENCV_UTIL_H
