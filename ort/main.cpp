//
// Created by 张海宁 on 2022/1/7.
//

#include <iostream>

#include "Onnx.h"
#include "UTIL.h"

int main(int argc, char **argv)
{
    const char *filepath = "./imgDir";
    HN_UTIL::VecString sdf = HN_UTIL::GetListFiles(filepath);
    const char *modelpath1 = "./segment_model/test.onnx";
    MyOnnxPack *tmpOnnx = new MyOnnxPack(modelpath1, sdf);

    // tmpOnnx->PrintfInputInfo();
    // tmpOnnx->ReadImg();
    tmpOnnx->InferenceVecImg();

    delete tmpOnnx;
    return 0;
}