// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// using namespace std;
// using namespace cv;

// //声明CUDA纹理
// texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex1;
// texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex2;
// //声明CUDA数组
// cudaArray *cuArray1;
// cudaArray *cuArray2;
// //通道数
// cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();

// __global__ void weightAddKerkel(uchar *pDstImgData, int imgHeight, int imgWidth, int channels)
// {
//     const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//     const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

//     if (tidx < imgWidth && tidy < imgHeight)
//     {
//         float4 lenaBGR, moonBGR;
//         //使用tex2D函数采样纹理
//         lenaBGR = tex2D(refTex1, tidx, tidy);
//         moonBGR = tex2D(refTex2, tidx, tidy);

//         int idx = (tidy * imgWidth + tidx) * channels;
//         float alpha = 0.5;
//         pDstImgData[idx + 0] = (alpha * lenaBGR.x + (1 - alpha) * moonBGR.x) * 255;
//         pDstImgData[idx + 1] = (alpha * lenaBGR.y + (1 - alpha) * moonBGR.y) * 255;
//         pDstImgData[idx + 2] = (alpha * lenaBGR.z + (1 - alpha) * moonBGR.z) * 255;
//         pDstImgData[idx + 3] = 0;
//     }
// }

// int main()
// {
//     Mat Lena = imread("data/lena.jpg");
//     Mat moon = imread("data/moon.jpg");
//     cvtColor(Lena, Lena, COLOR_BGR2BGRA);
//     cvtColor(moon, moon, COLOR_BGR2BGRA);
//     int imgWidth = Lena.cols;
//     int imgHeight = Lena.rows;
//     int channels = Lena.channels();

//     //设置纹理属性
//     cudaError_t t;
//     refTex1.addressMode[0] = cudaAddressModeClamp;
//     refTex1.addressMode[1] = cudaAddressModeClamp;
//     refTex1.normalized = false;
//     refTex1.filterMode = cudaFilterModeLinear;
//     //绑定cuArray到纹理
//     cudaMallocArray(&cuArray1, &cuDesc, imgWidth, imgHeight);
//     t = cudaBindTextureToArray(refTex1, cuArray1);

//     refTex2.addressMode[0] = cudaAddressModeClamp;
//     refTex2.addressMode[1] = cudaAddressModeClamp;
//     refTex2.normalized = false;
//     refTex2.filterMode = cudaFilterModeLinear;
//     cudaMallocArray(&cuArray2, &cuDesc, imgWidth, imgHeight);
//     t = cudaBindTextureToArray(refTex2, cuArray2);

//     //拷贝数据到cudaArray
//     t = cudaMemcpyToArray(cuArray1, 0, 0, Lena.data, imgWidth * imgHeight * sizeof(uchar) * channels, cudaMemcpyHostToDevice);
//     t = cudaMemcpyToArray(cuArray2, 0, 0, moon.data, imgWidth * imgHeight * sizeof(uchar) * channels, cudaMemcpyHostToDevice);

//     //输出图像
//     Mat dstImg = Mat::zeros(imgHeight, imgWidth, CV_8UC4);
//     uchar *pDstImgData = NULL;
//     t = cudaMalloc(&pDstImgData, imgHeight * imgWidth * sizeof(uchar) * channels);

//     //核函数，实现两幅图像加权和
//     dim3 block(8, 8);
//     dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);
//     weightAddKerkel<<<grid, block, 0>>>(pDstImgData, imgHeight, imgWidth, channels);
//     cudaThreadSynchronize();

//     //从GPU拷贝输出数据到CPU
//     t = cudaMemcpy(dstImg.data, pDstImgData, imgWidth * imgHeight * sizeof(uchar) * channels, cudaMemcpyDeviceToHost);

//     // cv::Mat tt;
//     cv::cvtColor(dstImg,dstImg, COLOR_BGRA2BGR);
//     cv::imwrite("./data/result.png", dstImg);
//     //显示
//     // namedWindow("show");
//     // imshow("show", dstImg);
//     // waitKey(0);
//     return 0;
// }
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "comman.h"

// using namespace std;
// using namespace cv;

int main(int argc, char **argv)
{
    cv::Mat Lena = cv::imread("./data/lena.jpg");
    cv::Mat moon = cv::imread("./data/moon.jpg");
    cv::cvtColor(Lena, Lena, cv::COLOR_BGR2BGRA);
    cv::cvtColor(moon, moon, cv::COLOR_BGR2BGRA);

    int imgWidth = Lena.cols;
    int imgHeight = Lena.rows;
    int channels = Lena.channels();

    cv::Mat tmp;
    test(Lena.ptr<uchar>(), moon.ptr<uchar>(), imgWidth, imgHeight, channels, tmp);
    cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2BGR);
    cv::imwrite("./data/result.png", tmp);
    std::cout << "run over, has imwrite img" << std::endl;
    //显示
    return 0;
}