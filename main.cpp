//
// Created by 张海宁 on 2022/7/5.
//
#include <iostream>
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
#include "myGMM.h"
#include "virgoGmm.h"
#include "gpuGmm.h"
#include "comman.h"

struct pixelFSM
{
    // short  long   state
    //   0    0        0
    //   0    1        1
    //   1    0        2
    //   1    1        3
public:
    int state_now;
    int state_pre;
    int static_count;
    int dispear_count;
    bool staticFG_candidate;
    bool staticFG_stable;

    pixelFSM()
    {
        state_now = 0;
        state_pre = 0;
        static_count = 0;
        dispear_count = 0;
        staticFG_candidate = false;
        staticFG_stable = false;
    }
};

int main(int argc, char **argv)
{

    int SizeW = 800;
    int SizeH = 500;

    cv::VideoCapture capture("./test1/灰色_20_1（傍晚）_B-9.mp4");
    if (!capture.isOpened())
    {
        std::cout << "fail to opencv video" << std::endl;
    }
    cv::Mat frame;
    long currentFrame = 0;

    std::shared_ptr<GpuGmm> ptGpu = std::make_shared<GpuGmm>(0.004);
    std::shared_ptr<GpuGmm> ptGpu2 = std::make_shared<GpuGmm>(0.0002);

    // std::shared_ptr<myGMM> ptGpu = std::make_shared<myGMM>(0.004);
    // std::shared_ptr<myGMM> ptGpu2 = std::make_shared<myGMM>(0.0002);

    cv::Size writerSize = cv::Size(SizeW * 3, SizeH * 3);

    cv::Mat bin_img1 = cv::Mat(SizeH, SizeW, CV_8UC1, cv::Scalar::all(0));
    cv::Mat bin_img2 = cv::Mat(SizeH, SizeW, CV_8UC1, cv::Scalar::all(0));

    cv::VideoWriter writer;
    std::string videoWritePath = "./result/result.avi";

    writer.open(videoWritePath.c_str(), cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, writerSize);

    while (true)
    {
        if (!capture.read(frame))
        {
            break;
        }
        cv::Mat imgFrame = cv::Mat(SizeH * 3, SizeW * 3, CV_8UC3, cv::Scalar::all(0));
        cv::resize(frame, frame, cv::Size(SizeW, SizeH));
        cv::Mat frame2 = frame;

        if (currentFrame == 0)
        {
            ptGpu->initial(frame2);
            ptGpu2->initial(frame2);
        }

        ptGpu->process(frame2, bin_img1);
        ptGpu2->process(frame2, bin_img2);

        // cv::Mat structureElement = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), Point(-1, -1));
        // cv::dilate(bin_img1, bin_img1, structureElement, cv::Point(-1, -1));
        // cv::dilate(bin_img2, bin_img2, structureElement, cv::Point(-1, -1));

        cv::Mat tmp = imgFrame(cv::Rect(0, 0, (int)frame.cols, (int)frame.rows));
        cv::putText(frame, std::to_string(currentFrame), cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
        frame.copyTo(tmp);

        cv::Mat bin3 = imgFrame(cv::Rect(frame.cols, 0, frame.cols, frame.rows));
        cv::Mat colorB3;
        cv::cvtColor(bin_img1, colorB3, COLOR_GRAY2BGR);
        colorB3.copyTo(bin3);

        cv::Mat bin2 = imgFrame(cv::Rect(frame.cols * 2, 0, frame.cols, frame.rows));
        cv::Mat colorB2;
        cv::cvtColor(bin_img2, colorB2, COLOR_GRAY2BGR);
        colorB2.copyTo(bin2);

        cv::Mat imgBack = ptGpu->getBackImg();
        cv::Mat bin4 = imgFrame(cv::Rect(0, frame.rows, frame.cols, frame.rows));
        imgBack.copyTo(bin4);

        cv::Mat imgBack2 = ptGpu2->getBackImg();
        cv::Mat bin5 = imgFrame(cv::Rect(frame.cols, frame.rows, frame.cols, frame.rows));
        imgBack2.copyTo(bin5);

        writer.write(imgFrame);
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------current Frame Id: -------------------    " << currentFrame++ << std::endl;
    }

    writer.release();
    return 0;
}
