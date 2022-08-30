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

int main(int argc, char **argv)
{

    cv::VideoCapture capture("./50035.mp4");
    if (!capture.isOpened())
    {
        std::cout << "fail to opencv video" << std::endl;
    }
    cv::Mat frame;
    long currentFrame = 0;
    std::shared_ptr<myGMM> _myGMM = std::make_shared<myGMM>(0.004);

    std::shared_ptr<virGoGmm> _myGMM22 = std::make_shared<virGoGmm>(0.004);

    const int sizeeeH = 800;
    const int sizeeeW = 500;
    cv::Mat bin_img = cv::Mat(sizeeeH, sizeeeW, CV_8UC1, cv::Scalar::all(0));
    cv::Mat bin_img2 = cv::Mat(sizeeeH, sizeeeW, CV_8UC1, cv::Scalar::all(0));

    cv::VideoWriter writer;
    std::string videoWritePath = "./result/result.avi";

    cv::Size writerSize = cv::Size(capture.get(CAP_PROP_FRAME_WIDTH) + sizeeeW * 2, capture.get(CAP_PROP_FRAME_HEIGHT) + sizeeeH);
    writer.open(videoWritePath.c_str(), cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, writerSize);

    while (true)
    {
        cv::Mat imgFrame = cv::Mat(capture.get(CAP_PROP_FRAME_HEIGHT) + sizeeeH, capture.get(CAP_PROP_FRAME_WIDTH) + sizeeeW * 2, CV_8UC3, cv::Scalar::all(0));

        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------current Frame Id: -------------------    " << currentFrame << std::endl;
        if (!capture.read(frame))
        {
            break;
        }
        //        if(currentFrame >= 20)
        //        {
        //            std::cout << std::endl;
        //        }

        cv::Mat frame2 = frame(cv::Rect(100, 100, sizeeeW, sizeeeH));
        if (currentFrame == 0)
        {
            _myGMM22->initial(frame2);
            _myGMM->initial(frame2);
        }
        _myGMM->process(frame2, bin_img2);
        _myGMM22->process(frame2, bin_img);

        cv::Mat diff = bin_img2 - bin_img;
        //        std::vector<cv::Point> pp;
        //        cv::findNonZero(diff, pp);

        cv::rectangle(frame, cv::Rect(100, 100, sizeeeW, sizeeeH), cv::Scalar(0, 0, 255), 1);
        //        capture>>frame;
        //        if(frame.empty())
        //        {
        //            break;
        //        }

        cv::circle(frame, cv::Point(500, 500), 1, cv::Scalar(0, 0, 255), 2);
        currentFrame++;
        // cv::imshow("bin_img", bin_img);
        // cv::imshow("bin_img2", bin_img2);

        cv::Mat tmp = imgFrame(cv::Rect(0, 0, (int)frame.cols, (int)frame.rows));
        frame.copyTo(tmp);

        cv::Mat bin2 = imgFrame(cv::Rect(frame.cols, 0, sizeeeW, sizeeeH));
        cv::Mat colorB2;
        cv::cvtColor(bin_img2, colorB2, COLOR_GRAY2BGR);
        colorB2.copyTo(bin2);

        cv::Mat bin = imgFrame(cv::Rect(frame.cols + sizeeeW, 0, sizeeeW, sizeeeH));
        cv::Mat colorB;
        cv::cvtColor(bin_img, colorB, COLOR_GRAY2BGR);
        colorB.copyTo(bin);

        cv::Mat diffTmp = imgFrame(cv::Rect(0, frame.rows, sizeeeW, sizeeeH));
        cv::Mat colorDiff;
        cv::cvtColor(diff, colorDiff, COLOR_GRAY2BGR);
        colorDiff.copyTo(diffTmp);

        // cv::imshow("frame", frame);
        writer.write(imgFrame);
        cv::waitKey(30);
    }
    writer.release();
    return 0;
}
