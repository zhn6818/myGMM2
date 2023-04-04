#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaobjdetect.hpp>
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
#include <memory>
#include "gpuGmm.h"
#include "virgoGmm.h"
#include "myGMM.h"
#include "comman.h"

// using namespace cv;
// using namespace cuda;

int main(int argc, char **argv)
{
    // test();
    // cv::cuda::GpuMat img;
    cv::VideoCapture capture("/data2/zhn/video/1.mp4");
    if (!capture.isOpened())
    {
        std::cout << "fail to opencv video" << std::endl;
    }
    cv::Mat frame;
    long currentFrame = 0;

    cv::Size writerSize = cv::Size(capture.get(CAP_PROP_FRAME_WIDTH) * 3, capture.get(CAP_PROP_FRAME_HEIGHT) * 3);

    std::shared_ptr<GpuGmm> ptGpu = std::make_shared<GpuGmm>(0.04);
    std::shared_ptr<GpuGmm> ptGpu2 = std::make_shared<GpuGmm>(0.0002);

    cv::VideoWriter writer;
    std::string videoWritePath = "./result/result6.avi";
    std::string imgStringSrc = "/data2/zhn/video/result/";

    writer.open(videoWritePath.c_str(), cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, writerSize);

    while (true)
    {
        if (!capture.read(frame))
        {
            break;
        }
        // cv::resize(frame, frame, cv::Size(writerSize.width, writerSize.height));
        cv::Mat imageTmp = cv::Mat(writerSize.height, writerSize.width, CV_8UC3, cv::Scalar::all(0));
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------current Frame Id: -------------------    " << currentFrame << std::endl;
        currentFrame++;
        // cv::medianBlur(frame, frame, 5);
        cv::Mat frame3 = frame.clone();
        cv::Mat frame2;
        // cv::cvtColor(frame3, frame2, cv::COLOR_LBGR2Lab);
        frame2 = frame.clone();
        if (currentFrame == 1)
        {
            ptGpu->initial(frame2);
            ptGpu2->initial(frame3);
        }

        cv::Mat bin_img = cv::Mat(frame2      .rows, frame2.cols, CV_8UC1, cv::Scalar::all(0));
        ptGpu->process(frame2, bin_img);
        cv::Mat imgSrc = ptGpu->getBackImg();
        cv::Mat roiBackImg = imageTmp(cv::Rect(0, frame2.rows, frame2.cols, frame2.rows));
        imgSrc.copyTo(roiBackImg);

        cv::Mat bin_img2 = cv::Mat(frame3.rows, frame3.cols, CV_8UC1, cv::Scalar::all(0));
        ptGpu2->process(frame3, bin_img2);
        cv::Mat imgSrc2 = ptGpu2->getBackImg();
        cv::Mat roiBackImg2 = imageTmp(cv::Rect(frame2.cols, frame2.rows, frame2.cols, frame2.rows));
        imgSrc2.copyTo(roiBackImg2);

        // if (currentFrame % 100 == 0)
        // {
        // cv::resize(frame, frame, cv::Size(512, 512));
        // cv::resize(imgSrc2, imgSrc2, cv::Size(512, 512));
        // std::string imgName = imgStringSrc + "AA/" + std::to_string(currentFrame) + ".png";
        // std::string bgName = imgStringSrc + "BB/" + std::to_string(currentFrame) + ".png";
        // cv::imwrite(imgName, frame);
        // cv::imwrite(bgName, imgSrc2);
        // }

        // cv::Mat diffImgSrc = imgSrc - imgSrc2;
        // cv::Mat roiDiff = imageTmp(cv::Rect(0, frame2.rows * 2, frame2.cols, frame2.rows));
        // diffImgSrc.copyTo(roiDiff);

        // cv::cuda::GpuMat resultGpu = cv::cuda::GpuMat(frame2.rows, frame2.cols, CV_8UC3, cv::Scalar::all(0));

        // cv::cuda::GpuMat imgSrcGpu, imgSrc2Gpu, frame2Gpu;
        // imgSrcGpu.upload(imgSrc);
        // imgSrc2Gpu.upload(imgSrc2);
        // frame2Gpu.upload(frame2);
        // processDiff(imgSrcGpu, imgSrc2Gpu, frame2Gpu, resultGpu);

        // cv::Mat result;
        // resultGpu.download(result);
        // cv::Mat roiresult = imageTmp(cv::Rect(frame2.cols, frame2.rows * 2, frame2.cols, frame2.rows));
        // result.copyTo(roiresult);

        // cv::Mat colorB;
        // cv::cvtColor(bin_img, colorB, cv::COLOR_GRAY2BGR);
        // cv::Mat roiTmpColor = imageTmp(cv::Rect(0, 0, frame2.cols, frame2.rows));
        // colorB.copyTo(roiTmpColor);

        // cv::Mat colorB2;
        // cv::cvtColor(bin_img2, colorB2, cv::COLOR_GRAY2BGR);
        // cv::Mat roiTmpColor2 = imageTmp(cv::Rect(frame2.cols, 0, frame2.cols, frame2.rows));
        // colorB2.copyTo(roiTmpColor2);

        // cv::Mat imgRoi = imageTmp(cv::Rect(frame2.cols * 2, frame2.rows, frame2.cols, frame2.rows));
        // frame2.copyTo(imgRoi);

        // cv::putText(imageTmp, std::to_string(currentFrame), cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
        writer.write(imageTmp);
        }
    writer.release();

    return 0;
}
