#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <memory>
#include "gpuGmm.h"

// using namespace cv;
// using namespace cuda;

int main(int argc, char **argv)
{
    // test();
    // cv::cuda::GpuMat img;
    cv::VideoCapture capture("./50035.mp4");
    if (!capture.isOpened())
    {
        std::cout << "fail to opencv video" << std::endl;
    }
    cv::Mat frame;
    long currentFrame = 0;

    const int sizeeeH = 800;
    const int sizeeeW = 1000;

    std::shared_ptr<GpuGmm> ptGpu = std::make_shared<GpuGmm>(0.004);

    cv::VideoWriter writer;
    std::string videoWritePath = "./result/result2.avi";
    cv::Size writerSize = cv::Size(sizeeeW, sizeeeH);
    writer.open(videoWritePath.c_str(), cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, writerSize);

    while (true)
    {
        if (!capture.read(frame))
        {
            break;
        }
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------current Frame Id: -------------------    " << currentFrame << std::endl;
        currentFrame++;

        cv::Mat frame2 = frame(cv::Rect(100, 100, sizeeeW, sizeeeH)).clone();
        if (currentFrame == 1)
        {
            ptGpu->initial(frame2);
        }
        cv::Mat bin_img = cv::Mat(frame2.rows, frame2.cols, CV_8UC1, cv::Scalar::all(0));
        ptGpu->process(frame2, bin_img);

        cv::Mat colorB;
        cv::cvtColor(bin_img, colorB, cv::COLOR_GRAY2BGR);
        writer.write(colorB);
        // cv::waitKey(30);
        // std::cout << "this img info : " << frame.cols << " " << frame.rows << std::endl;
    }
    writer.release();

    return 0;
}
