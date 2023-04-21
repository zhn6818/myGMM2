#include "gpuGmm.h"
#include "comman.h"

GpuGmm::GpuGmm(double LearningRate)
{
    alpha = LearningRate; // in use
    cT = 0.05;
    covariance0 = 11.0; // in use
    cf = 0.1;
    cfbar = 1.0 - cf; // in use
    temp_thr = 9.0 * covariance0 * covariance0;
    prune = -alpha * cT;     // in use
    alpha_bar = 1.0 - alpha; // in use
    // NodePixelGpu
}

GpuGmm::~GpuGmm()
{
}

void GpuGmm::initial(cv::Mat &orig_img)
{
#if DEBUGINFO
    std::cout << "orig_img: " << orig_img.step << std::endl;
    std::cout << "sss: " << sizeof(NodePixelGpu) << std::endl;
    std::cout << orig_img.cols << " " << orig_img.rows << std::endl;
#endif
    node = new NodePixelGpu[orig_img.cols * orig_img.rows];
#if DEBUGINFO
    std::cout << "img" << std::endl;
    for (int i = 0; i < orig_img.rows; i++)
    {
        for (int j = 0; j < orig_img.cols; j++)
        {
            std::cout << (int)orig_img.at<cv::Vec3b>(i, j)[0] << "," << (int)orig_img.at<cv::Vec3b>(i, j)[1] << "," << (int)orig_img.at<cv::Vec3b>(i, j)[2] << std::endl;
        }
    }

    std::cout << "img" << std::endl;
#endif
    cudaMalloc((void **)&devArray, orig_img.cols * orig_img.rows * sizeof(NodePixelGpu));
    std::cout << "orig_img: " << (int)orig_img.at<cv::Vec3b>(450, 650)[0] << " " << (int)orig_img.at<cv::Vec3b>(450, 650)[1] << " " << (int)orig_img.at<cv::Vec3b>(450, 650)[2] << std::endl;
    tmpImg.upload(orig_img);

    InitNode(tmpImg, devArray, covariance0);

    cudaMemcpy(node, devArray, orig_img.cols * orig_img.rows * sizeof(NodePixelGpu), cudaMemcpyDeviceToHost);

    GmmImg = cv::cuda::GpuMat(orig_img.rows, orig_img.cols, CV_8UC3, cv::Scalar::all(0));

#if DEBUGINFO
    for (int i = 0; i < orig_img.cols * orig_img.rows; i++)
    {
        std::cout << "realSize: " << node[i].realSize << " ";
        for (int j = 0; j < node[i].realSize; j++)
        {
            std::cout << "mean: " << node[i].gaussian[j].mean[0] << "," << node[i].gaussian[j].mean[1] << "," << node[i].gaussian[j].mean[2] << "," << node[i].gaussian[j].weight << "," << node[i].gaussian[j].covariance << std::endl;
        }
    }

    std::cout
        << std::endl;
#endif
}

void GpuGmm::process(cv::Mat &orig_img, cv::Mat &bin_img)
{
    tmpImg.upload(orig_img);
    outImg.upload(bin_img);
    processNode(tmpImg, outImg, devArray, covariance0, alpha, alpha_bar, prune, cfbar);
    outImg.download(bin_img);
    
    // std::cout << "bin img : " << (int)bin_img.at<uchar>(0, 0) << std::endl;
    // std::cout << std::endl;
}

cv::Mat GpuGmm::getBackImg()
{
    GetNode(GmmImg, devArray);
    GmmImg.download(imgGmm);
    return imgGmm;
}