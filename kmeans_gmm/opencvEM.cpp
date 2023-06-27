#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;
// 使用EM算法实现样本的聚类及预测

int main()
{
    const int N = 10;
    float arrHeight[N] = {30, 330, 331, 332, 319, 318, 40, 35, 60, 70};

    Mat samples(N, 1, CV_32FC1, arrHeight);
    std::cout << samples << std::endl;

    Mat labels; // 标注，不需要事先知道
    Ptr<EM> em_model = EM::create();
    em_model->setClustersNumber(2);
    em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
    em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
    em_model->trainEM(samples, noArray(), labels, noArray());

    Mat sample(1, 1, CV_32FC1);
    for (int i = 0; i < 360; i += 5)
    {
        sample.at<float>(0) = (float)i;
        int response = cvRound(em_model->predict2(sample, noArray())[1]);
        std::cout << i << " " << response << std::endl;
        // std::cout << std::endl;
    }

    return 0;
}

// int main()
// {
//     const int N = 5; // 分成4类
//     const int N1 = (int)sqrt((double)N);
//     // 定义四种颜色，每一类用一种颜色表示
//     const Scalar colors[] =
//         {
//             Scalar(0, 0, 255),
//             Scalar(0, 255, 0),
//             Scalar(0, 255, 255),
//             Scalar(255, 255, 0),
//             Scalar(255, 0, 0),
//             Scalar(128, 0, 255),
//         };
//     int i, j;
//     int nsamples = 30;                            // 100个样本点
//     Mat samples(nsamples, 2, CV_32FC1);            // 样本矩阵,100行2列，即100个坐标点
//     Mat img = Mat::zeros(Size(500, 500), CV_8UC3); // 待测数据，每一个坐标点为一个待测数据
//     // std::cout << samples.channels() << std::endl;
//     samples = samples.reshape(2, 0);
//     // std::cout << samples.channels() << std::endl;
//     // 循环生成四个类别样本数据，共样本100个，每类样本25个
//     for (i = 0; i < N; i++)
//     {
//         Mat samples_part = samples.rowRange(i * nsamples / N, (i + 1) * nsamples / N);
//         // 设置均值
//         Scalar mean(((i % N1) + 1) * img.rows / (N1 + 1),
//                     ((i / N1) + 1) * img.rows / (N1 + 1));
//         // 设置标准差
//         Scalar sigma(30, 30);
//         randn(samples_part, mean, sigma); // 根据均值和标准差，随机生成25个正态分布坐标点作为样本
//     }
//     std::cout << samples << " " << samples.channels() << std::endl;
//     samples = samples.reshape(1, 0);
//     std::cout << samples << " " << samples.channels() << std::endl;
//     // 训练分类器
//     Mat labels; // 标注，不需要事先知道
//     Ptr<EM> em_model = EM::create();
//     em_model->setClustersNumber(N);
//     em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
//     em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
//     em_model->trainEM(samples, noArray(), labels, noArray());
//     // 对每个坐标点进行分类，并根据类别用不同的颜色画出
//     Mat sample(1, 2, CV_32FC1);
//     for (i = 0; i < img.rows; i++)
//     {
//         for (j = 0; j < img.cols; j++)
//         {
//             sample.at<float>(0) = (float)j;
//             sample.at<float>(1) = (float)i;
//             // predict2返回的是double值，用cvRound进行四舍五入得到整型
//             // 此处返回的是两个值Vec2d，取第二个值作为样本标注
//             int response = cvRound(em_model->predict2(sample, noArray())[1]);
//             Scalar c = colors[response]; // 为不同类别设定颜色
//             circle(img, Point(j, i), 1, c * 0.75, FILLED);
//         }
//     }
//     // 画出样本点
//     for (i = 0; i < nsamples; i++)
//     {
//         Point pt(cvRound(samples.at<float>(i, 0)), cvRound(samples.at<float>(i, 1)));
//         circle(img, pt, 2, colors[labels.at<int>(i)], FILLED);
//     }
//     imwrite("EM.png", img);
//     waitKey(0);
//     return 0;
// }