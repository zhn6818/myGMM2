#include "threadpool.h"

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
#include <memory.h>
#include <vector>
#include <ctime>
#include <mutex>
#include <unistd.h>
#include <thread>

class Memtest
{
public:
    cv::Mat img;
    cv::Mat dst;
    int count = 0;
    std::condition_variable condition;
    std::mutex _mutex, _mutex_cond;

    Memtest() {}
    ~Memtest(){};
    void copy(cv::Mat &imgin);
    cv::Mat getImg() { return dst; }
    void mm(uchar *dst, uchar *img, int step, int ii);
};

void Memtest::mm(uchar *dst, uchar *img, int step, int ii)
{
    int step1 = ii;
    // memcpy(dst, img, step);

    int waittime = (128 - step1) * 10 + 10;
    // cv::waitKey(waittime);
    usleep(waittime * 1000);
    _mutex.lock();
    count--;
    _mutex.unlock();
    std::cout << "thread: " << ii << "waittime: " << waittime << "count: " << count << std::endl;
}

void Memtest::copy(cv::Mat &imgin)
{
    img = imgin;
    dst = imgin.clone().setTo(0);
    std::vector<int> results;

    for (int i = 0; i < imgin.rows; i++)
    {
        _mutex.lock();
        count++;
        _mutex.unlock();

        std::thread th([&](uchar *srcptr, uchar *dstptr, int size, int ii)
                       { mm(dstptr, srcptr, size, ii);return 0; },
                       img.ptr<uchar>(i), dst.ptr<uchar>(i), imgin.step, i);
        th.detach();
    }

    std::unique_lock<std::mutex> lock(_mutex_cond);
    condition.wait(lock, [=]()
                   { return count != 0; });
}

int main(int argc, char **argv)
{
    std::cout << "thread test" << std::endl;
    cv::Mat img = cv::Mat(128, 128, CV_8UC1, cv::Scalar::all(255));
    cv::Mat test = cv::Mat(128, 128, CV_8UC1, cv::Scalar::all(0));

    // Memtest *m = new Memtest();
    std::shared_ptr<Memtest> m = std::make_shared<Memtest>();
    m->copy(img);
    cv::Mat imgimg = m->getImg();
    cv::imwrite("./data/resultthread.png", imgimg);

    return 0;
}