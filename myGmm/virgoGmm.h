
#ifndef _VIRGOGMM_H_
#define _VIRGOGMM_H_
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <deque>
#include "utils.h"
#define DD  1

struct Gaussian
{
    double mean[3];
    double covariance;
    double weight;
    Gaussian()
    {
        mean[0] = 0.0;
        mean[1] = 0.0;
        mean[2] = 0.0;
        covariance = 0.0;
        weight = 0.0;
    }
    Gaussian(double b, double g, double r)
    {
        mean[0] = b;
        mean[1] = g;
        mean[2] = r;

    }
    void setWight(double weightIn)
    {
        this->weight = weightIn;
    }
    void setCov(double cov)
    {
        this->covariance = cov;
    }
};

static bool compareGausWeight(const std::shared_ptr<Gaussian> &a, const std::shared_ptr<Gaussian> &b)
{
    return a->weight > b->weight;
}

struct NodePixel
{
    int maxSize;
    int realSize;
    std::deque<std::shared_ptr<Gaussian>> gaussianVec;

    NodePixel()
    {
        maxSize = NGaussian;
        realSize = 0;
    }

    void Push(std::shared_ptr<Gaussian> &gauTmp)
    {
        this->gaussianVec.push_back(gauTmp);
        this->realSize++;
    }
    void Sort(int idx)
    {
        //v1版本
//        sort(this->gaussianVec.begin(), this->gaussianVec.end(), compareGausWeight);

        //v2版本
        for(auto rit = this->gaussianVec.rbegin() + idx - 1; rit != this->gaussianVec.rend() && (rit + 1) != this->gaussianVec.rend(); rit++)
        {
            if((*rit)->weight <= (*(rit+1))->weight)
            {
                break;
            } else{
                std::swap(*rit, *(rit+1));
                break;
            }
        }
    }
    void Sort()
    {
        sort(this->gaussianVec.begin(), this->gaussianVec.end(), compareGausWeight);
    }
    void CheckSize()
    {
        assert(this->realSize == this->gaussianVec.size());
        if (this->gaussianVec.size() > maxSize)
        {
            this->gaussianVec.pop_back();
            this->realSize--;
        }
    }
    void Erase(int k)
    {
        this->gaussianVec.erase(this->gaussianVec.begin() + k);
        this->realSize--;
    }
};

class virGoGmm{
    public:
        double alpha;
        double cT;
        double covariance0;
        double cf;
        double cfbar;
        double temp_thr;
        double prune;
        double alpha_bar;

        std::vector<std::vector<NodePixel>> vectorField;
        //run parameter
        double mal_dist;
        double sum;

        bool close;
        int background;
        double mult;
        //	double duration,duration1,duration2,duration3;
        double temp_cov;
        double weight;
        double var;
        double muR, muG, muB, dR, dG, dB, rVal, gVal, bVal;

        //img ptr

        unsigned char *r_ptr;
        unsigned char *b_ptr;
        unsigned char *m_ptr;
        virGoGmm(double LearningRate);
        ~virGoGmm();

        void ChangeLearningRate(float new_learn_rate);
        void initial(cv::Mat &orig_img);

         void process(cv::Mat &orig_img, cv::Mat &bin_img);
};

#endif