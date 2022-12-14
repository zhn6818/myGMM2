/************************************************************************/
/* Acknowledgements: This GMM code was slightly modified by Kevin (Ke-Yun) Lin
   based on the source code from Gurpinder Singh Sandhu
   The original version can be found here
   https://github.com/gpsinghsandhu/Background-Subtraction-using-GMM    */
/************************************************************************/

// /*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
// License Agreement
//
// Copyright (C) 2013, Gurpinder Singh Sandhu, all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * The name of the copyright holders may not be used to endorse or promote products
// derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall Gurpinder Singh Sandhu be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.*/

#include "myGMM.h"
#include <iostream>



myGMM::myGMM(double LearningRate)
{
    N_start = NULL;

    alpha = LearningRate; // in use
    cT = 0.05;
    covariance0 = 11.0; // in use
    cf = 0.1;
    cfbar = 1.0 - cf; // in use
    temp_thr = 9.0 * covariance0 * covariance0;
    prune = -alpha * cT; // in use
    alpha_bar = 1.0 - alpha; // in use
    overall = 0;

    sum = 0.0;
    //count = 0;
    close = false;
    temp_cov = 0.0;
    weight = 0.0;
    var = 0.0;
}

#ifdef USE_OPENCV

void myGMM::initial(Mat &orig_img)
{
    for (int i = 0; i < orig_img.rows; i++)
    {
        r_ptr = orig_img.ptr(i);
        for (int j = 0; j < orig_img.cols; j++)
        {

            LNode *N_ptr = Create_Node(*r_ptr, *(r_ptr + 1), *(r_ptr + 2));
            if (N_ptr != NULL)
            {
                N_ptr->pixel_s->weight = 1.0;
                Insert_End_Node(N_ptr);
            }
            else
            {
                std::cout << "Memory limit reached... ";
                //_getch();
                exit(0);
            }
            r_ptr += 3;
        }
    }
}

#endif

myGMM::~myGMM()
{
    LNode* next_N_ptr;
    gaussian* next_ptr;
    N_ptr = N_start;
    while (N_ptr != NULL)
    {
        start = N_ptr->pixel_s;
        ptr = start;
        for (int i = 0; i < N_ptr->no_of_components; i++)
        {
            next_ptr = ptr->Next;
            delete ptr;
            ptr = next_ptr;
            
        }
        next_N_ptr = N_ptr->Next;
        delete N_ptr;
        N_ptr = next_N_ptr;
    }
    std::cout << "Destructor called myGMM" << std::endl;
}

void myGMM::ChangeLearningRate(float new_learn_rate)
{
    alpha = new_learn_rate;
}

LNode *myGMM::Create_Node(double info1, double info2, double info3)
{
    N_ptr = new LNode;
    if (N_ptr != NULL)
    {
        N_ptr->Next = NULL;
        N_ptr->no_of_components = 1;
        N_ptr->pixel_s = N_ptr->pixel_r = Create_gaussian(info1, info2, info3);
    }
    return N_ptr;
}

gaussian *myGMM::Create_gaussian(double info1, double info2, double info3)
{
    ptr = new gaussian;
    if (ptr != NULL)
    {
        ptr->mean[0] = info1;
        ptr->mean[1] = info2;
        ptr->mean[2] = info3;
        ptr->covariance = covariance0;
        ptr->weight = alpha;
        ptr->Next = NULL;
        ptr->Previous = NULL;
    }
    return ptr;
}

void myGMM::Insert_End_Node(LNode *np)
{
    if (N_start != NULL)
    {
        N_rear->Next = np;
        N_rear = np;
    }
    else
        N_start = N_rear = np;
}

void myGMM::Insert_End_gaussian(gaussian *nptr)
{
    if (start != NULL)
    {
        rear->Next = nptr;
        nptr->Previous = rear;
        rear = nptr;
    }
    else
        start = rear = nptr;
}

gaussian *myGMM::Delete_gaussian(gaussian *nptr)
{
    previous = nptr->Previous;
    next = nptr->Next;
    if (start != NULL)
    {
        if (nptr == start && nptr == rear)
        {
            start = rear = NULL;
            delete nptr;
        }
        else if (nptr == start)
        {
            next->Previous = NULL;
            start = next;
            delete nptr;
            nptr = start;
        }
        else if (nptr == rear)
        {
            previous->Next = NULL;
            rear = previous;
            delete nptr;
            nptr = rear;
        }
        else
        {
            previous->Next = next;
            next->Previous = previous;
            delete nptr;
            nptr = next;
        }
    }
    else
    {
        std::cout << "Underflow........";
        // getch();
        exit(0);
    }
    return nptr;
}

void myGMM::process(Mat &orig_img, Mat &bin_img, cv::Mat &maskRoi)
{
    N_ptr = N_start;
//    duration = static_cast<double>(cv::getTickCount());
    for (int i = 0; i < orig_img.rows; i++)
    {
        r_ptr = orig_img.ptr(i);
        b_ptr = bin_img.ptr(i);
        m_ptr = maskRoi.ptr(i);
        for (int j = 0; j < orig_img.cols; j++, N_ptr = N_ptr->Next)
        {

            sum = 0.0;
            // sum1 = 0.0;
            close = false;
            background = 255;

            rVal = *(r_ptr++);
            gVal = *(r_ptr++);
            bVal = *(r_ptr++);

            start = N_ptr->pixel_s;
            rear = N_ptr->pixel_r;
            ptr = start;

            temp_ptr = NULL;

            if (*m_ptr++ > 0)
            {

                if (N_ptr->no_of_components > 4)
                {
                    Delete_gaussian(rear);
                    N_ptr->no_of_components--;
                }

                for (int k = 0; k < N_ptr->no_of_components; k++)
                {

                    weight = ptr->weight;
                    mult = alpha / weight;
                    weight = weight * alpha_bar + prune;
                    if (close == false)
                    {
                        muR = ptr->mean[0];
                        muG = ptr->mean[1];
                        muB = ptr->mean[2];

                        dR = rVal - muR;
                        dG = gVal - muG;
                        dB = bVal - muB;

                        /*del[0] = value[0]-ptr->mean[0];
                            del[1] = value[1]-ptr->mean[1];
                            del[2] = value[2]-ptr->mean[2];*/

                        var = ptr->covariance;

                        mal_dist = (dR * dR + dG * dG + dB * dB);

                        if ((sum < cfbar) && (mal_dist < 16.0 * var * var))
                            background = 0;

                        if (mal_dist < 9.0 * var * var)
                        {
                            weight += alpha;
                            close = true;
                            ptr->mean[0] = muR + mult * dR;
                            ptr->mean[1] = muG + mult * dG;
                            ptr->mean[2] = muB + mult * dB;
                            temp_cov = var + mult * (mal_dist - var);
                            ptr->covariance = temp_cov < 5.0 ? 5.0 : (temp_cov > 20.0 ? 20.0 : temp_cov);
                            temp_ptr = ptr;
                        }
                    }

                    if (weight < -prune)
                    {
                        ptr = Delete_gaussian(ptr);
                        weight = 0;
                        N_ptr->no_of_components--;
                    }
                    else
                    {
                        // if(ptr->weight > 0)
                        sum += weight;
                        ptr->weight = weight;
                    }

                    ptr = ptr->Next;
                }

                if (close == false)
                {
                    ptr = new gaussian;
                    ptr->weight = alpha;
                    ptr->mean[0] = rVal;
                    ptr->mean[1] = gVal;
                    ptr->mean[2] = bVal;
                    ptr->covariance = covariance0;
                    ptr->Next = NULL;
                    ptr->Previous = NULL;
                    // Insert_End_gaussian(ptr);
                    if (start == NULL)
                        //
                        start = rear = NULL;
                    else
                    {
                        ptr->Previous = rear;
                        rear->Next = ptr;
                        rear = ptr;
                    }
                    temp_ptr = ptr;
                    N_ptr->no_of_components++;
                }

                ptr = start;
                while (ptr != NULL)
                {
                    ptr->weight /= sum;
                    ptr = ptr->Next;
                }

                while (temp_ptr != NULL && temp_ptr->Previous != NULL)
                {
                    if (temp_ptr->weight <= temp_ptr->Previous->weight)
                        break;
                    else
                    {
                        // count++;
                        next = temp_ptr->Next;
                        previous = temp_ptr->Previous;
                        if (start == previous)
                            start = temp_ptr;
                        previous->Next = next;
                        temp_ptr->Previous = previous->Previous;
                        temp_ptr->Next = previous;
                        if (previous->Previous != NULL)
                            previous->Previous->Next = temp_ptr;
                        if (next != NULL)
                            next->Previous = previous;
                        else
                            rear = previous;
                        previous->Previous = temp_ptr;
                    }

                    temp_ptr = temp_ptr->Previous;
                }

                N_ptr->pixel_s = start;
                N_ptr->pixel_r = rear;

                *b_ptr++ = background;
            }
            else
            {
                *b_ptr++ = 0;
            }
        }
    }
}

void myGMM::process(Mat &orig_img, Mat &bin_img)
{

    N_ptr = N_start;
//    duration = static_cast<double>(cv::getTickCount());
    for (int i = 0; i < orig_img.rows; i++)
    {
        r_ptr = orig_img.ptr(i);
        b_ptr = bin_img.ptr(i);
        for (int j = 0; j < orig_img.cols * 3; j += 3, N_ptr = N_ptr->Next)
        {
            sum = 0.0;
            // sum1 = 0.0;
            close = false;
            background = 255;

            rVal = *(r_ptr++);
            gVal = *(r_ptr++);
            bVal = *(r_ptr++);

            start = N_ptr->pixel_s;
            rear = N_ptr->pixel_r;
            ptr = start;

#if DEBUG_INFO
//            std::cout << "rVal: " << rVal << " gVal: " << gVal << " bVal: " << bVal  <<  std::endl;
            std::cout << "rows: " << i << " cols: " << j/3 << std::endl;
            std::string imgVal = "~~~~~~~~~~~~~bVal: " + std::to_string((int)bVal) + " gVal: " + std::to_string((int)gVal) + " rVal: " + std::to_string((int)rVal);
            green_print(imgVal);
            std::string components = " N_ptr->no_of_components: " + std::to_string(N_ptr->no_of_components);
            red_print(components);
#endif
            temp_ptr = NULL;

            if (N_ptr->no_of_components > 4)
            {
                Delete_gaussian(rear);
                N_ptr->no_of_components--;
            }

            for (int k = 0; k < N_ptr->no_of_components; k++)
            {

                weight = ptr->weight;
#if DEBUG_INFO
                std::cout << std::endl;
                std::cout << "node: " << k << std::endl;
                std::cout << "pre weight: " << weight << ",";
#endif
                mult = alpha / weight;
#if DEBUG_INFO
                std::cout << "  mult: " << mult << ",";
#endif
                weight = weight * alpha_bar + prune;
#if DEBUG_INFO
                std::cout << "  back weight: " << weight << std::endl;
#endif
                if (close == false)
                {
                    muR = ptr->mean[0];
                    muG = ptr->mean[1];
                    muB = ptr->mean[2];

                    dR = rVal - muR;
                    dG = gVal - muG;
                    dB = bVal - muB;
#if DEBUG_INFO
                    std::string valPtr = "~~~~~~~~~~~~~muB: " + std::to_string((int)muB) + " muG: " + std::to_string((int)muG) + " muR: " + std::to_string((int)muR);
                    green_print(valPtr);
                    std::cout << " dB: " << dB << "," << " dG: " << dG << "," << "  dR: " << dR << std::endl;
#endif
                    /*del[0] = value[0]-ptr->mean[0];
                        del[1] = value[1]-ptr->mean[1];
                        del[2] = value[2]-ptr->mean[2];*/

                    var = ptr->covariance;
#if DEBUG_INFO
                    std::cout << "  var: " << var << ",";
#endif
                    mal_dist = (dR * dR + dG * dG + dB * dB);
#if DEBUG_INFO
                    std::cout << "  mal_dist: " << mal_dist << "," << "  9*Var " << var * var * 9 << std::endl;
                    if(mal_dist < var * var * 9)
                        red_print("true update");
                    else
                        red_print("false update");
#endif
#if DEBUG_INFO
                    std::cout << "sum: " << sum << "," << " cfbar: " << cfbar << "," << " var: " << var << "," << " 16*Var " << var * var * 16 << std::endl;
#endif
                    if ((sum < cfbar) && (mal_dist < 16.0 * var * var))
                        background = 0;

                    if (mal_dist < 9.0 * var * var)
                    {

                        weight += alpha;
                        // mult = mult < 20.0*alpha ? mult : 20.0*alpha;

                        close = true;

                        ptr->mean[0] = muR + mult * dR;
                        ptr->mean[1] = muG + mult * dG;
                        ptr->mean[2] = muB + mult * dB;
                        // if( mult < 20.0*alpha)
                        // temp_cov = ptr->covariance*(1+mult*(mal_dist - 1));
                        temp_cov = var + mult * (mal_dist - var);
                        ptr->covariance = temp_cov < 5.0 ? 5.0 : (temp_cov > 20.0 ? 20.0 : temp_cov);
                        temp_ptr = ptr;
#if DEBUG_INFO
                        red_print("need update weight: ");
                        std::cout << "study weight: " << weight << "," << " close: " << close << ","  << " temp_cov: " << temp_cov << ","  << " ptr->covariance : " << ptr->covariance << "," << std::endl;
#endif
                    }

                }
#if DEBUG_INFO
                std::cout << "-prune: " << -prune << std::endl;
#endif
                if (weight < -prune)
                {
                    ptr = Delete_gaussian(ptr);
                    weight = 0;
                    N_ptr->no_of_components--;
                }
                else
                {
                    // if(ptr->weight > 0)
                    sum += weight;
                    ptr->weight = weight;
                }

                ptr = ptr->Next;
#if DEBUG_INFO
                std::string sumval = "sum: " + std::to_string(sum);
                red_print(sumval);
#endif
            }
#if DEBUG_INFO
            if(close == false)
            {
                red_print("need new Gaussian");
            }else{
                green_print("dont new Gaussian");
            }

#endif
            if (close == false)
            {
                ptr = new gaussian;
                ptr->weight = alpha;
                ptr->mean[0] = rVal;
                ptr->mean[1] = gVal;
                ptr->mean[2] = bVal;
                ptr->covariance = covariance0;
                ptr->Next = NULL;
                ptr->Previous = NULL;
                // Insert_End_gaussian(ptr);
                if (start == NULL)
                    // ??
                    start = rear = NULL;
                else
                {
                    ptr->Previous = rear;
                    rear->Next = ptr;
                    rear = ptr;
                }
                temp_ptr = ptr;
                N_ptr->no_of_components++;
            }

            ptr = start;
            while (ptr != NULL)
            {
#if DEBUG_INFO
                std::cout << "pre ptr->weight:" << ptr->weight;
#endif
                ptr->weight /= sum;
#if DEBUG_INFO
                std::cout << "  bac ptr->weight:" << ptr->weight;
                std::cout << ", meanval: mean[0]: " << ptr->mean[0] << ", mean[1]: " << ptr->mean[1] << ", mean[2]: " << ptr->mean[2] << ", weight: " << ptr->weight << ", covariance: " << ptr->covariance ;
                std::cout << std::endl;
#endif
                ptr = ptr->Next;
            }

            while (temp_ptr != NULL && temp_ptr->Previous != NULL)
            {
                if (temp_ptr->weight <= temp_ptr->Previous->weight)
                    break;
                else
                {
                    // count++;
                    next = temp_ptr->Next;
                    previous = temp_ptr->Previous;
                    if (start == previous)
                        start = temp_ptr;
                    previous->Next = next;
                    temp_ptr->Previous = previous->Previous;
                    temp_ptr->Next = previous;
                    if (previous->Previous != NULL)
                        previous->Previous->Next = temp_ptr;
                    if (next != NULL)
                        next->Previous = previous;
                    else
                        rear = previous;
                    previous->Previous = temp_ptr;
                }

                temp_ptr = temp_ptr->Previous;
            }

            N_ptr->pixel_s = start;
            N_ptr->pixel_r = rear;
#if DEBUG_INFO
            std::cout << std::endl;
            red_print("end");
#endif
            *b_ptr++ = background;

            ptr = start;
            while (ptr != NULL)
            {
#if DEBUG_INFO
                // 
                std::cout << "  bac ptr->weight:" << ptr->weight;
                std::cout << ", meanval: mean[0]: " << ptr->mean[0] << ", mean[1]: " << ptr->mean[1] << ", mean[2]: " << ptr->mean[2] << ", weight: " << ptr->weight << ", covariance: " << ptr->covariance ;
                std::cout << std::endl;
#endif
                ptr = ptr->Next;
            }
        }
    }
}
