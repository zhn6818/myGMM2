#include "virgoGmm.h"

// void red_print(std::string out){
//     std::cout << "\033[31;1m" << out << "\033[0m" << std::endl;
// }
// void green_print(std::string out){
//     std::cout << "\033[32;1m" << out << "\033[0m" << std::endl;
// }

virGoGmm::virGoGmm(double LearningRate)
{
    alpha = LearningRate; // in use
    cT = 0.05;
    covariance0 = 11.0; // in use
    cf = 0.1;
    cfbar = 1.0 - cf; // in use
    temp_thr = 9.0 * covariance0 * covariance0;
    prune = -alpha * cT;     // in use
    alpha_bar = 1.0 - alpha; // in use
}
virGoGmm::~virGoGmm()
{
}

void virGoGmm::ChangeLearningRate(float new_learn_rate)
{
    alpha = new_learn_rate;
}
void virGoGmm::initial(cv::Mat &orig_img)
{

    vectorField = std::vector<std::vector<NodePixel>>(orig_img.rows, std::vector<NodePixel>(orig_img.cols));

    for (int i = 0; i < orig_img.rows; i++)
    {
        r_ptr = orig_img.ptr(i);
        for (int j = 0; j < orig_img.cols; j++)
        {
            std::shared_ptr<Gaussian> ptrGuass = std::make_shared<Gaussian>(*r_ptr, *(r_ptr + 1), *(r_ptr + 2));
            if (ptrGuass)
            {
                ptrGuass->setWight(1.0);
                ptrGuass->setCov(covariance0);
                vectorField[i][j].Push(ptrGuass);
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
void virGoGmm::process(cv::Mat &orig_img, cv::Mat &bin_img)
{
    for (int i = 0; i < orig_img.rows; i++)
    {
        r_ptr = orig_img.ptr(i);
        b_ptr = bin_img.ptr(i);
        for (int j = 0; j < orig_img.cols; j++)
        {
            sum = 0.0;
            close = false;
            background = 255;

            rVal = *(r_ptr++);
            gVal = *(r_ptr++);
            bVal = *(r_ptr++);
#if DEBUG_INFO
            //            std::cout << "rVal: " << rVal << " gVal: " << gVal << " bVal: " << bVal  <<  std::endl;

            std::cout << "rows: " << i << " cols: " << j << std::endl;

            std::string imgVal = "~~~~~~~~~~~~~bVal: " + std::to_string((int)bVal) + " gVal: " + std::to_string((int)gVal) + " rVal: " + std::to_string((int)rVal);
            green_print(imgVal);
            std::string components = " N_ptr->no_of_components: " + std::to_string(vectorField[i][j].realSize);
            red_print(components);
#endif
            int ldTmp = 1;

            vectorField[i][j].CheckSize();
            for (int k = 0; k < vectorField[i][j].realSize; k++)
            {
                weight = vectorField[i][j].gaussianVec[k]->weight;
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
                    muR = vectorField[i][j].gaussianVec[k]->mean[0];
                    muG = vectorField[i][j].gaussianVec[k]->mean[1];
                    muB = vectorField[i][j].gaussianVec[k]->mean[2];

                    dR = rVal - muR;
                    dG = gVal - muG;
                    dB = bVal - muB;
#if DEBUG_INFO
                    std::string valPtr =
                        "~~~~~~~~~~~~~muB: " + std::to_string((int)muB) + " muG: " + std::to_string((int)muG) +
                        " muR: " + std::to_string((int)muR);
                    green_print(valPtr);
                    std::cout << " dB: " << dB << ","
                              << " dG: " << dG << ","
                              << "  dR: " << dR << std::endl;
#endif

                    var = vectorField[i][j].gaussianVec[k]->covariance;
#if DEBUG_INFO
                    std::cout << "  var: " << var << ",";
#endif
                    mal_dist = (dR * dR + dG * dG + dB * dB);
#if DEBUG_INFO
                    std::cout << "  mal_dist: " << mal_dist << ","
                              << "  9*Var " << var * var * 9 << std::endl;
                    if (mal_dist < var * var * 9)
                        red_print("true update");
                    else
                        red_print("false update");
#endif
#if DEBUG_INFO
                    std::cout << "sum: " << sum << ","
                              << " cfbar: " << cfbar << ","
                              << " var: " << var << ","
                              << " 16*Var " << var * var * 16 << std::endl;
#endif
                    if ((sum < cfbar) && (mal_dist < 16.0 * var * var))
                        background = 0;

                    if (mal_dist < 9.0 * var * var)
                    {

                        weight += alpha;
                        // mult = mult < 20.0*alpha ? mult : 20.0*alpha;

                        close = true;

                        vectorField[i][j].gaussianVec[k]->mean[0] = muR + mult * dR;
                        vectorField[i][j].gaussianVec[k]->mean[1] = muG + mult * dG;
                        vectorField[i][j].gaussianVec[k]->mean[2] = muB + mult * dB;
                        // if( mult < 20.0*alpha)
                        // temp_cov = ptr->covariance*(1+mult*(mal_dist - 1));
                        temp_cov = var + mult * (mal_dist - var);

                        ldTmp = std::min(NGaussian - k, vectorField[i][j].realSize - k);
                        vectorField[i][j].gaussianVec[k]->covariance = temp_cov < 5.0 ? 5.0 : (temp_cov > 20.0 ? 20.0 : temp_cov);
#if DEBUG_INFO
                        red_print("need update weight: ");
                        std::cout << "study weight: " << weight << ","
                                  << " close: " << close << ","
                                  << " temp_cov: " << temp_cov << ","
                                  << " ptr->covariance : " << vectorField[i][j].gaussianVec[k]->covariance << "," << std::endl;
#endif
                    }
                }
#if DEBUG_INFO
                std::cout << "-prune: " << -prune << std::endl;
#endif
                if (weight < -prune)
                {
                    vectorField[i][j].Erase(k);
                    //                    ptr = Delete_gaussian(ptr);
                    //                    weight = 0;
                    //                    N_ptr->no_of_components--;
                }
                else
                {
                    // if(ptr->weight > 0)
                    sum += weight;
                    vectorField[i][j].gaussianVec[k]->weight = weight;
                }
#if DEBUG_INFO
                std::string sumval = "sum: " + std::to_string(sum);
                red_print(sumval);
#endif
            }
#if DEBUG_INFO
            if (close == false)
            {
                red_print("need new Gaussian");
            }
            else
            {
                green_print("dont new Gaussian");
            }

#endif

            if (close == false)
            {
                ldTmp = 1;
                std::shared_ptr<Gaussian> ptrGuass = std::make_shared<Gaussian>(rVal, gVal, bVal);
                ptrGuass->setCov(covariance0);
                ptrGuass->setWight(alpha);
                vectorField[i][j].Push(ptrGuass);
            }

            for (int m = 0; m < vectorField[i][j].gaussianVec.size(); m++)
            {
#if DEBUG_INFO
                std::cout << "pre ptr->weight:" << vectorField[i][j].gaussianVec[m]->weight;
#endif
                vectorField[i][j].gaussianVec[m]->weight /= sum;
#if DEBUG_INFO
                std::cout << "  bac ptr->weight:" << vectorField[i][j].gaussianVec[m]->weight;
                std::cout << ", meanval: mean[0]: " << vectorField[i][j].gaussianVec[m]->mean[0] << ", mean[1]: " << vectorField[i][j].gaussianVec[m]->mean[1] << ", mean[2]: " << vectorField[i][j].gaussianVec[m]->mean[2] << ", weight: " << vectorField[i][j].gaussianVec[m]->weight << ", covariance: " << vectorField[i][j].gaussianVec[m]->covariance;
                std::cout << std::endl;
#endif
            }
            // if (ldTmp)
            // {
            //     vectorField[i][j].Sort(ldTmp);
            // }

            vectorField[i][j].Sort();

            *b_ptr++ = background;
#if DEBUG_INFO
            for (int m = 0; m < vectorField[i][j].gaussianVec.size(); m++)
            {
                red_print("end");
                std::cout << "  bac ptr->weight:" << vectorField[i][j].gaussianVec[m]->weight;
                std::cout << ", meanval: mean[0]: " << vectorField[i][j].gaussianVec[m]->mean[0] << ", mean[1]: " << vectorField[i][j].gaussianVec[m]->mean[1] << ", mean[2]: " << vectorField[i][j].gaussianVec[m]->mean[2] << ", weight: " << vectorField[i][j].gaussianVec[m]->weight << ", covariance: " << vectorField[i][j].gaussianVec[m]->covariance;
                std::cout << std::endl;
            }
#endif
        }
    }
}
