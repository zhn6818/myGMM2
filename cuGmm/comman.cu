#include "comman.h"


__global__ void initKernel(uchar* img, int step, int w, int h, float* nodeP, int nodeStep, double cov)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
    if (x >= w || y >= h)
        return;
    // if(y == 0 && x == 0)
    // {
    //     printf("hello");
    // }
    uchar b = img[y * step + x * 3];
    uchar g = img[y * step + x * 3 + 1];
    uchar r = img[y * step + x * 3 + 2];
    // printf("%d, %d, %d, %d, %d \n", x, y, b, g, r);
    
    int indexNode = x + y * w;
    char* data = (char*)nodeP + indexNode * nodeStep;

    NodePixelGpu *nodeDev = (NodePixelGpu *)(data);
    nodeDev->realSize = 0;
    int index = nodeDev->realSize;
    
    nodeDev->gaussian[index].mean[0] = b;
    nodeDev->gaussian[index].mean[1] = g;
    nodeDev->gaussian[index].mean[2] = r;
    nodeDev->gaussian[index].covariance = cov;
    nodeDev->gaussian[index].weight = 1.0;
    nodeDev->realSize = nodeDev->realSize + 1;
    // if(y == 500 && x == 500)
    // {
    //     printf("Dev realSize： %f \n", nodeDev->realSize);
    //     printf("%d, %d, %f, %f, %f %f\n", x, y, (float)nodeDev->gaussian[0].mean[0], (float)nodeDev->gaussian[0].mean[1], (float)nodeDev->gaussian[0].mean[2], nodeDev->realSize);
    //     printf("%d, %d %d \n", (int)b, (int)g, (int)r);
    // }
}


void InitNode(cv::cuda::GpuMat &tmpImg, float* nodeP, double cov)
{
    std::cout << "tmpImg: " << tmpImg.cols << " " << tmpImg.rows << std::endl;

    const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(tmpImg.cols, blockDim.x), iDivUp(tmpImg.rows,blockDim.y));
    initKernel<<<gridDim, blockDim>>>(tmpImg.ptr<uchar>(), tmpImg.step, tmpImg.cols, tmpImg.rows, nodeP, sizeof(NodePixelGpu), cov);

    cudaThreadSynchronize();

    std::cout << std::endl;
}


__global__ void processKernel(uchar* out, int outStep, uchar* img, int step, int w, int h, float* nodeP, int nodeStep, double cov, double alpha, double alpha_bar, double prune, double cfbar)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
    if (x >= w || y >= h)
        return;

    double rVal = img[y * step + x * 3];
    double gVal = img[y * step + x * 3 + 1];
    double bVal = img[y * step + x * 3 + 2];

    double sum = 0.0;
    bool close = false;
    int background = 255;
    float weight = 0.0;
    double mal_dist = 0.0;
    double mult = 0.0;
    double temp_cov;
    double var;
    double muR, muG, muB, dR, dG, dB;

    int indexNode = x + y * w;
    char* data = (char*)nodeP + indexNode * nodeStep;
    NodePixelGpu *nodeDev = (NodePixelGpu *)(data);

    // printf("%d, %d, %f, %f, %f \n", x, y, rVal, gVal, bVal);
    // if(y == 500 && x == 500)
    // {
    //     printf("%d, %d, %f, %f, %f %f\n", x, y, nodeDev->gaussian[0].mean[0], nodeDev->gaussian[0].mean[1], nodeDev->gaussian[0].mean[2], nodeDev->realSize);
    // }
    // if(x == 0 && y == 0)
    // {
        if(nodeDev->realSize > MaxSize)
        {
            nodeDev->realSize = nodeDev->realSize - 1;
            nodeDev->gaussian[MaxSize].mean[0] = 0;
            nodeDev->gaussian[MaxSize].mean[1] = 0;
            nodeDev->gaussian[MaxSize].mean[2] = 0;
            nodeDev->gaussian[MaxSize].covariance = 0;
            nodeDev->gaussian[MaxSize].weight = 0;
        }
        for(int k = 0; k < nodeDev->realSize; k++)
        {
            weight = nodeDev->gaussian[k].weight;
            mult = alpha / weight;
            weight = weight * alpha_bar + prune;
            // printf("weight: %f", weight);
            if(close == false)
            {
                muR = nodeDev->gaussian[k].mean[0];
                muG = nodeDev->gaussian[k].mean[1];
                muB = nodeDev->gaussian[k].mean[2];
                dR = rVal - muR;
                dG = gVal - muG;
                dB = bVal - muB;
                var = nodeDev->gaussian[k].covariance;
                mal_dist = (dR * dR + dG * dG + dB * dB);
                if ((sum < cfbar) && (mal_dist < 16.0 * var * var))
                {
                    background = 0;
                }
                if (mal_dist < 9.0 * var * var)
                {

                    weight += alpha;
                    close = true;
                    nodeDev->gaussian[k].mean[0] = muR + mult * dR;
                    nodeDev->gaussian[k].mean[1] = muG + mult * dG;
                    nodeDev->gaussian[k].mean[2] = muB + mult * dB;
                    temp_cov = var + mult * (mal_dist - var);
                    nodeDev->gaussian[k].covariance = temp_cov < 5.0 ? 5.0 : (temp_cov > 20.0 ? 20.0 : temp_cov);
                }

            }
            if(weight < -prune)
            {
                nodeDev->realSize = nodeDev->realSize - 1;
                nodeDev->gaussian[k].mean[0] = 0;
                nodeDev->gaussian[k].mean[1] = 0;
                nodeDev->gaussian[k].mean[2] = 0;
                nodeDev->gaussian[k].covariance = 0;
                nodeDev->gaussian[k].weight = 0;
            }else{
                sum += weight;
                nodeDev->gaussian[k].weight = weight;
            }
        }
        if (close == false)
        {
            int index = nodeDev->realSize;
            nodeDev->gaussian[index].mean[0] = rVal;
            nodeDev->gaussian[index].mean[1] = gVal;
            nodeDev->gaussian[index].mean[2] = bVal;
            nodeDev->gaussian[index].covariance = cov;
            nodeDev->gaussian[index].weight = alpha;
            nodeDev->realSize = nodeDev->realSize + 1;
        }
        for (int m = 0; m < nodeDev->realSize; m++)
        {
            nodeDev->gaussian[m].weight /= sum;
        }
        // printf("back %d, ", background);
        for (int m = nodeDev->realSize - 1; m > 0 && (m - 1) >= 0; m--)
        {
            if(nodeDev->gaussian[m].weight > nodeDev->gaussian[m - 1].weight)
            {
                // printf("%f, %f, %f \n", *((float*)nodeDev->gaussian[m]), *((float*)nodeDev->gaussian[m] + 1), *((float*)nodeDev->gaussian[m] + 2));
                float t1 = nodeDev->gaussian[m].mean[0];
                float t2 = nodeDev->gaussian[m].mean[1];
                float t3 = nodeDev->gaussian[m].mean[2];
                float t4 = nodeDev->gaussian[m].covariance;
                float t5 = nodeDev->gaussian[m].weight;
                nodeDev->gaussian[m].mean[0] = nodeDev->gaussian[m - 1].mean[0];
                nodeDev->gaussian[m].mean[1] = nodeDev->gaussian[m - 1].mean[1];
                nodeDev->gaussian[m].mean[2] = nodeDev->gaussian[m - 1].mean[2];
                nodeDev->gaussian[m].covariance = nodeDev->gaussian[m - 1].covariance;
                nodeDev->gaussian[m].weight = nodeDev->gaussian[m - 1].weight;
                nodeDev->gaussian[m - 1].mean[0] = t1;
                nodeDev->gaussian[m - 1].mean[1] = t2;
                nodeDev->gaussian[m - 1].mean[2] = t3;
                nodeDev->gaussian[m - 1].covariance = t4;
                nodeDev->gaussian[m - 1].weight = t5;
            }
        }

        out[y * outStep + x] = (uchar)background;
    // }

}



void processNode(cv::cuda::GpuMat &tmpImg, cv::cuda::GpuMat &outImg, float *nodeP, double cov, double alpha, double alpha_bar, double prune, double cfbar)
{
    assert(tmpImg.cols == outImg.cols && tmpImg.rows == outImg.rows);
    // std::cout << std::endl;
    const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(tmpImg.cols, blockDim.x), iDivUp(tmpImg.rows,blockDim.y));
    processKernel<<<gridDim, blockDim>>>(outImg.ptr<uchar>(), outImg.step, tmpImg.ptr<uchar>(), tmpImg.step, tmpImg.cols, tmpImg.rows, nodeP, sizeof(NodePixelGpu), cov, alpha, alpha_bar, prune, cfbar);

    cudaThreadSynchronize();

    // std::cout << std::endl;
}

__global__ void getImgKernel(uchar* imgGmm, int step, int h, int w, float* nodeP, int nodeStep)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
    if (x >= w || y >= h)
        return;

    int indexNode = x + y * w;
    char* data = (char*)nodeP + indexNode * nodeStep;
    NodePixelGpu *nodeDev = (NodePixelGpu *)(data);

    float b = nodeDev->gaussian[0].mean[0];
    float g = nodeDev->gaussian[0].mean[1];
    float r = nodeDev->gaussian[0].mean[2];


    imgGmm[y * step + x * 3] = static_cast<uchar>(b);
    imgGmm[y * step + x * 3 + 1] = static_cast<uchar>(g);
    imgGmm[y * step + x * 3 + 2] = static_cast<uchar>(r);

    // if(y == 450 && x == 650)
    // {
    //     printf("ttttttttt%d, %d, %f, %f, %f %f\n", x, y, nodeDev->gaussian[0].mean[0], nodeDev->gaussian[0].mean[1], nodeDev->gaussian[0].mean[2], nodeDev->realSize);
    //     printf("%d %d %d \n", (int)imgGmm[y * step + x * 3], (int)imgGmm[y * step + x * 3 + 1], (int)imgGmm[y * step + x * 3 + 2]);
    // }
}


void GetNode(cv::cuda::GpuMat &imgGmm, float *nodeP)
{
    // assert(imgGmm.cols == outImg.cols && imgGmm.rows == outImg.rows);
    const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(imgGmm.cols, blockDim.x), iDivUp(imgGmm.rows,blockDim.y));
    // std::cout << "imgGmm: " << imgGmm.cols << " " << imgGmm.rows << " " << std::endl;
    getImgKernel<<<gridDim, blockDim>>>(imgGmm.ptr<uchar>(), imgGmm.step, imgGmm.rows, imgGmm.cols, nodeP, sizeof(NodePixelGpu));
    cudaThreadSynchronize();
}

__global__ void processDiffKernel(uchar* img1, int img1step, int w, int h, uchar* img2, int img2step, uchar* src, int srcstep, uchar* result, int resultstep)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= w || y >= h)
        return;

    int b = img1[y * img1step + x * 3];
    int g = img1[y * img1step + x * 3 + 1];
    int r = img1[y * img1step + x * 3 + 2];

    int b1 = img2[y * img2step + x * 3];
    int g1 = img2[y * img2step + x * 3 + 1];
    int r1 = img2[y * img2step + x * 3 + 2];

    int diffb = abs(b1 - b);
    int diffg = abs(g1 - g);
    int diffr = abs(r1 - r);

    if(sqrt(float(diffb * diffb + diffr * diffr + diffg * diffg)) > 200)
    {
        result[y * resultstep + x * 3] = 255;
        result[y * resultstep + x * 3 + 1] = 255;
        result[y * resultstep + x * 3 + 2] = 255;
    }
}

void processDiff(cv::cuda::GpuMat &img1, cv::cuda::GpuMat& img2, cv::cuda::GpuMat& src, cv::cuda::GpuMat& result)
{
    assert(img1.cols == img2.cols && img1.rows == img2.rows);
    const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(img1.cols, blockDim.x), iDivUp(img1.rows,blockDim.y));
    processDiffKernel<<<gridDim, blockDim>>>(img1.ptr<uchar>(), img1.step, img1.cols, img1.rows, img2.ptr<uchar>(), img2.step, src.ptr<uchar>(), src.step, result.ptr<uchar>(), result.step);
    cudaThreadSynchronize();
}

__global__ void processDiffKernel(uchar* img1, uchar* img2, uchar* result, int w, int h, int binSize)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= w || y >= h)
        return;
    float sum = 0;
    float sum1 = 0;
    float sum2 = 0;
    for(int i = x; i < x + binSize; i ++)
    {
        for(int j = y; j < y + binSize; i++)
        {
            int b = img1[j * img1step + i * 3];
            int g = img1[j * img1step + i * 3 + 1];
            int r = img1[j * img1step + i * 3 + 2];

            int b1 = img2[j * img2step + i * 3];
            int g1 = img2[j * img2step + i * 3 + 1];
            int r1 = img2[j * img2step + i * 3 + 2];
        }
    }
}

void caculateSim(cv::Mat &img1, cv::Mat &img2, cv::Mat& result, int binSize)
{
    cv::cuda::GpuMat img1Gpu, img2Gpu, resultGpu;
    img1Gpu.upload(img1);
    img2Gpu.upload(img2);

    int sizeH = img1.rows / binSize;
    int sizeW = img1.cols / binSize;

    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(sizeW, blockDim.x), iDivUp(sizeH,blockDim.y));

    caculateSim<<<gridDim, blockDim>>>(img1Gpu.ptr<uchar>(), img2Gpu.ptr<uchar>(), resultGpu.ptr<uchar>(), sizeW, sizeH, binSize););

    resultGpu.download(result);
}



