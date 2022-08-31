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

    int index = nodeDev->realSize;
    
    nodeDev->gaussian[index].mean[0] = b;
    nodeDev->gaussian[index].mean[1] = g;
    nodeDev->gaussian[index].mean[2] = r;
    nodeDev->gaussian[index].covariance = cov;
    nodeDev->gaussian[index].weight = 1.0;
    nodeDev->realSize = nodeDev->realSize + 1;

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
    std::cout << std::endl;
     const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(tmpImg.cols, blockDim.x), iDivUp(tmpImg.rows,blockDim.y));
    processKernel<<<gridDim, blockDim>>>(outImg.ptr<uchar>(), outImg.step, tmpImg.ptr<uchar>(), tmpImg.step, tmpImg.cols, tmpImg.rows, nodeP, sizeof(NodePixelGpu), cov, alpha, alpha_bar, prune, cfbar);

    cudaThreadSynchronize();

    std::cout << std::endl;
}


//声明CUDA纹理
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex1;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex2;
//声明CUDA数组
cudaArray *cuArray1;
cudaArray *cuArray2;
//通道数
cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();


__global__ void weightAddKerkel(uchar *pDstImgData, int imgHeight, int imgWidth,int channels)
{
    const int tidx=blockDim.x*blockIdx.x+threadIdx.x;
    const int tidy=blockDim.y*blockIdx.y+threadIdx.y;

    if (tidx<imgWidth && tidy<imgHeight)
    {
        float4 lenaBGR,moonBGR;
        //使用tex2D函数采样纹理
        lenaBGR=tex2D(refTex1, tidx, tidy);
        moonBGR=tex2D(refTex2, tidx, tidy);

        int idx=(tidy*imgWidth+tidx)*channels;
        float alpha=0.5;
        pDstImgData[idx+0]=(alpha*lenaBGR.x+(1-alpha)*moonBGR.x)*255;
        pDstImgData[idx+1]=(alpha*lenaBGR.y+(1-alpha)*moonBGR.y)*255;
        pDstImgData[idx+2]=(alpha*lenaBGR.z+(1-alpha)*moonBGR.z)*255;
        pDstImgData[idx+3]=0;
    }
}

void test(uchar* input1, uchar* input2, int imgWidth, int imgHeight, int channels, cv::Mat& tmp)
{
    //设置纹理属性
    cudaError_t t;
    refTex1.addressMode[0] = cudaAddressModeClamp;
    refTex1.addressMode[1] = cudaAddressModeClamp;
    refTex1.normalized = false;
    refTex1.filterMode = cudaFilterModeLinear;
    //绑定cuArray到纹理
    cudaMallocArray(&cuArray1, &cuDesc, imgWidth, imgHeight);
    t = cudaBindTextureToArray(refTex1, cuArray1);

    refTex2.addressMode[0] = cudaAddressModeClamp;
    refTex2.addressMode[1] = cudaAddressModeClamp;
    refTex2.normalized = false;
    refTex2.filterMode = cudaFilterModeLinear;
     cudaMallocArray(&cuArray2, &cuDesc, imgWidth, imgHeight);
    t = cudaBindTextureToArray(refTex2, cuArray2);

    //拷贝数据到cudaArray
    t=cudaMemcpyToArray(cuArray1, 0,0, input1, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyHostToDevice);
    t=cudaMemcpyToArray(cuArray2, 0,0, input2, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyHostToDevice);

    //输出图像
    cv::Mat dstImg = cv::Mat(imgHeight, imgWidth, CV_8UC4, cv::Scalar::all(0));
    // cv::Mat::zeros(imgHeight, imgWidth, cv::CV_8UC4);
    uchar *pDstImgData=NULL;
    t=cudaMalloc(&pDstImgData, imgHeight*imgWidth*sizeof(uchar)*channels);

    //核函数，实现两幅图像加权和
    dim3 block(8,8);
    dim3 grid( (imgWidth+block.x-1)/block.x, (imgHeight+block.y-1)/block.y );
    weightAddKerkel<<<grid, block, 0>>>(pDstImgData, imgHeight, imgWidth, channels);
    cudaThreadSynchronize();

    //从GPU拷贝输出数据到CPU
    t=cudaMemcpy(dstImg.data, pDstImgData, imgWidth*imgHeight*sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
    tmp = dstImg.clone();
    // cv::imwrite("data/result.png", dstImg);
}




